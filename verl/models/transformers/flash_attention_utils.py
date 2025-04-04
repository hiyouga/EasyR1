# Copyright 2024 The Fairseq Authors and the HuggingFace Inc. team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Based on https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/modeling_flash_attention_utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from transformers.modeling_flash_attention_utils import _flash_attention_forward, fa_peft_integration_check
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

from ...utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    _flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    _flash_deterministic_enabled = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
    _flash_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


def prepare_fa2_from_position_ids(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, position_ids: torch.Tensor
):
    query = query.view(-1, query.size(-2), query.size(-1))
    key = key.contiguous().view(-1, key.size(-2), key.size(-1))
    value = value.contiguous().view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)
    cu_seqlens = torch.cat(
        (
            indices_q[position_ids == 0],
            torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
        )
    )
    max_length = cu_seqlens.diff().max()  # use cu_seqlens to infer max_length for qwen2vl mrope
    return (query, key, value, indices_q, (cu_seqlens, cu_seqlens), (max_length, max_length))


def _custom_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool = True,
    position_ids: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    deterministic: Optional[bool] = None,
    **kwargs,
):
    """
    Patches flash attention forward to handle 3D position ids in mrope. (3, batch_size, seq_length)
    """
    if not use_top_left_mask:
        causal = is_causal
    else:
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if _flash_supports_deterministic:
        flash_kwargs["deterministic"] = deterministic if deterministic is not None else _flash_deterministic_enabled

    if kwargs.get("softcap") is not None:
        flash_kwargs["softcap"] = kwargs.pop("softcap")

    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype=torch.bfloat16
    )

    sp_size = get_ulysses_sequence_parallel_world_size()
    if sp_size > 1:
        # (batch_size, seq_length, num_head, head_size)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)
        position_ids_lst = [torch.empty_like(position_ids) for _ in range(sp_size)]
        position_ids = dist.all_gather(position_ids_lst, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.cat(position_ids_lst, dim=-1)  # (..., batch_size, seq_length)

    if position_ids is not None and position_ids.dim() == 3:  # qwen2vl mrope
        position_ids = position_ids[0]

    if position_ids is not None and query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all():
        batch_size = query_states.size(0)
        query_states, key_states, value_states, _, cu_seq_lens, max_seq_lens = prepare_fa2_from_position_ids(
            query_states, key_states, value_states, position_ids
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=kwargs.pop("dropout", 0.0),
            softmax_scale=kwargs.pop("softmax_scale", None),
            causal=causal,
            **flash_kwargs,
        )
        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))
    else:
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal=is_causal,
            sliding_window=sliding_window,
            use_top_left_mask=use_top_left_mask,
            deterministic=deterministic,
            **kwargs,
        )  # do not pass position_ids to old flash_attention_forward

    if sp_size > 1:
        # (batch_size, seq_length, num_head, head_size)
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    return attn_output


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # This is before the transpose
    q_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    attn_output = _custom_flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=q_len,
        is_causal=True,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_flash_use_top_left_mask,
        **kwargs,
    )

    return attn_output, None
