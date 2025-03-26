# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Contain small torch utilities
"""

from typing import List, Literal, Optional, Union

import torch
import torch.distributed
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


@torch.compiler.disable()
def log_probs_from_logits_flash_attn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    output = cross_entropy_loss(logits, labels, inplace_backward=True)
    if not isinstance(output, tuple):
        raise ValueError(
            "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
        )

    return -output[0]


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log probs on the label ids given logits.

    We may use torch compile to speed up computing.

    Args:
        logits (torch.Tensor): logits of the model, shape (batch_size, seqlen, vocab_size)
        labels (torch.Tensor): labels of the model, shape (batch_size, seqlen)

    Returns:
        torch.Tensor: log probs of the labels, shape (batch_size, seqlen)
    """
    batch_dim = logits.shape[:-1]
    vocab_dim = logits.shape[-1]
    logits = logits.contiguous().view(-1, vocab_dim)
    labels = labels.contiguous().view(-1)
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        output = log_probs_from_logits_flash_attn(logits, labels)
    else:  # fall back to torch kernel, upcast logits to fp32
        output = F.cross_entropy(logits.float(), labels, reduction="none")

    return output.view(*batch_dim)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(dim=dim) / mask.sum(dim=dim)


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum <= 1:
            raise ValueError("The sum of the mask is less than one, which can cause a division by zero.")

        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction

    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    return (values - mean) * torch.rsqrt(var + 1e-8)


def get_eos_mask(response_ids: torch.Tensor, eos_token_id: Union[int, List[int]] = 2, dtype: torch.dtype = torch.long):
    """Get the mask for the response ids, the mask will be 0 after the first eos token.

    eos_token_id can be int or list: 1 or [1, 2].
    ```
    e.g. eos_token = 1
    response_ids: [0, 0, 2, 4, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1, 1, 1, 1, 0, 0]
    ```
    """
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    eos_mask = torch.zeros_like(response_ids, dtype=torch.bool)
    for token_id in eos_token_id:
        eos_mask |= response_ids.eq(token_id)

    eos_mask = eos_mask.long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask


def pad_2d_list_to_length(
    response: List[List[int]], pad_token_id: int, max_length: Optional[int] = None
) -> torch.Tensor:
    """Pad a 2D list (e.g. responses, log_probs) to a 2D tensor."""
    max_response_length = max(len(sub_list) for sub_list in response)
    if max_length is not None and max_length > max_response_length:
        target_length = max_length
    else:
        target_length = max_response_length

    padded_response = [tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list)) for sub_list in response]
    tensor = torch.tensor(padded_response)
    return tensor


def pad_sequence_to_length(
    tensor: torch.Tensor, max_seq_len: int, pad_token_id: int, left_pad: bool = False
) -> torch.Tensor:
    """Pad a nD tensors in the last dim to max_seq_len."""
    if tensor.size(-1) >= max_seq_len:
        return tensor

    pad_shape = list(tensor.shape)
    pad_shape[-1] = max_seq_len - tensor.size(-1)
    pad_tensor = torch.full(pad_shape, fill_value=pad_token_id, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((pad_tensor, tensor), dim=-1) if left_pad else torch.cat((tensor, pad_tensor), dim=-1)


def postprocess_data(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    left_pad: bool = True,
    truncation: Literal["left", "right", "error"] = "error",
):
    """Pad or truncate data."""
    assert truncation in ["left", "right", "error"]
    seq_length = len(input_ids)
    if seq_length < max_length:
        input_ids = pad_sequence_to_length(
            input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
        position_ids = pad_sequence_to_length(position_ids, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad)
    elif seq_length > max_length:
        if truncation == "left":  # actually, left truncation may not be reasonable
            input_ids = input_ids[..., -max_length:]
            attention_mask = attention_mask[..., -max_length:]
            position_ids = position_ids[..., -max_length:]
        elif truncation == "right":
            input_ids = input_ids[..., :max_length]
            attention_mask = attention_mask[..., :max_length]
            position_ids = position_ids[..., :max_length]
        elif truncation == "error":
            raise NotImplementedError(f"{seq_length} is larger than {max_length}.")
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}.")

    return input_ids, attention_mask, position_ids


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Get the lr scheduler for constant lr."""

    def lr_lambda(current_step: int) -> float:
        return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
