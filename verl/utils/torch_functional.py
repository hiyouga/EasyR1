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

import math
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.contiguous().view(-1, last_dim)
        labels = labels.contiguous().view(-1)
        output = logprobs_from_logits_flash_attn(logits, labels)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    output = cross_entropy_loss(logits, labels)
    assert isinstance(output, tuple), (
        "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    )
    return -output[0]


def logprobs_from_logits_v2(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    A memory efficient implementation of logprobs_from_logits
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_logprobs = F.log_softmax(row_logits.float(), dim=-1)
            row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)

        logprobs_labels = torch.stack(logprobs_labels)

    return logprobs_labels


def clip_by_value(tensor: torch.Tensor, tensor_min: torch.Tensor, tensor_max: torch.Tensor) -> torch.Tensor:
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(tensor, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy from logits."""
    prob_dist = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
    return entropy


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: int = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def get_eos_mask(response_ids: torch.Tensor, eos_token: Union[int, List[int]] = 2, dtype: torch.dtype = torch.int64):
    """
    end of sentence token can be int or list: 1 or [1, 2]
    e.g. eos_token=1
    response_ids: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    """
    if isinstance(eos_token, int):
        eos_token = [eos_token]

    eos_mask = torch.zeros_like(response_ids, dtype=torch.bool)
    for token in eos_token:
        eos_mask |= response_ids.eq(token)

    eos_mask = eos_mask.long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask


def pad_2d_list_to_length(
    response: List[List[int]], pad_token_id: int, max_length: Optional[int] = None
) -> torch.Tensor:
    """
    pad a 2D list (e.g. responses, logprobs) to a 2D tensor.
    """
    response_length = max(len(sub_list) for sub_list in response)
    if max_length is not None and max_length > response_length:
        target_length = max_length
    else:
        target_length = response_length
    padded_response = [tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list)) for sub_list in response]
    tensor = torch.tensor(padded_response)
    return tensor


def pad_sequence_to_length(
    tensor: torch.Tensor, max_seq_len: int, pad_token_id: int, left_pad: bool = False
) -> torch.Tensor:
    """
    Pad a nD tensors in the last dim to max_seq_len.
    """
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
    """
    Pad or truncate data.
    """
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


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(0.0, x * coef + intercept)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        return min(1, float(current_step) / float(max(1, num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def allgather_dict_tensors(tensors: Union[Dict[str, torch.Tensor], TensorDict], size, group, dim=0):
    """
    TODO: optimize this.
    - We can use async ops
    - We can use only one allgather
    Args:
        tensors:
        size:
        group:

    Returns:

    """
    if isinstance(tensors, TensorDict):
        is_tensor_dict = True
        tensors_as_dict = tensors.to_dict()
    else:
        tensors_as_dict = tensors
        is_tensor_dict = False

    output = {}
    sorted_keys = sorted(tensors_as_dict.keys())
    for key in sorted_keys:
        val = tensors_as_dict[key]
        output[key] = [torch.empty_like(val) for _ in range(size)]
        torch.distributed.all_gather(output[key], val, group=group, async_op=False)
        output[key] = torch.cat(output[key], dim=dim)

    if is_tensor_dict:
        output = TensorDict(source=output, batch_size=tensors.batch_size[0] * size)

    return output
