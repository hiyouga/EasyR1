# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright Meta Platforms, Inc. and affiliates
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

from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.distributed
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .torch_dtypes import PrecisionType


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


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = None, eps: float = 1e-8) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(dim=dim) / (mask.sum(dim=dim) + eps)


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum <= 1:
            print("The sum of the mask is less than one, which can cause a division by zero.")
            return variance

        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction

    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    return (values - mean) * torch.rsqrt(var + eps)


def get_response_mask(
    response_ids: torch.Tensor, eos_token_id: Union[int, List[int]] = 2, dtype: torch.dtype = torch.long
):
    """Get the mask for the response ids, the mask will be 0 after the first eos token.

    eos_token_id can be int or list: 1 or [1, 2].
    ```
    e.g. eos_token = 1
    response_ids:  [0, 0, 2, 4, 3, 5, 1, 0, 0]
    response_mask: [1, 1, 1, 1, 1, 1, 1, 0, 0]
    ```
    """
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    response_mask = torch.zeros_like(response_ids, dtype=torch.bool)
    for token_id in eos_token_id:
        response_mask |= response_ids.eq(token_id)

    response_mask = response_mask.long()
    response_mask = (torch.cumsum(response_mask, dim=1) - response_mask).bool()
    response_mask = torch.logical_not(response_mask).to(dtype)
    return response_mask


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
            raise RuntimeError(f"Input sequence length {seq_length} is longer than max length {max_length}.")
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


# https://github.com/meta-llama/llama-cookbook/blob/v0.0.5/src/llama_cookbook/policies/anyprecision_optimizer.py
class AnyPrecisionAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_kahan_summation: bool = True,
        momentum_dtype: str = "bfloat16",
        variance_dtype: str = "bfloat16",
        compensation_buffer_dtype: str = "bfloat16",
    ):
        """
        AnyPrecisionAdamW: a flexible precision AdamW optimizer
        with optional Kahan summation for high precision weight updates.
        Allows direct control over momentum, variance and auxiliary compensation buffer dtypes.
        Optional Kahan summation is used to offset precision reduction for the weight updates.
        This allows full training in BFloat16 (equal or better than FP32 results in many cases)
        due to high precision weight updates.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay coefficient (default: 1e-2)

            # Any Precision specific
            use_kahan_summation = creates auxiliary buffer to ensure high precision
            model param updates (default: False)
            momentum_dtype = dtype for momentum  (default: bfloat16)
            variance_dtype = dtype for uncentered variance (default: bfloat16)
            compensation_buffer_dtype = dtype for Kahan summation buffer (default: bfloat16)

            # Usage
            This optimizer implements optimizer states, and Kahan summation
            for high precision updates, all in user controlled dtypes.
            Defaults are variance in BF16, Momentum in FP32.
            This can be run in FSDP mixed precision, amp, or full precision,
            depending on what training pipeline you wish to work with.

            Setting to use_kahan_summation = False, and changing momentum and
            variance dtypes to FP32, reverts this to a standard AdamW optimizer.

        """
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_kahan_summation": use_kahan_summation,
            "momentum_dtype": momentum_dtype,
            "variance_dtype": variance_dtype,
            "compensation_buffer_dtype": compensation_buffer_dtype,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = PrecisionType.to_dtype(group["momentum_dtype"])
            variance_dtype = PrecisionType.to_dtype(group["variance_dtype"])
            compensation_buffer_dtype = PrecisionType.to_dtype(group["compensation_buffer_dtype"])
            for p in group["params"]:
                assert isinstance(p, torch.Tensor)  # lint
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("AnyPrecisionAdamW does not support sparse gradients.")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(p, dtype=compensation_buffer_dtype)

                # Main processing
                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad

                if weight_decay:  # weight decay, AdamW style
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # update momentum
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # update uncentered variance

                bias_correction1 = 1 - beta1**step  # adjust using bias1
                step_size = lr / bias_correction1

                denom_correction = (1 - beta2**step) ** 0.5  # adjust using bias2 and avoids math import
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(eps, alpha=1)

                if use_kahan_summation:  # lr update to compensation
                    compensation = state["compensation"]
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))
                else:  # usual AdamW updates
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)
