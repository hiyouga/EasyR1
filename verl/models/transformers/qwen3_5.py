# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Adapted from:
# https://github.com/huggingface/transformers/blob/v5.4.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
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
Standalone position ID computation for Qwen3.5-VL.

Qwen3.5 uses mRoPE like Qwen2/3-VL but with key differences:
- Uses mm_token_type_ids (0=text, 1=image, 2=video) instead of scanning for special tokens
- Position advances by max(H, W) / spatial_merge_size after each vision block (not T*H*W)
- get_vision_position_ids computes spatial positions with start_position offset
"""

import itertools
from typing import Optional

import torch
from transformers import ProcessorMixin


def get_vision_position_ids(
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute 3D positional indices for vision tokens from a single image or video.

    Args:
        start_position: Offset added to all computed positional indices.
        grid_thw: Tensor of shape (3,) — (T, H, W) grid of the vision feature.
        spatial_merge_size: Factor by which H and W are reduced in the backbone.
        device: Device for the output tensor.

    Returns:
        torch.LongTensor of shape (3, sequence_length): [temporal, height, width] positions.
    """
    llm_grid_t = grid_thw[0].item()
    llm_grid_h = grid_thw[1].item() // spatial_merge_size
    llm_grid_w = grid_thw[2].item() // spatial_merge_size

    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
        llm_grid_h * llm_grid_t
    )
    position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
        llm_grid_w * llm_grid_t
    )
    position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)

    return torch.stack([position_temporal, position_height, position_width], dim=0)


def get_rope_index(
    processor: "ProcessorMixin",
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Compute mRoPE position IDs for Qwen3.5, adapted from Qwen3_5Model.get_rope_index.

    This is a standalone (non-method) version that works on single (unbatched) samples,
    matching the interface used in EasyR1's dataset.py.

    Args:
        processor: The Qwen3.5 processor (used to get spatial_merge_size).
        input_ids: 1D tensor of token IDs (seq_length,).
        mm_token_type_ids: 1D tensor (seq_length,) — 0=text, 1=image, 2=video.
        image_grid_thw: Tensor of shape (num_images, 3) or None.
        video_grid_thw: Tensor of shape (num_videos, 3) or None.
        attention_mask: 1D tensor (seq_length,) or None.

    Returns:
        torch.Tensor of shape (3, seq_length): mRoPE position IDs [temporal, height, width].
    """
    # Qwen3.5 splits video_grid_thw by temporal dimension (timestamps separate frames)
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    spatial_merge_size = processor.image_processor.merge_size

    position_ids = torch.zeros(3, len(input_ids), dtype=input_ids.dtype, device=input_ids.device)

    if attention_mask is not None:
        mask = attention_mask.bool()
        input_ids_masked = input_ids[mask]
        mm_types_masked = mm_token_type_ids[mask]
    else:
        input_ids_masked = input_ids
        mm_types_masked = mm_token_type_ids

    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    # Group consecutive tokens by modality type
    input_type_group = []
    for key, group in itertools.groupby(enumerate(mm_types_masked.tolist()), lambda x: x[1]):
        group = list(group)
        start_index = group[0][0]
        end_index = group[-1][0] + 1
        input_type_group.append((key, start_index, end_index))

    current_pos = 0
    llm_pos_ids_list = []
    for modality_type, start_idx, end_idx in input_type_group:
        if modality_type == 0:  # text
            text_len = end_idx - start_idx
            llm_pos_ids_list.append(
                torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
            )
            current_pos += text_len
        else:  # image (1) or video (2)
            grid_thw = next(grid_iters[modality_type])
            vision_pos = get_vision_position_ids(
                current_pos, grid_thw, spatial_merge_size, device=input_ids.device
            )
            llm_pos_ids_list.append(vision_pos)
            # Qwen3.5 advances position by max(H, W) / spatial_merge_size (NOT T*H*W)
            current_pos += max(grid_thw[1].item(), grid_thw[2].item()) // spatial_merge_size

    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    if attention_mask is not None:
        position_ids[..., attention_mask.bool()] = llm_positions.to(position_ids.device)
    else:
        position_ids = llm_positions.to(position_ids.device)

    return position_ids
