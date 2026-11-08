# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
# Based on:
# https://github.com/huggingface/transformers/blob/2f6249179233e880770c1dbc47ea834ada29fe59/src/transformers/models/qwen3_5/modeling_qwen3_5.py
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

import itertools
from typing import Optional

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
)


def _get_vision_position_ids(
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    device: torch.device,
    temp_merge_size: int = 1,
    time_interval: int = 1,
) -> torch.Tensor:
    llm_grid_t, llm_grid_h, llm_grid_w = (
        grid_thw[0].item() // temp_merge_size,
        grid_thw[1].item() // spatial_merge_size,
        grid_thw[2].item() // spatial_merge_size,
    )
    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
        llm_grid_h * llm_grid_t
    )
    position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
        llm_grid_w * llm_grid_t
    )
    position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
    position_temporal = position_temporal * time_interval
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    M-RoPE position ids for Qwen3.5 (VL), shape (3, sequence_length) for one example (no batch dim).

    Aligns with ``Qwen3_5Model.get_rope_index`` in HuggingFace transformers so Ulysses / FSDP can shard the
    sequence consistently before the forward pass.

    Requires ``mm_token_type_ids`` from the processor (text=0, image=1, video=2).
    """
    spatial_merge_size = processor.image_processor.merge_size
    if input_ids.ndim != 1 or mm_token_type_ids.ndim != 1:
        raise ValueError("get_rope_index expects 1D input_ids and mm_token_type_ids per example.")

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    position_ids = torch.zeros(3, input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    current_input_ids = input_ids[attention_mask == 1]
    input_token_type = mm_token_type_ids[attention_mask == 1]

    input_type_group = []
    for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
        group = list(group)
        start_index = group[0][0]
        end_index = group[-1][0] + 1
        input_type_group.append((key, start_index, end_index))

    current_pos = 0
    llm_pos_ids_list = []
    for modality_type, start_idx, end_idx in input_type_group:
        if modality_type == 0:
            text_len = end_idx - start_idx
            llm_pos_ids_list.append(
                torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
            )
            current_pos += text_len
        else:
            grid_thw = next(grid_iters[modality_type])
            vision_position_ids = _get_vision_position_ids(
                current_pos, grid_thw, spatial_merge_size, input_ids.device
            )
            llm_pos_ids_list.append(vision_position_ids)
            current_pos += max(grid_thw[1].item(), grid_thw[2].item()) // spatial_merge_size

    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    position_ids[:, attention_mask == 1] = llm_positions.to(position_ids.device)
    return position_ids


def _vision_pooler_output(
    model: "Qwen3_5Model",
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    out = model.visual(pixel_values, grid_thw=grid_thw, return_dict=True)
    merged = out.pooler_output
    split_sizes = (grid_thw.prod(-1) // model.visual.spatial_merge_size**2).tolist()
    return torch.cat(torch.split(merged, split_sizes), dim=0)


def _get_input_embeds(
    model: "Qwen3_5Model",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = _vision_pooler_output(model, pixel_values, image_grid_thw)
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds = _vision_pooler_output(model, pixel_values_videos, video_grid_thw)
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, video_embeds)

    if pixel_values is None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds = _vision_pooler_output(model, pixel_values, image_grid_thw)
        inputs_embeds += 0.0 * image_embeds.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }


def qwen3_5_base_forward(
    self: "Qwen3_5Model",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    input_kwargs = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )
    kwargs.update(input_kwargs)
    outputs = self.language_model(input_ids=None, **kwargs)
    return Qwen3_5ModelOutputWithPast(last_hidden_state=outputs.last_hidden_state)


def qwen3_5_model_forward(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor,
    labels: Optional[torch.LongTensor] = None,
    **kwargs,
) -> "Qwen3_5CausalLMOutputWithPast":
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    return Qwen3_5CausalLMOutputWithPast(logits=logits)
