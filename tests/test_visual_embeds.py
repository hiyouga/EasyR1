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

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from verl.models.transformers import qwen2_vl, qwen3_vl


HIDDEN_SIZE = 64
IMAGE_TOKEN_ID, VIDEO_TOKEN_ID = 10, 11


def _build_visual(name: str) -> tuple[torch.nn.Module, int]:
    if name == "qwen2_vl":
        config = Qwen2VLVisionConfig(depth=2, embed_dim=32, hidden_size=HIDDEN_SIZE, num_heads=2, mlp_ratio=2)
        visual_cls = Qwen2VisionTransformerPretrainedModel
    elif name == "qwen2_5_vl":
        config = Qwen2_5_VLVisionConfig(
            depth=2,
            hidden_size=32,
            out_hidden_size=HIDDEN_SIZE,
            intermediate_size=64,
            num_heads=2,
            window_size=56,
            fullatt_block_indexes=[1],
        )
        visual_cls = Qwen2_5_VisionTransformerPretrainedModel
    else:
        config = Qwen3VLVisionConfig(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=HIDDEN_SIZE,
            num_position_embeddings=64,
            deepstack_visual_indexes=[0, 1],
        )
        visual_cls = Qwen3VLVisionModel

    config._attn_implementation = "eager"
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
    return visual_cls(config).eval(), patch_dim


def _inputs(patch_dim: int) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.randn(16, patch_dim),  # one image, grid (1, 4, 4), 4 tokens
        "pixel_values_videos": torch.randn(32, patch_dim),  # one video, grid (2, 4, 4), 8 tokens
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "video_grid_thw": torch.tensor([[2, 4, 4]]),
    }


@pytest.mark.parametrize("name", ["qwen2_vl", "qwen2_5_vl", "qwen3_vl"])
@torch.no_grad()
def test_image_and_video_share_one_vision_tower_call(name: str):
    torch.manual_seed(0)
    visual, patch_dim = _build_visual(name)
    inputs = _inputs(patch_dim)
    expected_image = visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    expected_video = visual(inputs["pixel_values_videos"], grid_thw=inputs["video_grid_thw"])

    calls = []
    visual.register_forward_hook(lambda *args: calls.append(None))
    module = qwen3_vl if name == "qwen3_vl" else qwen2_vl
    image_outputs, video_outputs = module._get_visual_embeds(SimpleNamespace(visual=visual), **inputs)
    assert len(calls) == 1
    torch.testing.assert_close(image_outputs, expected_image)
    torch.testing.assert_close(video_outputs, expected_video)


@pytest.mark.parametrize("name", ["qwen2_vl", "qwen2_5_vl", "qwen3_vl"])
@torch.no_grad()
def test_mixed_batch_input_embeds(name: str):
    torch.manual_seed(0)
    visual, patch_dim = _build_visual(name)
    inputs = _inputs(patch_dim)
    model = SimpleNamespace(
        visual=visual,
        config=SimpleNamespace(image_token_id=IMAGE_TOKEN_ID, video_token_id=VIDEO_TOKEN_ID),
        get_input_embeddings=lambda: torch.nn.Embedding(16, HIDDEN_SIZE),
    )
    input_ids = torch.tensor([[1] + [IMAGE_TOKEN_ID] * 4 + [2] + [VIDEO_TOKEN_ID] * 8 + [3]])
    expected_image = visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    expected_video = visual(inputs["pixel_values_videos"], grid_thw=inputs["video_grid_thw"])
    if name == "qwen3_vl":
        expected_image, expected_video = expected_image[0], expected_video[0]

    calls = []
    visual.register_forward_hook(lambda *args: calls.append(None))
    module = qwen3_vl if name == "qwen3_vl" else qwen2_vl
    inputs_embeds = module._get_input_embeds(model, input_ids, **inputs)["inputs_embeds"][0]
    assert len(calls) == 1
    torch.testing.assert_close(inputs_embeds[1:5], expected_image)
    torch.testing.assert_close(inputs_embeds[6:14], expected_video)
