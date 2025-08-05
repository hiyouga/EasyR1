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


from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..utils.py_functional import is_transformers_version_greater_than
from .transformers.flash_attention_utils import flash_attention_forward
from .transformers.qwen2_vl import (
    qwen2_vl_attn_forward,
    qwen2_vl_base_forward_new,
    qwen2_vl_forward_new,
    qwen2_vl_forward_old,
)


def apply_ulysses_patch(model_type: str) -> None:
    if model_type in ("llama", "gemma", "gemma2", "mistral", "qwen2", "qwen3", "qwen3_moe"):
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
    elif model_type in ("qwen2_vl", "qwen2_5_vl"):
        if is_transformers_version_greater_than("4.54.0"):
            # transformers 4.54.0 does not need special patch: https://github.com/huggingface/transformers/pull/39447
            ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
        elif is_transformers_version_greater_than("4.53.0"):
            raise NotImplementedError("Transformers 4.53.* is not compatible with Qwen2-VL. Use 4.54.0 or later.")
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2

            Qwen2VLFlashAttention2.forward = qwen2_vl_attn_forward
            Qwen2_5_VLFlashAttention2.forward = qwen2_vl_attn_forward

        if is_transformers_version_greater_than("4.52.0"):
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration,
                Qwen2_5_VLModel,
            )
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel

            Qwen2VLModel.forward = qwen2_vl_base_forward_new
            Qwen2_5_VLModel.forward = qwen2_vl_base_forward_new
            Qwen2VLForConditionalGeneration.forward = qwen2_vl_forward_new
            Qwen2_5_VLForConditionalGeneration.forward = qwen2_vl_forward_new
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

            Qwen2VLForConditionalGeneration.forward = qwen2_vl_forward_old
            Qwen2_5_VLForConditionalGeneration.forward = qwen2_vl_forward_old
    else:
        raise NotImplementedError(f"Model architecture {model_type} is not supported yet.")
