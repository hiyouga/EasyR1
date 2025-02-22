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


# Supported models using HF Rmpad
# TODO(sgm): HF may supported more than listed here, we should add more after testing
from transformers import GemmaConfig, LlamaConfig, MistralConfig, Qwen2Config


_REOVEPAD_MODELS = {"llama": LlamaConfig, "mistral": MistralConfig, "gemma": GemmaConfig, "qwen2": Qwen2Config}


def check_model_support_rmpad(model_type: str):
    assert isinstance(model_type, str)
    if model_type not in _REOVEPAD_MODELS.keys():
        raise ValueError(
            f"Model architecture {model_type} is not supported for now. "
            f"RMPad supported architectures: {_REOVEPAD_MODELS.keys()}."
            f"Please set `use_remove_padding=False` in the model config."
        )
