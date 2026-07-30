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

from importlib.metadata import version
from typing import List

from msgspec import field
from packaging import version as vs
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

try:
    from vllm.lora.lora_model import LoRAModel  # vLLM >= 0.12
except ImportError:  # older vLLM
    from vllm.lora.models import LoRAModel


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class VLLMHijack:
    @staticmethod
    def hijack():
        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            """
            based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors
            Reason:
            VLLM does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
            To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to load memory-based LoRA tensors.
            """
            from vllm.lora.peft_helper import PEFTHelper

            supported_lora_modules = self._adapter_manager.supported_lora_modules
            packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            expected_lora_lst: List[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_lst.extend(packed_modules_mapping[module])
                else:
                    expected_lora_lst.append(module)
                if module == "experts":
                    expected_lora_lst.append(module)
            expected_lora_modules = set(expected_lora_lst)

            if isinstance(lora_request, TensorLoRARequest):
                peft_helper = PEFTHelper.from_dict(lora_request.peft_config)
                lora_tensors = lora_request.lora_tensors
                lora_path = None
            else:
                lora_path = get_adapter_absolute_path(lora_request.lora_path)
                peft_helper = PEFTHelper.from_local_dir(
                    lora_path,
                    self.max_position_embeddings,
                    getattr(lora_request, "tensorizer_config_dict", None),
                )
                lora_tensors = None

            # Validates the LoRA configuration against requirements before
            # loading weights, throwing an exception if validation fails.
            peft_helper.validate_legal(self.lora_config)

            # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
            # to ensure correct loading of lora weights.
            model = self._adapter_manager.model
            hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
            lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)

            if isinstance(lora_request, TensorLoRARequest):
                lora = self._lora_model_cls.from_lora_tensors(
                    lora_model_id=lora_request.lora_int_id,
                    tensors=lora_tensors,
                    peft_helper=peft_helper,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    model_vocab_size=self.vocab_size,
                    weights_mapper=hf_to_vllm_mapper,
                    skip_prefixes=lora_skip_prefixes,
                )
            else:
                lora = self._lora_model_cls.from_local_checkpoint(
                    lora_path,
                    expected_lora_modules,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    model_vocab_size=self.vocab_size,
                    tensorizer_config_dict=getattr(lora_request, "tensorizer_config_dict", None),
                    weights_mapper=hf_to_vllm_mapper,
                    skip_prefixes=lora_skip_prefixes,
                )

            return lora

        setattr(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)

        if vs.parse(version("vllm")).base_version == "0.11.0":
            from vllm.model_executor.models.module_mapping import MultiModelKeys
            from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

            def hijack__get_mm_mapping(self) -> MultiModelKeys:
                """
                Patch vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration.get_mm_mapping in vLLM 0.11.0
                Reason:
                vLLM 0.11.0 uses "model.visual.*" prefixes for Qwen3-VL, but the real module names are "visual.*".
                This breaks LoRA filtering for multimodal parts, so we align the prefixes to the real module names.
                Fixed upstream: https://github.com/vllm-project/vllm/commit/9f4e309
                """
                return MultiModelKeys.from_string_field(
                    language_model="language_model",
                    connector="visual.merger.",
                    tower_model="visual.",
                )

            setattr(Qwen3VLForConditionalGeneration, "get_mm_mapping", hijack__get_mm_mapping)
