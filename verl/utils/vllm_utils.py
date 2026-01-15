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
from vllm.lora.models import LoRAModel
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager


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
            supported_lora_modules = self._adapter_manager.supported_lora_modules
            packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            expected_lora_modules: List[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_modules.extend(packed_modules_mapping[module])
                else:
                    expected_lora_modules.append(module)

            expected_lora_modules = list(set(expected_lora_modules))

            lora_tensors = None
            from vllm.lora.peft_helper import PEFTHelper

            if isinstance(lora_request, TensorLoRARequest):
                peft_config = lora_request.peft_config
                lora_tensors = lora_request.lora_tensors
                peft_helper = PEFTHelper.from_dict(peft_config)
            else:
                lora_path = get_adapter_absolute_path(lora_request.lora_path)

                peft_helper = PEFTHelper.from_local_dir(lora_path, self.max_position_embeddings)

            # Validates the LoRA configuration against requirements before
            # loading weights, throwing an exception if validation fails.
            peft_helper.validate_legal(self.lora_config)

            # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
            # to ensure correct loading of lora weights.
            model = self._adapter_manager.model
            hf_to_vllm_mapper = None
            if hasattr(model, "hf_to_vllm_mapper") and model.hf_to_vllm_mapper is not None:
                hf_to_vllm_mapper = model.hf_to_vllm_mapper

            if isinstance(lora_request, TensorLoRARequest):
                lora = self._lora_model_cls.from_lora_tensors(
                    lora_model_id=lora_request.lora_int_id,
                    tensors=lora_tensors,
                    peft_helper=peft_helper,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    embeddings=None,
                    target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=hf_to_vllm_mapper,
                )
            else:
                lora = self._lora_model_cls.from_local_checkpoint(
                    lora_path,
                    expected_lora_modules,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=hf_to_vllm_mapper,
                )

            if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
                raise ValueError(
                    f"LoRA added vocab size {lora.extra_vocab_size} "
                    f"is greater than lora_extra_vocab_size "
                    f"{self.lora_config.lora_extra_vocab_size}."
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
