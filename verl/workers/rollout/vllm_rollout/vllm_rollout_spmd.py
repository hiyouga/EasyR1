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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
"""

from contextlib import contextmanager
from typing import Any, List, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams

from ....protocol import DataProto
from ....utils import torch_functional as VF
from ....utils.torch_dtypes import PrecisionType
from ..base import BaseRollout
from ..config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if not config.enforce_eager and config.free_cache_engine:
            raise ValueError("CUDA graph should be disabled when `free_cache_engine` is True.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        vllm_init_kwargs = {}
        if config.limit_images > 0:
            vllm_init_kwargs = {"limit_mm_per_prompt": {"image": config.limit_images}}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            max_model_len=config.prompt_length + config.response_length,
            max_num_batched_tokens=config.max_num_batched_tokens,
            enable_sleep_mode=True,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=True,
            disable_log_stats=config.disable_log_stats,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **vllm_init_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {"max_tokens": config.response_length, "detokenize": False}
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        do_sample = prompts.meta_info.get("do_sample", True)
        if not do_sample:
            kwargs = {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
            }

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params
            )

        response_ids = []
        for completion in completions:
            for output in completion.outputs:
                response_ids.append(output.token_ids)

        response_ids = VF.pad_2d_list_to_length(
            response_ids, self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)

        if self.config.n > 1 and do_sample:
            batch_size = batch_size * self.config.n
            input_ids = _repeat_interleave(input_ids, self.config.n)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            if "multi_modal_inputs" in non_tensor_batch.keys():
                non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                    non_tensor_batch["multi_modal_inputs"], self.config.n
                )

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = VF.get_eos_mask(
            response_ids=response_ids, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
