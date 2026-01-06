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

import inspect
import re
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Iterable, Union

import torch
import torch.distributed as dist
from peft import PeftModel
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel
from vllm import LLM
from vllm.distributed import parallel_state as vllm_ps

from ...protocol import DataProto, all_gather_data_proto
from ...utils.fsdp_utils import fsdp_version, load_fsdp_model, offload_fsdp_model
from ...utils.model_utils import print_gpu_memory_usage
from ...utils.vllm_utils import TensorLoRARequest, VLLMHijack, is_version_ge
from .base import BaseShardingManager


class FSDPVLLMShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        device_mesh: DeviceMesh,
        use_param_offload: bool,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.use_param_offload = use_param_offload
        self.loaded = False
        self.base_sync_done = False

        self.world_size = dist.get_world_size()
        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group

        # Record freed bytes to estimate memory usage correctly
        # https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
        self.freed_bytes = 0

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        gen_dp_rank = self.device_mesh["dp"].get_local_rank()
        torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.torch_random_states)

        if is_version_ge(pkg="vllm", minver="0.7.3"):
            VLLMHijack.hijack()

    def _rename_weight_keys(self, actor_weights: dict[str, Union[torch.Tensor, DTensor]], model: PreTrainedModel):
        # convert state dict keys: https://github.com/huggingface/transformers/pull/38385
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return actor_weights

        reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
        original_weights = {}
        for key, value in actor_weights.items():
            for pattern, replacement in reverse_key_mapping.items():
                replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
                replacement = re.sub(r"\(.*\)", "", replacement)
                key, n_replace = re.subn(pattern, replacement, key)
                # Early exit of the loop
                if n_replace > 0:
                    break

            original_weights[key] = value

        return original_weights

    def _make_weight_iterator(
        self, actor_weights: dict[str, Union[torch.Tensor, DTensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        device = torch.device("cuda", torch.cuda.current_device())
        for name, tensor in actor_weights.items():
            yield name, tensor.to(device, non_blocking=True).full_tensor() if isinstance(tensor, DTensor) else tensor

    def _collect_lora_params(self):
        """
        collect lora params or full params if base model is not ready in vllm
        work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
        """
        from peft.utils.save_and_load import get_peft_model_state_dict

        lora_params = OrderedDict()
        if fsdp_version(self.module) > 0:
            with FSDP.summon_full_params(self.module, writeback=False):
                if self.base_sync_done:
                    lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
                    lora_params = {
                        name: param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in lora_params.items()
                    }
                else:
                    model = self.module._fsdp_wrapped_module.base_model.model
                    orig_dev = next(model.parameters()).device
                    model = model.to("cpu")
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ["_flat_param", "lora_"]):
                            continue
                        name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                        lora_params[name] = (
                            param.full_tensor().detach().cpu()
                            if hasattr(param, "full_tensor")
                            else param.detach().cpu()
                        )
                    model = model.to(orig_dev)
            torch.cuda.empty_cache()
        else:
            if self.base_sync_done:
                lora_params = get_peft_model_state_dict(self.module._fsdp_wrapped_module)
            else:
                model = self.module._fsdp_wrapped_module.base_model.model
                orig_dev = next(model.parameters()).device
                model = model.to("cpu")
                for name, param in model.state_dict().items():
                    if any(x in name for x in ["_flat_param", "lora_"]):
                        continue
                    name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                    lora_params[name] = param.detach().cpu()
                model = model.to(orig_dev)
        return lora_params

    def _sync_weight_to_vllm(self):
        if self.use_param_offload:
            load_fsdp_model(self.module)

        peft_config = None
        if isinstance(self.module._fsdp_wrapped_module, PeftModel):
            peft_config = self.module._fsdp_wrapped_module.peft_config.get("default", None)
            actor_weights = self._collect_lora_params()
        else:
            actor_weights = get_model_state_dict(self.module)
            actor_weights = self._rename_weight_keys(actor_weights, self.module._fsdp_wrapped_module)

        print_gpu_memory_usage("After gather model weights in sharding manager")

        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        if peft_config:
            if self.base_sync_done:
                lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
                lora_reqest = TensorLoRARequest(
                    lora_name=f"{lora_int_id}",
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=actor_weights,
                )
                self.inference_engine.llm_engine.add_lora(lora_reqest)
                print_gpu_memory_usage("After load LoRA weights in sharding manager")
                return
            else:

                def replace_lora_wrapper(k):
                    stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    if any(k.endswith(f"{s}.weight") for s in stacked_params):
                        return k.replace(".weight", ".base_layer.weight")
                    if any(k.endswith(f"{s}.bias") for s in stacked_params):
                        return k.replace(".bias", ".base_layer.bias")
                    return k

                actor_weights = {replace_lora_wrapper(k): v for k, v in actor_weights.items()}

        model.load_weights(self._make_weight_iterator(actor_weights))

        self.base_sync_done = True

        del actor_weights
        if self.use_param_offload:
            offload_fsdp_model(self.module)

        torch.cuda.empty_cache()
        print_gpu_memory_usage("After sync model weights in sharding manager")

    def load_vllm_and_sync_weights(self):
        """Load vllm engine and sync model weights to vllm model."""
        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        torch.cuda.empty_cache()
        assert self.loaded is False, "vllm engine has already been loaded"
        self.loaded = True

        print_gpu_memory_usage("Before vllm wake up in sharding manager")
        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["weights"])
        else:
            self.inference_engine.wake_up()

        self._sync_weight_to_vllm()

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["kv_cache"])

        print_gpu_memory_usage("After vllm wake up in sharding manager")
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def offload_vllm(self):
        """Offload vllm engine."""
        assert self.loaded is True, "vllm engine has not been loaded"
        self.loaded = False

        print_gpu_memory_usage("Before vllm offload in sharding manager")
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]
        self.inference_engine.sleep(level=1)
        free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
        self.freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        print_gpu_memory_usage("After vllm offload in sharding manager")

        self.module.train()
        torch.cuda.empty_cache()  # add empty cache after each compute

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        all_gather_data_proto(data, size=self.tp_size, group=self.tp_group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size > 1:
            data = data.chunk(chunks=self.tp_size)[self.tp_rank]

        return data
