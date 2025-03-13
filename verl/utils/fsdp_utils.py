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

import gc
from collections import defaultdict
from functools import partial
from typing import Callable, Union

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import Optimizer
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name


def get_init_fn(model: nn.Module, device: Union[str, torch.device]) -> Callable[[nn.Module], None]:
    param_occurrence = defaultdict(int)
    for _, param in model.named_parameters(remove_duplicate=False):
        param_occurrence[param] += 1

    duplicated_params = {param for param in param_occurrence.keys() if param_occurrence[param] > 1}
    materialized_params = {}

    def init_fn(module: nn.Module):
        for name, param in module.named_parameters(recurse=False):
            if param in duplicated_params:
                module._parameters[name] = materialized_params.setdefault(
                    param, nn.Parameter(torch.empty_like(param.data, device=device), requires_grad=param.requires_grad)
                )
            else:
                module._parameters[name] = nn.Parameter(
                    torch.empty_like(param.data, device=device), requires_grad=param.requires_grad
                )

    return init_fn


def get_fsdp_wrap_policy(model: PreTrainedModel):
    """Get FSDP wrap policy for the model.

    Args:
        module: The module to get wrap policy for
    """
    transformer_cls_to_wrap = set()
    for module in model._no_split_modules:
        transformer_cls = get_module_class_from_name(model, module)
        if transformer_cls is None:
            raise Exception(f"Cannot find {module} in pretrained model.")
        else:
            transformer_cls_to_wrap.add(transformer_cls)

    return partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_cls_to_wrap)


@torch.no_grad()
def offload_fsdp_model(model: FSDP, empty_cache: bool = True):
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue

        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to("cpu", non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data
        assert id(flat_param._local_shard) != id(flat_param.data)

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_model(model: FSDP, empty_cache: bool = True):
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model loading to GPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue

        flat_param = handle.flat_param
        handle.flat_param_to("cuda", non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data

    if empty_cache:
        gc.collect()


@torch.no_grad()
def offload_fsdp_optimizer(optimizer: Optimizer, empty_cache: bool = True):
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_optimizer(optimizer: Optimizer, empty_cache: bool = True):
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cuda", non_blocking=True)

    if empty_cache:
        gc.collect()
