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
Utilities to create common models
"""

from functools import lru_cache
from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn


@lru_cache
def is_rank0() -> int:
    return (not dist.is_initialized()) or (dist.get_rank() == 0)


def print_gpu_memory_usage(prefix: str) -> None:
    if is_rank0():
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"{prefix} memory allocated: {memory_allocated:.2f} GB, memory reserved: {memory_reserved:.2f} GB.")


def get_model_size(model: nn.Module, scale: str = "auto") -> Tuple[float, str]:
    n_params = sum(p.numel() for p in model.parameters())

    if scale == "auto":
        if n_params > 1e9:
            scale = "B"
        elif n_params > 1e6:
            scale = "M"
        elif n_params > 1e3:
            scale = "K"
        else:
            scale = ""

    if scale == "B":
        n_params = n_params / 1e9
    elif scale == "M":
        n_params = n_params / 1e6
    elif scale == "K":
        n_params = n_params / 1e3
    elif scale == "":
        pass
    else:
        raise NotImplementedError(f"Unknown scale {scale}.")

    return n_params, scale


def print_model_size(model: nn.Module, name: str = None) -> None:
    n_params, scale = get_model_size(model, scale="auto")
    if name is None:
        name = model.__class__.__name__

    print(f"{name} contains {n_params:.2f}{scale} parameters")
