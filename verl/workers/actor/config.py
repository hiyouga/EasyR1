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
Actor config
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ActorOptimConfig:
    lr: float = 1e-6
    lr_warmup_steps_ratio: float = 0.0
    min_lr_ratio: Optional[float] = None
    warmup_style: Literal["constant", "cosine"] = "constant"
    training_steps: int = field(default=-1, init=False)


@dataclass
class ActorWorkerConfig:
    param_offload: bool = False
    optimizer_offload: bool = False
    fsdp_size: int = -1


@dataclass
class ActorConfig:
    strategy: Literal["fsdp"] = "fsdp"
    global_mini_batch_size: int = 256
    micro_batch_size_per_device: int = 8
    max_token_len_per_gpu: int = 16384
    max_grad_norm: float = 1.0
    clip_ratio: float = 0.2
    entropy_coeff: float = 1e-3
    use_kl_loss: bool = False
    kl_loss_coef: bool = 1e-3
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    ulysses_sequence_parallel_size: int = 1
    mini_batch_size_per_device: int = field(default=-1, init=False)
    use_dynamic_bsz: bool = field(default=False, init=False)
    use_remove_padding: bool = field(default=False, init=False)
    optim: ActorOptimConfig = field(default_factory=ActorOptimConfig)
    worker: ActorWorkerConfig = field(default_factory=ActorWorkerConfig)
