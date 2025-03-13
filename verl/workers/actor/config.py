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
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    override_config: Dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    freeze_vision_tower: bool = False

    def post_init(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path


@dataclass
class OptimConfig:
    lr: float = 1e-6
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    lr_warmup_ratio: float = 0.0
    min_lr_ratio: Optional[float] = None
    warmup_style: str = "constant"
    """auto keys"""
    training_steps: int = field(default=-1, init=False)


@dataclass
class FSDPConfig:
    enable_full_shard: bool = True
    enable_cpu_offload: bool = False
    enable_rank0_init: bool = False
    use_orig_params: bool = False
    torch_dtype: Optional[str] = None
    fsdp_size: int = -1
    mp_param_dtype: str = "bf16"
    mp_reduce_dtype: str = "fp32"
    mp_buffer_dtype: str = "fp32"


@dataclass
class OffloadConfig:
    offload_params: bool = False
    offload_optimizer: bool = False


@dataclass
class ActorConfig:
    strategy: str = "fsdp"
    global_batch_size: int = 256
    micro_batch_size_per_device_for_update: int = 4
    micro_batch_size_per_device_for_experience: int = 16
    max_grad_norm: float = 1.0
    clip_ratio: float = 0.2
    entropy_coeff: float = 1e-3
    use_kl_loss: bool = True
    kl_loss_coef: float = 1e-3
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    padding_free: bool = False
    ulysses_sequence_parallel_size: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    """auto keys"""
    global_batch_size_per_device: int = field(default=-1, init=False)


@dataclass
class RefConfig:
    strategy: str = "fsdp"
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    """auto keys"""
    micro_batch_size_per_device_for_experience: int = field(default=-1, init=False)
    padding_free: bool = field(default=False, init=False)
    ulysses_sequence_parallel_size: int = field(default=1, init=False)
