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
Rollout config
"""

from dataclasses import asdict, dataclass, field


@dataclass
class RolloutConfig:
    name: str = "vllm"
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = False
    free_cache_engine: bool = False
    enable_chunked_prefill: bool = False
    tensor_parallel_size: int = 2
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 1024
    disable_log_stats: bool = True
    do_sample: bool = True
    n: int = 1
    limit_images: int = 0
    """auto keys"""
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)

    def to_dict(self):
        return asdict(self)
