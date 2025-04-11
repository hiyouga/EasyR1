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
Reward config
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardConfig:
    reward_type: str = "function"
    score_function: Optional[str] = None
    score_function_kwargs: dict = field(default_factory=dict)
    skip_special_tokens: bool = True

    def post_init(self):
        if self.score_function is not None and os.path.exists(self.score_function):
            self.score_function = os.path.abspath(self.score_function)
