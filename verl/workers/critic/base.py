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
Base class for Critic
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from ...protocol import DataProto
from .config import CriticConfig


__all__ = ["BasePPOCritic"]


class BasePPOCritic(ABC):
    def __init__(self, config: CriticConfig):
        self.config = config

    @abstractmethod
    def compute_values(self, data: DataProto) -> torch.Tensor:
        """Compute values"""
        pass

    @abstractmethod
    def update_critic(self, data: DataProto) -> dict[str, Any]:
        """Update the critic"""
        pass
