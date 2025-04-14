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

import importlib.util
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


ScoreFunction = Callable[[str, str], RewardScore]


@dataclass
class FunctionRewardManager:
    config: RewardConfig
    tokenizer: PreTrainedTokenizer

    def __post_init__(self):
        """Load score function."""
        if self.config.score_function is None:
            raise ValueError("Score function is not provided.")

        if not os.path.exists(self.config.score_function):
            raise FileNotFoundError(f"Score function file {self.config.score_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_score_fn", self.config.score_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_score_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load score function: {e}")

        if not hasattr(module, self.config.score_function_name):
            raise AttributeError(f"Module {module} does not have function {self.config.score_function_name}.")

        score_fn: ScoreFunction = getattr(module, self.config.score_function_name)
        print(f"Using score function `{self.config.score_function_name}` from `{self.config.score_function}`.")
        self.score_fn = partial(score_fn, **self.config.score_function_kwargs)

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ground_truth = data_item.non_tensor_batch["ground_truth"]

            score = self.score_fn(response_str, ground_truth)
            reward_tensor[i, valid_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
