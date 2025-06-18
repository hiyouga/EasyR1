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

import re
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else -1.0


def soft_overlong_punishment(response_length: int, max_response_length: int, overlong_buffer_len: int):
    if response_length <= max_response_length - overlong_buffer_len:
        return 0.0
    elif (max_response_length - overlong_buffer_len) < response_length <= max_response_length:
        return ((max_response_length - overlong_buffer_len) - response_length) / overlong_buffer_len
    else:
        return -1.0


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    max_response_length: int,
    overlong_buffer_len: int,
    overlong_penalty_factor: float,
) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        response_length = reward_input["response_length"]
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        soft_overlong_punishment_score = soft_overlong_punishment(
            response_length, max_response_length, overlong_buffer_len
        )
        scores.append(
            {
                "overall": accuracy_score + soft_overlong_punishment_score * overlong_penalty_factor,
                "accuracy": accuracy_score,
                "soft_overlong_punishment": soft_overlong_punishment_score,
            }
        )

    return scores
