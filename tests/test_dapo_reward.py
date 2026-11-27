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
from pathlib import Path

import pytest


def load_dapo_reward():
    path = Path(__file__).resolve().parents[1] / "examples" / "reward_function" / "dapo.py"
    spec = importlib.util.spec_from_file_location("dapo_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dapo = load_dapo_reward()


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Answer: $5$", 1.0),
        ("Answer: 5", 1.0),
        ("Answer: $\\boxed{5}$", 1.0),
        ("Answer: $6$", -1.0),
        ("Answer: 5 or 6", -1.0),
        ("Answer: $5$ or $6$", -1.0),
        ("Answer: $5$ is incorrect; my final answer is $6$.", -1.0),
        ("Answer: $\\boxed{5}$ or $\\boxed{6}$", -1.0),
        ("Answer: $5$ where $n$ is the count", 1.0),
        ("Answer: $5$, so $x=5$", 1.0),
        ("Answer: $6$\nAnswer: $5$", 1.0),
    ],
)
def test_accuracy_reward_gold_5(response, expected):
    assert dapo.accuracy_reward(response, "5") == expected


def test_same_number_written_twice_is_not_a_conflict():
    assert dapo.accuracy_reward("Answer: $1,000$, that is $1000$", "1000") == 1.0
    assert dapo.accuracy_reward("Answer: $-3$ or $3$", "3") == -1.0


def test_compute_score_rejects_hedged_answer():
    reward_inputs = [
        {"response": "Answer: $5$ or $6$", "response_length": 10, "ground_truth": "5"},
        {"response": "Answer: $5$", "response_length": 10, "ground_truth": "5"},
    ]
    scores = dapo.compute_score(
        reward_inputs, max_response_length=20480, overlong_buffer_length=4096, overlong_penalty_factor=1.0
    )
    assert [score["overall"] for score in scores] == [-1.0, 1.0]
