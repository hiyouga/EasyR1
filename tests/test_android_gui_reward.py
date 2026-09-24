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


def _load_android_gui():
    path = Path(__file__).resolve().parents[1] / "examples" / "reward_function" / "android_gui.py"
    spec = importlib.util.spec_from_file_location("android_gui_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


android_gui = _load_android_gui()


def _score(response: str, ground_truth: str) -> float:
    return android_gui.compute_score([{"response": response, "ground_truth": ground_truth}])[0]["overall"]


@pytest.mark.parametrize(
    "response,ground_truth,expected",
    [
        ("1", "1", 1.0),
        ("0", "1", 0.0),
        ("I don't know", "1", 0.0),
        ("", "1", 0.0),
        ("The answer is 1", "1", 1.0),
        ("I choose option 2", "2", 1.0),
    ],
)
def test_bare_and_single_digit_replies_are_scored_as_before(response: str, ground_truth: str, expected: float):
    assert _score(response, ground_truth) == expected


@pytest.mark.parametrize(
    "response,ground_truth,expected",
    [
        ("The numbers are 12, 45, 3. Green light so the largest, 45, is at position 2.", "2", 1.0),
        ("The numbers are 12, 45, 3. Green light so the largest, 45, is at position 2.", "1", 0.0),
        ("1. Green light. 2. Numbers 7, 3, 9. 3. Largest is 9. Output: 2", "2", 1.0),
        ("1. Green light. 2. Numbers 7, 3, 9. 3. Largest is 9. Output: 2", "1", 0.0),
        ("2024", "2", 0.0),
        ("10", "1", 0.0),
    ],
)
def test_the_last_standalone_digit_is_the_answer(response: str, ground_truth: str, expected: float):
    assert _score(response, ground_truth) == expected
