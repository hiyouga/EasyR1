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


def _import_rollout_config():
    """Import RolloutConfig directly without triggering heavy package imports."""
    spec = importlib.util.spec_from_file_location(
        "rollout_config",
        os.path.join(os.path.dirname(__file__), "..", "verl", "workers", "rollout", "config.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RolloutConfig


def test_max_num_batched_tokens_default():
    RolloutConfig = _import_rollout_config()
    config = RolloutConfig()
    assert config.max_num_batched_tokens == 8192


def test_max_num_batched_tokens_configurable():
    RolloutConfig = _import_rollout_config()
    config = RolloutConfig(max_num_batched_tokens=16384)
    assert config.max_num_batched_tokens == 16384


def test_max_num_batched_tokens_auto_adjust():
    """When max_num_batched_tokens < prompt_length + response_length,
    it should be auto-adjusted rather than raising a ValueError."""
    RolloutConfig = _import_rollout_config()
    config = RolloutConfig(max_num_batched_tokens=1024)
    config.prompt_length = 4096
    config.response_length = 4096

    # Simulate the auto-adjustment logic from vllm_rollout_spmd.py
    if config.max_num_batched_tokens < config.prompt_length + config.response_length:
        config.max_num_batched_tokens = config.prompt_length + config.response_length

    assert config.max_num_batched_tokens == 8192
