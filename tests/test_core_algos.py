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

import numpy as np
import pytest
import torch

from verl.trainer.core_algos import compute_grpo_outcome_advantage


def _advantages(reward_groups):
    """Outcome advantages for groups of per-response rewards (response length 1)."""
    rewards = [reward for group in reward_groups for reward in group]
    token_level_rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.concatenate([np.full(len(group), i, dtype=np.int64) for i, group in enumerate(reward_groups)])
    advantages, _ = compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    return advantages.squeeze(-1)


@pytest.mark.parametrize("reward", [0.1, 0.3, 0.6, 0.7, 0.9])
@pytest.mark.parametrize("group_size", [8, 16, 64])
def test_grpo_uniform_reward_group_has_zero_advantage(reward, group_size):
    advantages = _advantages([[reward] * group_size])
    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_grpo_uniform_group_next_to_mixed_group():
    advantages = _advantages([[0.3] * 8, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert torch.equal(advantages[:8], torch.zeros(8))
    # the mixed group keeps the usual standardized advantages
    assert advantages[8].item() == pytest.approx(2.4749, abs=1e-3)
    assert advantages[9].item() == pytest.approx(-0.3536, abs=1e-3)
