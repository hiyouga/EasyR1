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

import torch

from verl.trainer.core_algos import compute_gae_advantage_return


def _reference_gae(rewards, values, mask, gamma, lam):
    """Per-sample GAE that skips masked (observation) tokens."""
    advantages = torch.zeros_like(rewards)
    for i in range(rewards.shape[0]):
        nextvalue, lastgaelam = 0.0, 0.0
        for t in reversed(range(rewards.shape[1])):
            gaelam = rewards[i, t] + gamma * nextvalue - values[i, t] + gamma * lam * lastgaelam
            if mask[i, t]:
                nextvalue, lastgaelam = values[i, t], gaelam
            advantages[i, t] = lastgaelam
    return advantages


def test_gae_batched_with_observation_tokens():
    torch.manual_seed(0)
    rewards = torch.zeros(3, 6)
    rewards[0, 3] = rewards[1, 5] = rewards[2, 1] = 1.0
    values = torch.randn(3, 6)
    # sample 0 ends early, sample 1 has an observation span, sample 2 is a full response
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.float)
    _, returns = compute_gae_advantage_return(
        token_level_rewards=rewards, values=values, response_mask=mask, gamma=0.9, lam=0.8
    )
    expected = _reference_gae(rewards, values, mask, gamma=0.9, lam=0.8)
    torch.testing.assert_close(returns, expected + values)
