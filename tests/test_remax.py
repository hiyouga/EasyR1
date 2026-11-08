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

from verl.trainer.core_algos import compute_remax_outcome_advantage


def test_remax_advantage_for_a_batch():
    token_level_rewards = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0]])
    response_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32)
    reward_baselines = torch.tensor([0.25, 0.5, 1.0])

    advantages, returns = compute_remax_outcome_advantage(
        token_level_rewards=token_level_rewards, reward_baselines=reward_baselines, response_mask=response_mask
    )

    torch.testing.assert_close(advantages, torch.tensor([[0.75], [0.0], [-1.0]]) * response_mask)
    torch.testing.assert_close(
        returns, torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]])
    )
