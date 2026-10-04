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
from torch import nn

from verl.protocol import DataProto
from verl.workers.actor.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.critic.config import CriticConfig
from verl.workers.critic.dp_critic import DataParallelPPOCritic


LENGTHS = [1, 7, 2, 6, 5, 3, 8, 4]  # response lengths
RESPONSE_LENGTH = 8


def _create_data(lengths: list[int], rows: list[int]) -> DataProto:
    response_mask = (torch.arange(RESPONSE_LENGTH)[None, :] < torch.tensor(lengths)[rows, None]).long()
    prompt = torch.ones(len(rows), 2, dtype=torch.long)
    attention_mask = torch.cat([prompt, response_mask], dim=-1)
    # the first input id holds the row, so that the fake forward pass can look up the row's parameters
    input_ids = torch.cat([torch.tensor(rows)[:, None], prompt[:, :1], torch.zeros_like(response_mask)], dim=-1)
    targets = torch.tensor([float(row + 1) * (-1) ** row for row in rows])[:, None] * response_mask
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": torch.cumsum(attention_mask, dim=-1),
            "responses": torch.zeros_like(response_mask),
            "response_mask": response_mask,
            "old_log_probs": torch.zeros(len(rows), RESPONSE_LENGTH),
            "advantages": targets,
            "values": torch.zeros(len(rows), RESPONSE_LENGTH),
            "returns": targets,
        },
        non_tensors={"multi_modal_inputs": np.array([None] * len(rows), dtype=object)},
        meta_info={"temperature": 1.0},
    )


def _update(monkeypatch, worker, mode, lengths, rows, world_size, micro, dynamic, total=None):
    """Run one update of one rank: returns the gradient and the value the rank passed to all_reduce.

    The update forwards the rows of the rank and all_reduce returns ``total`` (or the rank's own value if None).
    """
    kwargs = dict(
        global_batch_size=len(rows),
        micro_batch_size_per_device_for_update=micro,
        micro_batch_size_per_device_for_experience=micro,
        loss_avg_mode=mode,
        padding_free=False,
        dynamic_batching=dynamic,
    )
    params = nn.Parameter(torch.zeros(len(lengths), RESPONSE_LENGTH))  # log probs or values of every token
    if worker == "actor":
        config = ActorConfig(**kwargs, use_torch_compile=False)
        model = DataParallelPPOActor(config=config, actor_module=nn.Linear(1, 1), actor_optimizer=None)
        update = model.update_policy
    else:
        config = CriticConfig(**kwargs)
        model = DataParallelPPOCritic(config=config, critic_module=nn.Linear(1, 1), critic_optimizer=None)
        update = model.update_critic

    config.global_batch_size_per_device = len(rows)
    model.world_size = world_size
    local_values = []

    def all_reduce(tensor, op=None):
        local_values.append(tensor.clone())
        if total is not None:
            tensor.fill_(total)

    monkeypatch.setattr(
        model, "_forward_micro_batch", lambda model_inputs, **kwargs: params[model_inputs["input_ids"][:, 0]]
    )
    monkeypatch.setattr(model, "_optimizer_step", lambda: torch.tensor(0.0))
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    update(_create_data(lengths, rows))
    return params.grad, local_values[0]


def _expected_grad(mode: str, lengths: list[int]) -> torch.Tensor:
    """d loss / d param at the initial point, where both losses reduce to -target * param per token."""
    data = _create_data(lengths, list(range(len(lengths))))
    mask = data.batch["response_mask"].float()
    tokens = mask.sum(-1, keepdim=True)
    if mode == "seq":  # the mean over the tokens of each response, then over the responses with a token
        weights = mask / tokens.clamp(min=1) / (tokens > 0).sum()
    else:  # the mean over all tokens
        weights = mask / mask.sum()

    return -data.batch["advantages"] * weights


def _world_grad(monkeypatch, worker, mode, lengths, world, micro, dynamic):
    """Average the gradients of the ranks as FSDP does; ``world`` holds the rows of every rank."""
    args = (monkeypatch, worker, mode, lengths)
    total = sum(_update(*args, rows, len(world), micro, dynamic)[1] for rows in world)
    grads = [_update(*args, rows, len(world), micro, dynamic, total)[0] for rows in world]
    return sum(grads) / len(world)


@pytest.mark.parametrize("worker", ["actor", "critic"])
@pytest.mark.parametrize("mode", ["seq", "token"])
@pytest.mark.parametrize("micro,dynamic", [(8, False), (4, False), (1, False), (2, True)])
def test_loss_avg_mode_with_micro_batches(monkeypatch, worker, mode, micro, dynamic):
    world = [list(range(len(LENGTHS)))]
    grad = _world_grad(monkeypatch, worker, mode, LENGTHS, world, micro, dynamic)
    torch.testing.assert_close(grad, _expected_grad(mode, LENGTHS))


@pytest.mark.parametrize("worker", ["actor", "critic"])
@pytest.mark.parametrize("mode", ["seq", "token"])
def test_loss_avg_mode_with_ranks(monkeypatch, worker, mode):
    world = [[0, 2, 3, 5], [1, 4, 6, 7]]  # the ranks hold responses of different lengths
    grad = _world_grad(monkeypatch, worker, mode, LENGTHS, world, micro=2, dynamic=False)
    torch.testing.assert_close(grad, _expected_grad(mode, LENGTHS))


@pytest.mark.parametrize("worker", ["actor", "critic"])
@pytest.mark.parametrize("mode", ["seq", "token"])
def test_loss_avg_mode_with_empty_response(monkeypatch, worker, mode):
    """A response without tokens gets no weight, so the other responses are not scaled down."""
    lengths = [1, 7, 2, 0, 5, 3, 8, 4]
    world = [[0, 2, 3, 5], [1, 4, 6, 7]]
    grad = _world_grad(monkeypatch, worker, mode, lengths, world, micro=2, dynamic=False)
    torch.testing.assert_close(grad, _expected_grad(mode, lengths))
