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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist

from verl.protocol import DataProto
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.workers.actor import dp_actor
from verl.workers.critic import dp_critic


def _create_random_mask(
    input_ids: torch.Tensor,
    max_ratio_of_valid_token: float,
    max_ratio_of_left_padding: float,
    min_ratio_of_valid_token: float = 0,
) -> torch.Tensor:
    """Create a random mask given input_ids. Support left padding and right padding.

    Process:
    - Sample valid token length
    - Sample left_padding length
    - Generate padding

    Args:
        input_ids:
            shape (batch_size, seq_len)

    Returns:
        mask:
            shape (batch_size, seq_len)
    """
    assert max_ratio_of_valid_token > 0 and max_ratio_of_valid_token <= 1.0
    assert max_ratio_of_left_padding >= 0 and max_ratio_of_left_padding < 1.0
    assert min_ratio_of_valid_token <= max_ratio_of_valid_token

    batch_size, sequence_length = input_ids.shape
    max_num_valid_tokens = int(sequence_length * max_ratio_of_valid_token)
    min_num_valid_tokens = max(1, int(sequence_length * min_ratio_of_valid_token))
    max_left_padding = int(sequence_length * max_ratio_of_left_padding)
    assert max_num_valid_tokens + max_left_padding <= sequence_length
    assert max_num_valid_tokens > 0 and max_ratio_of_valid_token <= sequence_length
    mask = torch.ones_like(input_ids, dtype=torch.int64)
    # TODO: we can make this faster
    for i in range(batch_size):
        num_left_padding = np.random.randint(low=0, high=max_left_padding + 1, dtype=np.int64)
        num_valid = np.random.randint(low=min_num_valid_tokens, high=max_num_valid_tokens + 1, dtype=np.int64)

        for index in range(num_left_padding):
            mask[i, index] = 0

        for index in range(num_left_padding + num_valid, sequence_length):
            mask[i, index] = 0

    return mask


def test_dynamic_batch():
    input_ids = torch.randint(low=0, high=10, size=(20, 100))
    attention_mask = _create_random_mask(
        input_ids=input_ids, max_ratio_of_left_padding=0.1, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.5
    )
    data = {"input_ids": input_ids, "attention_mask": attention_mask}
    dataproto = DataProto.from_single_dict(data)
    micro_batches, micro_bsz_idx_lst = prepare_dynamic_batch(dataproto, max_token_len=300)
    input_ids = torch.cat([micro_batch.batch["input_ids"] for micro_batch in micro_batches], dim=0)
    input_ids = restore_dynamic_batch(input_ids, micro_bsz_idx_lst)
    torch.testing.assert_close(input_ids, dataproto.batch["input_ids"])


def _make_batch(seqlens):
    seq_len = 8
    input_ids = torch.arange(len(seqlens) * seq_len).reshape(-1, seq_len) % 31 + 1
    attention_mask = (torch.arange(seq_len) >= seq_len - torch.tensor(seqlens).unsqueeze(-1)).long()
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": attention_mask[:, -4:].clone(),
        },
        non_tensors={
            "sample_id": np.arange(len(seqlens)),
            "multi_modal_inputs": np.array(
                [{"pixel_values": torch.tensor([[float(i)]])} for i in range(len(seqlens))], dtype=object
            ),
        },
    )


@pytest.mark.parametrize(
    "rank_seqlens",
    [([8, 8, 8], [1, 2]), ([8, 8, 8, 8], [1, 2]), ([8, 7, 6], [1, 2, 1])],
    ids=["uneven-samples", "multiple-dummies", "equal-samples-uneven-tokens"],
)
@pytest.mark.parametrize("explicit_group", [False, True])
@pytest.mark.parametrize("with_response_mask", [False, True])
def test_dynamic_batch_synchronized_counts(monkeypatch, rank_seqlens, explicit_group, with_response_mask):
    local_counts = [min(len(seqlens), (sum(seqlens) + 7) // 8) for seqlens in rank_seqlens]
    synced_count = max(local_counts)
    dp_group = object() if explicit_group else None
    observed_counts = []

    def all_reduce(tensor, op, group):
        assert tensor.device == torch.device("cpu")
        assert tensor.dtype == torch.long
        assert op == dist.ReduceOp.MAX
        assert group is dp_group
        observed_counts.append(tensor.item())
        tensor.fill_(synced_count)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    rank_batch_counts = []
    for seqlens in rank_seqlens:
        data = _make_batch(seqlens)
        if not with_response_mask:
            del data.batch["response_mask"]
        original = data.batch.clone()
        micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=8, dp_group=dp_group)
        rank_batch_counts.append(len(micro_batches))
        assert len(batch_idx_list) == synced_count
        assert sum(not indices for indices in batch_idx_list) == max(0, synced_count - len(data))

        for micro_batch, indices in zip(micro_batches, batch_idx_list):
            micro_batch.check_consistency()
            np.testing.assert_array_equal(micro_batch.non_tensor_batch["sample_id"], indices or [0])
            for inputs, index in zip(micro_batch.non_tensor_batch["multi_modal_inputs"], indices or [0]):
                torch.testing.assert_close(
                    inputs["pixel_values"], data.non_tensor_batch["multi_modal_inputs"][index]["pixel_values"]
                )
            if not indices:
                assert len(micro_batch) == 1
                assert not micro_batch.batch["attention_mask"].any()
                if with_response_mask:
                    assert not micro_batch.batch["response_mask"].any()
                torch.testing.assert_close(micro_batch.batch["input_ids"], data.batch["input_ids"][:1])

        for key, tensor in original.items():
            real_outputs = torch.cat(
                [micro_batch.batch[key] for micro_batch, indices in zip(micro_batches, batch_idx_list) if indices]
            )
            torch.testing.assert_close(restore_dynamic_batch(real_outputs, batch_idx_list), tensor)
            torch.testing.assert_close(data.batch[key], tensor)  # Dummy masking must not mutate sample 0.

    assert observed_counts == local_counts
    assert rank_batch_counts == [synced_count, synced_count]


class _TestModel(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(-0.1, 0.1, output_size))
        self.forward_count = 0
        self.backward_grads = []

    def forward(self, input_ids, attention_mask, position_ids, pixel_values, use_cache):
        # An all-zero dummy mask would either reach this assertion or fail during unpadding.
        assert input_ids.numel() > 0
        assert attention_mask is None or attention_mask.any()
        assert pixel_values.numel() > 0
        self.forward_count += 1
        logits = input_ids.float().unsqueeze(-1) * self.weight
        if logits.requires_grad:
            logits.register_hook(lambda grad: self.backward_grads.append(grad.detach().clone()))
        return SimpleNamespace(logits=logits)


@pytest.fixture(params=["actor", "critic"])
def worker_setup(request, monkeypatch):
    module = dp_actor if request.param == "actor" else dp_critic

    # Exercise the workers' padding-free paths on CPU without requiring FlashAttention kernels.
    def unpad_input(hidden_states, attention_mask):
        indices = attention_mask.flatten().nonzero().squeeze(-1)
        assert indices.numel() > 0
        return hidden_states.flatten(0, 1)[indices], indices, None, None

    def pad_input(hidden_states, indices, batch, seqlen):
        output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
        return output.index_copy(0, indices, hidden_states).reshape(batch, seqlen, -1)

    monkeypatch.setattr(module, "unpad_input", unpad_input, raising=False)
    monkeypatch.setattr(module, "pad_input", pad_input, raising=False)
    monkeypatch.setattr(module, "index_first_axis", lambda data, indices: data[indices], raising=False)
    monkeypatch.setattr(module, "rearrange", dp_actor.rearrange, raising=False)
    monkeypatch.setenv("RANK", "1")  # Disable progress bars.
    monkeypatch.setenv("WORLD_SIZE", "1")

    def make_worker(padding_free, dynamic_batching):
        if request.param == "actor":
            config = dp_actor.ActorConfig(use_torch_compile=False)
            model = _TestModel(32)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            worker = dp_actor.DataParallelPPOActor(config, model, optimizer)
            config.use_kl_loss = True
            config.kl_coef = 0.1
        else:
            config = dp_critic.CriticConfig()
            model = _TestModel(1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            worker = dp_critic.DataParallelPPOCritic(config, model, optimizer)
        config.padding_free = padding_free
        config.dynamic_batching = dynamic_batching
        config.global_batch_size_per_device = 2
        config.micro_batch_size_per_device_for_experience = 1
        config.micro_batch_size_per_device_for_update = 1
        return worker, model, optimizer

    return request.param, make_worker


@pytest.mark.parametrize("padding_free", [False, True])
def test_dynamic_batch_worker_dummy_steps(monkeypatch, worker_setup, padding_free):
    worker_type, make_worker = worker_setup
    baseline, baseline_model, baseline_optimizer = make_worker(padding_free, dynamic_batching=False)
    worker, model, optimizer = make_worker(padding_free, dynamic_batching=True)
    data = _make_batch([6, 8])
    data.batch["position_ids"] = torch.arange(8).expand(2, -1)
    data.batch["responses"] = data.batch["input_ids"][:, -4:].clone()
    data.batch["advantages"] = torch.tensor([[0.5] * 4, [-0.5] * 4])
    data.batch["old_log_probs"] = torch.full((2, 4), -3.0)
    data.batch["ref_log_probs"] = torch.full((2, 4), -3.5)
    data.batch["values"] = torch.zeros(2, 4)
    data.batch["returns"] = torch.ones(2, 4)
    data.meta_info["temperature"] = 1.0
    original = data.batch.clone()
    group = object()
    reductions = []

    def all_reduce(tensor, op, group=None):
        reductions.append(op)
        if op == dist.ReduceOp.MAX:
            assert group is dist.group.WORLD
            tensor.fill_(4)  # Two real micro-batches and two collective-only dummies.
        else:
            assert op == dist.ReduceOp.SUM
            assert tensor.item() == data.batch["response_mask"].sum().item()

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "group", SimpleNamespace(WORLD=group))
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    compute = "compute_log_prob" if worker_type == "actor" else "compute_values"
    expected = getattr(baseline, compute)(data)
    actual = getattr(worker, compute)(data)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 4)
    assert baseline_model.forward_count == 2
    assert model.forward_count == 4

    update = "update_policy" if worker_type == "actor" else "update_critic"
    baseline_metrics = getattr(baseline, update)(data)
    metrics = getattr(worker, update)(data)
    assert baseline_model.forward_count == 4
    assert model.forward_count == 8
    assert len(baseline_model.backward_grads) == 2
    assert len(model.backward_grads) == 4
    assert all(torch.isfinite(grad).all() and not grad.any() for grad in model.backward_grads[-2:])
    assert any(grad.any() for grad in model.backward_grads[:2])
    torch.testing.assert_close(model.weight, baseline_model.weight)
    torch.testing.assert_close(optimizer.state_dict(), baseline_optimizer.state_dict())
    loss_key = "actor/pg_loss" if worker_type == "actor" else "critic/vf_loss"
    assert len(metrics[loss_key]) == len(baseline_metrics[loss_key]) == 2
    assert reductions.count(dist.ReduceOp.MAX) == 2
    for key, tensor in original.items():
        torch.testing.assert_close(data.batch[key], tensor)
