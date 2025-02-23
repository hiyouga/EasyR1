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
"""
Implement Critic
"""

import os
from collections import defaultdict
from typing import Any, Dict

import torch
import torch.distributed
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

from verl import DataProto
from verl.trainer import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.workers.critic.base import BasePPOCritic
from verl.workers.critic.config import CriticConfig


__all__ = ["DataParallelPPOCritic"]


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config: CriticConfig, critic_module: nn.Module, critic_optimizer: torch.optim.Optimizer):
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        vision_inputs = {}
        if "pixel_values" in micro_batch:
            vision_inputs["pixel_values"] = torch.cat(micro_batch["pixel_values"], dim=0)
            vision_inputs["image_grid_thw"] = torch.cat(micro_batch["image_grid_thw"], dim=0)

        if self.config.padding_free:
            # TODO (yaowei): preprocess data for padding_free and ulysses
            raise NotImplementedError
        else:
            output = self.critic_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **vision_inputs,
                use_cache=False,
            )
            values: torch.Tensor = output.logits
            values = values[:, -response_length - 1 : -1].squeeze(-1)  # (bsz, response_length, vocab_size)

        return values

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_module.parameters(), max_norm=self.config.max_grad_norm
            )

        self.critic_optimizer.step()
        return grad_norm

    @torch.no_grad()
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "pixel_values" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["pixel_values", "image_grid_thw"]
        else:
            non_tensor_select_keys = None

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        values_lst = []
        for micro_batch in tqdm(micro_batches, "Compute values", disable=(self.rank != 0)):
            micro_batch.to("cuda")
            values = self._forward_micro_batch(micro_batch)
            values_lst.append(values)

        values = torch.concat(values_lst, dim=0)
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        response_length = responses.size(1)
        values = values * attention_mask[:, -response_length - 1 : -1]
        return values

    def update_critic(self, data: DataProto) -> Dict[str, Any]:
        self.critic_module.train()

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]
        if "pixel_values" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["pixel_values", "image_grid_thw"]
        else:
            non_tensor_select_keys = None

        # TODO (yaowei): support ppo epochs
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        n = len(mini_batches)
        for i, mini_batch in enumerate(mini_batches):
            gradient_accumulation = (
                self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
            )
            micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

            self.critic_optimizer.zero_grad()
            for micro_batch in tqdm(micro_batches, desc=f"Update critic [{i + 1}/{n}]", disable=(self.rank != 0)):
                micro_batch.to("cuda")
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                responses = model_inputs["responses"]
                attention_mask = model_inputs["attention_mask"]
                values = model_inputs["values"]
                returns = model_inputs["returns"]
                response_length = responses.size(1)
                eos_mask = attention_mask[:, -response_length - 1 : -1]

                vpreds = self._forward_micro_batch(data)
                vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                    vpreds=vpreds,
                    values=values,
                    returns=returns,
                    eos_mask=eos_mask,
                    cliprange_value=self.config.cliprange_value,
                )
                loss = vf_loss / gradient_accumulation
                loss.backward()

                batch_metrics = {
                    "critic/vf_loss": vf_loss.detach().item(),
                    "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                    "critic/vpred_mean": masked_mean(vpreds, eos_mask).detach().item(),
                }
                append_to_dict(metrics, batch_metrics)

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"critic/grad_norm": grad_norm.detach().item()})

        self.critic_optimizer.zero_grad()
        return metrics
