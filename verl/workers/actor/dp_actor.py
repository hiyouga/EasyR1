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
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.workers.actor.base import BasePPOActor
from verl.workers.actor.config import ActorConfig


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(
        self, micro_batch: Dict[str, torch.Tensor], temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
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
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **vision_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = logprobs_from_logits(logits, responses)  # (bsz, response_length)
            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

        return entropy, log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        self.actor_optimizer.step()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "pixel_values" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["pixel_values", "image_grid_thw"]
        else:
            non_tensor_select_keys = None

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        for micro_batch in tqdm(micro_batches, desc="Compute log probs", disable=(self.rank != 0)):
            micro_batch.to("cuda")
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            _, log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

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

            self.actor_optimizer.zero_grad()
            for micro_batch in tqdm(micro_batches, desc=f"Update policy [{i + 1}/{n}]", disable=(self.rank != 0)):
                micro_batch.to("cuda")
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                responses = model_inputs["responses"]
                response_length = responses.size(1)
                attention_mask = model_inputs["attention_mask"]
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = model_inputs["old_log_probs"]
                advantages = model_inputs["advantages"]

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(model_inputs, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    eos_mask=response_mask,
                    cliprange=clip_ratio,
                )
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]
                    # compute kl loss
                    kld = core_algos.kl_penalty(
                        logprob=log_prob,
                        ref_logprob=ref_log_prob,
                        kl_penalty=self.config.kl_loss_type,
                    )
                    kl_loss = masked_mean(kld, response_mask)
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics["actor/kl_loss"] = kl_loss.detach().item()
                    metrics["actor/kl_coef"] = self.config.kl_loss_coef

                loss = policy_loss / gradient_accumulation
                loss.backward()

                batch_metrics = {
                    "actor/entropy_loss": entropy_loss.detach().item(),
                    "actor/pg_loss": pg_loss.detach().item(),
                    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    "actor/ppo_kl": ppo_kl.detach().item(),
                }
                append_to_dict(metrics, batch_metrics)

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
