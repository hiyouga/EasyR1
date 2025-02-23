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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils import get_processor, get_tokenizer
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from verl.workers.reward import CustomRewardManager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config):
    # print initial config
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # instantiate tokenizer
    tokenizer = get_tokenizer(config.actor_rollout_ref.model.path)
    processor = get_processor(config.actor_rollout_ref.model.path, use_fast=True)

    # define worker classes
    ray_worker_group_cls = RayWorkerGroup
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    reward_fn = CustomRewardManager(
        tokenizer=tokenizer, num_examine=1, compute_score=config.reward_model.reward_compute_score
    )
    val_reward_fn = CustomRewardManager(
        tokenizer=tokenizer, num_examine=1, compute_score=config.reward_model.reward_compute_score
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
