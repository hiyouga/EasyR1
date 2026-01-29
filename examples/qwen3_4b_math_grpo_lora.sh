#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-4B  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=2048 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=64 \
    worker.actor.model.lora.alpha=64 \
    worker.actor.optim.lr=1e-5 \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen3_4b_math_grpo_lora \
    trainer.n_gpus_per_node=1
