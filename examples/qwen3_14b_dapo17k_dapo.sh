#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-14B-Base  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=Saigyouji-Yuyuko1000/dapo17k@train \
    data.val_files=Saigyouji-Yuyuko1000/dapo17k@test \
    data.format_prompt=./examples/format_prompt/dapo.jinja \
    data.max_prompt_length=2048 \
    data.max_response_length=20480 \
    data.rollout_batch_size=512 \
    data.mini_rollout_batch_size=256 \
    worker.actor.ulysses_size=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.weight_decay=0.1 \
    worker.actor.optim.lr_warmup_steps=10 \
    worker.actor.global_batch_size=32 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.clip_ratio_dual=10.0 \
    worker.rollout.n=16 \
    worker.rollout.max_num_batched_tokens=22528 \
    worker.rollout.val_override_config='{"n":16,"temperature":1.0,"top_p":0.7}' \
    worker.rollout.gpu_memory_utilization=0.8 \
    worker.rollout.tensor_parallel_size=4 \
    worker.reward.reward_function=./examples/reward_function/dapo.py:compute_score \
    worker.reward.reward_function_kwargs='{"max_response_length":20480,"overlong_buffer_length":4096,"overlong_penalty_factor":1.0}' \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    algorithm.filter_key=accuracy_normalized \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    trainer.total_epochs=10 \
    trainer.max_try_make_batch=10 \
    trainer.experiment_name=qwen3_14b_dapo17k_dapo \
    trainer.n_gpus_per_node=8
