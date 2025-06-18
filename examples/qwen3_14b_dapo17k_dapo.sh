#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-14B  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=Saigyouji-Yuyuko1000/dapo17k@train \
    data.val_files=Saigyouji-Yuyuko1000/dapo17k@test \
    data.format_prompt=./examples/format_prompt/dapo_format.jinja \
    data.max_prompt_length=2048 \
    data.max_response_length=20480 \
    data.rollout_batch_size=512 \
    data.mini_rollout_batch_size=1536 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.global_batch_size=32 \
    worker.rollout.n=16 \
    worker.rollout.max_num_batched_tokens=22529 \
    worker.reward.reward_function=./examples/reward_function/math_dapo.py:compute_score \
    worker.reward.reward_function_kwargs='{"max_response_length":20480,"overlong_buffer_len":4096,"overlong_penalty_factor":1.0}'\
    trainer.experiment_name=qwen3_8b_dapo17k_dapo \
    trainer.max_try_make_batch=10 \
    algorithm.online_filtering=true \
    algorithm.disable_kl=true \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99


