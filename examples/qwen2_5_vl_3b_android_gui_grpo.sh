#!/bin/bash

set -e

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_vl_3b_android_gui_grpo_$(date +%Y%m%d_%H%M%S)}

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=yuehua-s/numbergame@train \
    data.val_files=yuehua-s/numbergame@test \
    data.max_prompt_length=3072 \
    data.max_response_length=64 \
    data.rollout_batch_size=32 \
    data.val_batch_size=60 \
    data.format_prompt=examples/format_prompt/android_gui.jinja \
    data.seed=42 \
    data.filter_overlong_prompts=false \
    algorithm.kl_coef=4.0e-2 \
    worker.actor.global_batch_size=32 \
    worker.actor.max_grad_norm=0.1 \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-3B-Instruct \
    worker.actor.model.trust_remote_code=true \
    worker.actor.optim.lr=1.0e-5 \
    worker.actor.optim.weight_decay=1.0e-1 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.optim.lr_scheduler_type=constant \
    worker.rollout.n=8 \
    worker.rollout.temperature=0.9 \
    worker.rollout.top_p=0.95 \
    worker.rollout.limit_images=1 \
    worker.rollout.gpu_memory_utilization=0.75 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=examples/reward_function/android_gui.py:compute_score \
    trainer.total_epochs=3 \
    trainer.project_name=easy_r1 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_freq=2 \
    trainer.val_generations_to_log=10 \
    trainer.save_freq=10
