#!/bin/bash
# REMINDER: this script uses test data split and should ONLY be used for debugging. DO NOT use for training.

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/journeybench-multi-image-vqa@train \
    data.val_files=hiyouga/journeybench-multi-image-vqa@test \
    data.rollout_batch_size=256 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.limit_images=2 \
    trainer.experiment_name=qwen2_5_vl_7b_multi_image \
    trainer.n_gpus_per_node=8
