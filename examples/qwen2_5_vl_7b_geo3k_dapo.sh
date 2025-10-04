#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.mini_rollout_batch_size=128 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    trainer.experiment_name=qwen2_5_vl_7b_geo_dapo \
    trainer.n_gpus_per_node=8
