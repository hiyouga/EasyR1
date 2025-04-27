#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.use_kl_loss=false \
    algorithm.kl_penalty=kl \
    algorithm.kl_coef=1.0e-3 \
    trainer.experiment_name=qwen2_5_vl_7b_geo_reinforce_pp \
    trainer.n_gpus_per_node=8
