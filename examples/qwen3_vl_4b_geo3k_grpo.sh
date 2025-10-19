#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_4b_geo_grpo \
    trainer.n_gpus_per_node=2
