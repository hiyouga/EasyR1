#!/bin/bash

set -x

MODEL_PATH=zai-org/GLM-4.1V-9B-Base  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=glm4.1v_base_geo_grpo \
    worker.actor.padding_free=True \
    trainer.n_gpus_per_node=8
