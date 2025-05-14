#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export VERL_USE_MODELSCOPE=True

MODEL_PATH=Qwen/Qwen2.5-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH}
