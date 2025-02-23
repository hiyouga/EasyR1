set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=/mnt/hdfs/veomni/models/qwen2_5-7b-instruct/  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/grpo/qwen2_5_7b_math.yaml \
    worker.actor.model.model_path=${MODEL_PATH}
