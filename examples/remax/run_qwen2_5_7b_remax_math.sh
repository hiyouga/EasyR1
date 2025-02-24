set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=Qwen/Qwen2.5-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/remax/remax_example.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    trainer.n_gpus_per_node=4
