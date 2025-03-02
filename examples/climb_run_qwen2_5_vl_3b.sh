set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export
export MKL_SERVICE_FORCE_INTEL=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/grpo_climb.yaml \
    data.max_prompt_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_3b_climb \
    trainer.n_gpus_per_node=2
