set -x

/home/dvdai/miniconda3/bin/conda activate test

export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_BACKEND_LOG_LEVEL=debug
#export MKL_SERVICE_FORCE_INTEL=1

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.max_prompt_length=4096 \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_climb \
    trainer.n_gpus_per_node=4
