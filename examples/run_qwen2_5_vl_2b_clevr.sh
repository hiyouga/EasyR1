set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=BUAADreamer/clevr_count_70k_local@train \
    data.val_files=BUAADreamer/clevr_count_70k_local@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_3b_clevr \
    trainer.n_gpus_per_node=2 \
    worker.reward.compute_score=r1v