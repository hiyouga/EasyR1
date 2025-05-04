set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging_bf16.yaml \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-32B-Instruct \
    trainer.n_gpus_per_node=4 \
    trainer.experiment_name=drpo_32b