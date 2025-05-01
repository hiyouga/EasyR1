set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    algorithm.adv_estimator=grpo \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.n_gpus_per_node=4 \
    trainer.experiment_name=grpo_vanilla