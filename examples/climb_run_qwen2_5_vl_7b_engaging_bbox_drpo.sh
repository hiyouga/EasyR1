set -x

/home/dvdai/miniconda3/bin/conda activate test

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    algorithm.adv_estimator=drpo \
    worker.actor.model.model_path=/scratch/outputs/qwen/qwen25_vision_model \
    trainer.n_gpus_per_node=4 \
    trainer.experiment_name=drpo