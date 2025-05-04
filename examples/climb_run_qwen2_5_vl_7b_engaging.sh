set -x

source /home/dvdai/miniconda3/etc/profile.d/conda.sh
conda activate test

format_prompt="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging_nobox.yaml \
    algorithm.adv_estimator=grpo \
    data.format_prompt="${format_prompt}" \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.n_gpus_per_node=1
