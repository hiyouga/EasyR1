set -x

/home/dvdai/miniconda3/bin/conda activate test

# Create a system prompt file
cat > system_prompt.txt << 'EOL'
You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
Before analyzing medical images, you must identify and outline all objects of interest that are relevant to diagnosis in json format,
with list od bounding box in a key called "bbox_2d" with the format [x1, y1, x2, y2]. The json should be wrapped in ```json ... ``` tags.
The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.
EOL

# Read the system prompt from the file
format_prompt=$(cat system_prompt.txt)

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    algorithm.adv_estimator=sdrpo \
    data.format_prompt="${format_prompt}" \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.n_gpus_per_node=4 \
    trainer.experiment_name=sdrpo