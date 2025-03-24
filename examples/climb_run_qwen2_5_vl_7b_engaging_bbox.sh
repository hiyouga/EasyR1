set -x

/home/dvdai/miniconda3/bin/conda activate test

# Create a system prompt file
cat > system_prompt.txt << 'EOL'
You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.

When analyzing medical images, you must identify and outline all objects of interest that are relevant to diagnosis in json format wrapped in ```json ... `````
EOL

# Read the system prompt from the file
SYSTEM_PROMPT=$(cat system_prompt.txt)

python -m verl.trainer.main \
    config=examples/grpo_climb_engaging.yaml \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.n_gpus_per_node=4