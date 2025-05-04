set -x

# source /home/dvdai/miniconda3/etc/profile.d/conda.sh
# conda activate test
source /home/peili/miniconda3/etc/profile.d/conda.sh
conda activate easyr1

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
    config=examples/debug.yaml \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.format_prompt="${format_prompt}" \
    worker.actor.model.model_path=/home/peili/EasyR1/verl/models/transformers/time_series_qwen2_5_vl \
    trainer.n_gpus_per_node=2