set -x

source /home/dvdai/miniconda3/etc/profile.d/conda.sh
conda activate test


# # === Step 1: Isolate Ray (even in shared env) ===
# export RAY_TEMP_DIR=/home/$USER/tmp_ray_isolated
# rm -rf $RAY_TEMP_DIR
# mkdir -p $RAY_TEMP_DIR

# export RAY_PORT=7391
# export RAY_ADDRESS=10.1.25.0:$RAY_PORT

# # # Optional: if ~/.ray is corrupted or shared, isolate it
# export RAY_HOME=$HOME/.ray_isolated
# mkdir -p $RAY_HOME

# # # Optional: cleaner logs
# # export RAY_LOG_TO_STDERR=1

# # # === Step 2: Clean any zombie processes ===
# ray stop --force || true
# pkill -9 raylet gcs_server redis-server python || true

# rm -rf /tmp/ray/* || true
# rm -rf "$RAY_TEMP_DIR"/* || true


# # Remove all Ray temp data (yours and global)
# rm -rf /tmp/ray/* || true
# rm -rf /home/$USER/tmp_ray_isolated/* || true

# # # === Step 3: Start Ray cleanly ===
# ray start --num-gpus=2 --head --include-dashboard=false --port=$RAY_PORT --temp-dir=$RAY_TEMP_DIR --disable-usage-stats > ray.log 2>&1 &

# # export RAY_BACKEND_LOG_LEVEL=error

# trap "ray stop --force; pkill -9 raylet; pkill -9 gcs_server; pkill -9 redis-server" EXIT

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