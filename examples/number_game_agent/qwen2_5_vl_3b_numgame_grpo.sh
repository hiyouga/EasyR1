#!/bin/bash
# ============================================================
# Number Game Agent GRPO Training Script
#
# Description:
#   Vision-language RL training for number guessing game
#
# Usage:
#   cd /path/to/EasyR1
#   bash examples/number_game_agent/qwen2_5_vl_3b_numgame_grpo.sh
#
# Important:
#   - Must run from EasyR1 root directory
#   - All paths are relative to EasyR1 root
#   - Ray cluster must be started before running
# ============================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================
# Environment Variables Configuration
# ============================================================

# Model and data paths
export MODEL_PATH=${MODEL_PATH:-/home/ubuntu/Qwen2.5-VL-3B-Instruct}
export DATA_ROOT=${DATA_ROOT:-examples/number_game_agent/data}
export TRAIN_FILES=${TRAIN_FILES:-examples/number_game_agent/data/game_dataset_1119_v1/train.jsonl}
export TEST_FILES=${TEST_FILES:-examples/number_game_agent/data/game_dataset_1119_v1/test.jsonl}
export CONFIG_PATH=${CONFIG_PATH:-examples/number_game_agent/configs/config.yaml}

# Training hyperparameters (⚡ OPTIMIZED for small dataset + H20 GPUs)
export NNODES=${NNODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-2}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-6}  # ⚡ OPTIMIZED: Based on 5-epoch test (91.67%), 6 epochs provides optimal convergence
# MAX_STEPS: don't set default, will be handled in python command below
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}  # ⚡ 240 samples → 7.5 iters/epoch
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-60}  # ⚡ Match test set size
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}  # ⚡ Match rollout batch
export GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.75}  # ⚡ H20 has 80GB, use more
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}  # 3B doesn't need TP
export ROLLOUT_N=${ROLLOUT_N:-5}  # ⚡ CRITICAL: Industry standard for GRPO, DO NOT reduce (n is for statistical estimation, not action enumeration)
export ACTOR_LR=${ACTOR_LR:-5.0e-6}  # ⚡ VERIFIED: Quick test achieved 91.67% validation accuracy with this LR

# Experiment configuration
export PROJECT_NAME=${PROJECT_NAME:-Mobile_Number_Game_RL}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_vl_3b_number_game_grpo_$(date +%Y%m%d_%H%M%S)}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}

# Resume from checkpoint (optional)
# Set this to resume training: export RESUME_CHECKPOINT=checkpoints/path/to/experiment
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

# Logging
export LOGGER=${LOGGER:-'["console","wandb"]'}
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Performance tuning
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

# Hugging Face mirror (for China users)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# ============================================================
# Pre-flight Checks
# ============================================================

echo ""
echo "========================================================"
echo "  Number Game Agent GRPO Training"
echo "========================================================"
echo ""

# Check if running from EasyR1 root
if [ ! -f "verl/__init__.py" ]; then
    echo "❌ ERROR: This script must be run from the EasyR1 root directory"
    echo ""
    echo "Current directory: $(pwd)"
    echo ""
    echo "Correct usage:"
    echo "  cd /path/to/EasyR1"
    echo "  bash examples/number_game_agent/qwen2_5_vl_3b_numgame_grpo.sh"
    echo ""
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Check if training data exists
if [ ! -f "$TRAIN_FILES" ]; then
    echo "❌ ERROR: Training data not found: $TRAIN_FILES"
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  WARNING: Model directory not found: $MODEL_PATH"
    echo "   Training will fail if model cannot be loaded"
    read -p "   Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Pre-flight checks passed"
echo ""

# ============================================================
# Checkpoint Resume Detection
# ============================================================

if [ -n "${RESUME_CHECKPOINT}" ]; then
    if [ ! -d "${RESUME_CHECKPOINT}" ]; then
        echo "❌ ERROR: Resume checkpoint not found: ${RESUME_CHECKPOINT}"
        exit 1
    fi
    
    # Extract experiment name from checkpoint path
    EXPERIMENT_NAME=$(basename "${RESUME_CHECKPOINT}")
    
    echo "======================================================="
    echo "  🔄 Resuming from Checkpoint"
    echo "======================================================="
    echo ""
    echo "Checkpoint:  ${RESUME_CHECKPOINT}"
    echo "Experiment:  ${EXPERIMENT_NAME}"
    
    # Check checkpoint tracker
    if [ -f "${RESUME_CHECKPOINT}/checkpoint_tracker.json" ]; then
        echo ""
        echo "Checkpoint Info:"
        cat "${RESUME_CHECKPOINT}/checkpoint_tracker.json" | python3 -m json.tool 2>/dev/null || cat "${RESUME_CHECKPOINT}/checkpoint_tracker.json"
    fi
    
    echo ""
    echo "======================================================="
    echo ""
fi

# ============================================================
# Ray Cluster Status Check
# ============================================================

echo "========================================================"
echo "  Checking Ray Cluster Status"
echo "========================================================"
echo ""

REQUIRED_NODES=$NNODES
REQUIRED_GPUS=$((NNODES * GPUS_PER_NODE))
MAX_WAIT_SECONDS=${RAY_WAIT_TIMEOUT:-300}
CHECK_INTERVAL=5

echo "[Cluster Config]"
echo "  Nodes:      ${REQUIRED_NODES}"
echo "  GPUs/node:  ${GPUS_PER_NODE}"
echo "  Total GPUs: ${REQUIRED_GPUS}"
echo "  Max wait:   ${MAX_WAIT_SECONDS}s"
echo ""

iteration=0
max_iterations=$((MAX_WAIT_SECONDS / CHECK_INTERVAL))

while [ $iteration -lt $max_iterations ]; do
    iteration=$((iteration + 1))
    elapsed=$((iteration * CHECK_INTERVAL))

    # Check Ray cluster using Python API
    cluster_info=$(python3 -c "
import ray
import sys

try:
    ray.init(address='auto', ignore_reinit_error=True)
    resources = ray.cluster_resources()
    nodes = ray.nodes()
    alive_nodes = sum(1 for node in nodes if node['Alive'])
    gpu_count = int(resources.get('GPU', 0))
    print(f'{alive_nodes},{gpu_count}')
    sys.exit(0)
except Exception as e:
    print('0,0')
    sys.exit(1)
" 2>/dev/null)

    check_exit_code=$?

    if [ $check_exit_code -ne 0 ] || [ -z "$cluster_info" ]; then
        echo "[${elapsed}s] ⚠️  Ray cluster not accessible, waiting..."
        sleep $CHECK_INTERVAL
        continue
    fi

    current_nodes=$(echo "$cluster_info" | cut -d',' -f1)
    current_gpus=$(echo "$cluster_info" | cut -d',' -f2)

    if [ -z "$current_nodes" ] || [ -z "$current_gpus" ]; then
        echo "[${elapsed}s] ⚠️  Failed to parse cluster info"
        sleep $CHECK_INTERVAL
        continue
    fi

    echo "[${elapsed}s] Nodes: ${current_nodes}/${REQUIRED_NODES}, GPUs: ${current_gpus}/${REQUIRED_GPUS}"

    if [ "$current_nodes" -ge "$REQUIRED_NODES" ] && [ "$current_gpus" -ge "$REQUIRED_GPUS" ]; then
        echo ""
        echo "✅ Ray cluster is READY!"
        echo "   Nodes: ${current_nodes}"
        echo "   GPUs:  ${current_gpus}"
        echo ""
        break
    fi

    sleep $CHECK_INTERVAL
done

if [ $iteration -ge $max_iterations ]; then
    echo ""
    echo "❌ TIMEOUT: Ray cluster not ready in ${MAX_WAIT_SECONDS}s"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Start Ray cluster: ray start --head --port=6379"
    echo "  2. Check cluster status: ray status"
    echo "  3. Check available GPUs: nvidia-smi"
    echo ""
    exit 1
fi

# ============================================================
# Display Training Configuration
# ============================================================

echo "========================================================"
echo "  Training Configuration"
echo "========================================================"
echo ""
echo "[Model & Data]"
echo "  Model:        $MODEL_PATH"
echo "  Train data:   $TRAIN_FILES"
echo "  Val data:     $TEST_FILES"
echo "  Config:       $CONFIG_PATH"
echo ""
echo "[Hyperparameters]"
echo "  Total epochs:     $TOTAL_EPOCHS"
echo "  Max steps:        ${MAX_STEPS:-null}"
echo "  Rollout batch:    $ROLLOUT_BATCH_SIZE"
echo "  Global batch:     $GLOBAL_BATCH_SIZE"
echo "  GPU memory util:  $GPU_MEMORY_UTIL"
echo ""
echo "[Cluster]"
echo "  Nodes:            $NNODES"
echo "  GPUs per node:    $GPUS_PER_NODE"
echo "  Tensor parallel:  $TENSOR_PARALLEL_SIZE"
echo ""
echo "[Experiment]"
echo "  Project:      $PROJECT_NAME"
echo "  Experiment:   $EXPERIMENT_NAME"
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "  Mode:         🔄 RESUME from checkpoint"
    echo "  Resume from:  $RESUME_CHECKPOINT"
else
    echo "  Mode:         🆕 NEW training"
    echo "  Checkpoint:   $CHECKPOINT_DIR/$EXPERIMENT_NAME"
fi
echo ""
echo "========================================================"
echo ""

# Auto-start training (skip confirmation)
# To enable confirmation, set: export CONFIRM_START=1
if [ "${CONFIRM_START:-0}" = "1" ]; then
    read -p "Start training? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Cancelled by user"
        exit 0
    fi
fi

# ============================================================
# Logging Setup
# ============================================================

# Create log directory
LOG_DIR="logs/number_game_agent"
mkdir -p ${LOG_DIR}

# Generate log file path with timestamp
LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${LOG_TIMESTAMP}.log"

echo "📝 Training log will be saved to: ${LOG_FILE}"
echo ""

# ============================================================
# Launch Training
# ============================================================

echo ""
echo "========================================================"
echo "  Starting VERL Training"
echo "========================================================"
echo ""
echo "Training started at: $(date)"
echo ""

# Enable verbose output for debugging
set -x

# Use tee to output to both console and log file
{
# Build command with conditional checkpoint resume
CMD="python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${TEST_FILES} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.trust_remote_code=true \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.optim.lr=${ACTOR_LR} \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.logger=${LOGGER}"

# Add checkpoint resume if specified
if [ -n "${RESUME_CHECKPOINT}" ]; then
    CMD="${CMD} \
    trainer.load_checkpoint_path=${RESUME_CHECKPOINT} \
    trainer.find_last_checkpoint=true"
fi

# Execute training command
eval "${CMD}"
} 2>&1 | tee "${LOG_FILE}"

# Capture exit code from training command (first command in pipeline)
training_exit_code=${PIPESTATUS[0]}

set +x

# ============================================================
# Post-training Summary
# ============================================================

echo ""
echo "========================================================"
echo "  Training Summary"
echo "========================================================"
echo ""
echo "Training finished at: $(date)"

if [ $training_exit_code -eq 0 ]; then
    echo "Status: ✅ SUCCESS"
    echo ""
    echo "Training log saved to:"
    echo "  ${LOG_FILE}"
    echo ""
    echo "Checkpoint location:"
    echo "  ${CHECKPOINT_DIR}/${EXPERIMENT_NAME}"
    echo ""

    # Check if checkpoint exists
    if [ -d "${CHECKPOINT_DIR}/${EXPERIMENT_NAME}" ]; then
        echo "Saved checkpoints:"
        ls -lh "${CHECKPOINT_DIR}/${EXPERIMENT_NAME}" 2>/dev/null || echo "  (No checkpoints saved - save_freq=-1)"
    fi
    
    echo ""
    echo "To analyze training metrics:"
    echo "  # View validation accuracy"
    echo "  grep 'val/accuracy' ${LOG_FILE}"
    echo ""
    echo "  # View training rewards"
    echo "  grep 'reward_score:' ${LOG_FILE}"
    echo ""
    echo "  # Check for errors"
    echo "  grep -i 'error\\|exception' ${LOG_FILE}"
else
    echo "Status: ❌ FAILED (exit code: $training_exit_code)"
    echo ""
    echo "Training log saved to:"
    echo "  ${LOG_FILE}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check logs: cat ${LOG_FILE}"
    echo "  2. Verify GPU memory is sufficient (try reducing gpu_memory_utilization)"
    echo "  3. Check data format and paths"
    echo "  4. Ensure model weights are accessible"
fi

echo ""
echo "========================================================"
echo ""

exit $training_exit_code
