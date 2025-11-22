#!/bin/bash
# 🔥 Industry Best Practices Training Script
# Based on: DeepSeek R1 (kl=0.04, lr=1e-6) + Jackson Kek Vision GRPO (kl=0.06, lr=1e-5, n=8)

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate easyr1

MODEL_PATH=${MODEL_PATH:-/home/ubuntu/Qwen2.5-VL-3B-Instruct}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
ACTOR_LR=${ACTOR_LR:-1.0e-5}  # 🔥 Jackson Kek: 1e-5 for vision tasks
ROLLOUT_N=${ROLLOUT_N:-8}  # 🔥 Industry standard: 8 (Jackson + VLM-R1)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_vl_3b_numgame_industry_$(date +%Y%m%d_%H%M%S)}

echo "=========================================================="
echo "  🔥 Industry Best Practices GRPO Training"
echo "=========================================================="
echo ""
echo "Model:       $MODEL_PATH"
echo "Epochs:      $TOTAL_EPOCHS"
echo "LR:          $ACTOR_LR (Jackson: 1e-5)"
echo "Rollout n:   $ROLLOUT_N (Industry: 8)"
echo "Experiment:  $EXPERIMENT_NAME"
echo ""
echo "🔥 Key Industry Parameters:"
echo "  - kl_coef:        0.04 (DeepSeek=0.04, Jackson=0.06)"
echo "  - lr:             1e-5 (constant, Jackson best practice)"
echo "  - max_grad_norm:  0.1 (Jackson: 0.1)"
echo "  - weight_decay:   0.1 (Jackson: 0.1)"
echo "  - temperature:    0.9 (Jackson: 0.9)"
echo "  - rollout n:      8 (Jackson + VLM-R1)"
echo ""
echo "Expected: >95% accuracy (vs previous 83-88%)"
echo "=========================================================="
echo ""

python3 -m verl.trainer.main \
    config=examples/number_game_agent/configs/config.yaml \
    data.train_files=/home/ubuntu/numbergame/train-00000-of-00001.parquet \
    data.val_files=/home/ubuntu/numbergame/test-00000-of-00001.parquet \
    data.rollout_batch_size=32 \
    data.val_batch_size=60 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.trust_remote_code=true \
    worker.actor.global_batch_size=32 \
    worker.actor.optim.lr=${ACTOR_LR} \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.gpu_memory_utilization=0.75 \
    worker.rollout.tensor_parallel_size=1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.project_name=Mobile_Number_Game_RL \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.logger='["console","wandb"]'
