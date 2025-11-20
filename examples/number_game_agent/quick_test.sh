#!/bin/bash
# ============================================================
# Quick Test Script - 快速验证优化配置
#
# 用途: 在投入完整30 epochs训练前，快速验证配置是否正确
# 运行时间: 约10-15分钟
# 预期结果: 5 epochs后准确率应达到60-70%
#
# 使用方法:
#   cd /home/ubuntu/EasyR1
#   bash examples/number_game_agent/quick_test.sh
# ============================================================

set -e

echo ""
echo "=========================================================="
echo "  🚀 Number Game - Quick Configuration Test"
echo "=========================================================="
echo ""
echo "This will run 5 epochs to verify the optimized config."
echo "Expected time: ~10-15 minutes"
echo "Expected accuracy after 5 epochs: 60-70%"
echo ""
echo "If results look good, run full 30 epochs training."
echo ""
echo "=========================================================="
echo ""

# 检查工作目录
if [ ! -f "verl/__init__.py" ]; then
    echo "❌ ERROR: Must run from EasyR1 root directory"
    echo "   Current: $(pwd)"
    echo "   Run: cd /home/ubuntu/EasyR1 && bash examples/number_game_agent/quick_test.sh"
    exit 1
fi

# 创建日志目录
LOG_DIR="logs/number_game_agent"
mkdir -p ${LOG_DIR}

# 快速测试配置
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export TOTAL_EPOCHS=5
# MAX_STEPS留空，使用config.yaml中的null值
export ROLLOUT_BATCH_SIZE=32
export VAL_BATCH_SIZE=60
export GLOBAL_BATCH_SIZE=32
export GPU_MEMORY_UTIL=0.75
export ROLLOUT_N=5
export ACTOR_LR=5.0e-6
export VAL_FREQ=1  # 每个epoch都验证
export SAVE_FREQ=-1  # 不保存checkpoint
export EXPERIMENT_NAME="quick_test_${TIMESTAMP}"

# 日志文件路径
LOG_FILE="${LOG_DIR}/quick_test_${TIMESTAMP}.log"

echo "Test Configuration:"
echo "  Epochs:            ${TOTAL_EPOCHS}"
echo "  Rollout Batch:     ${ROLLOUT_BATCH_SIZE}"
echo "  GRPO Samples (n):  ${ROLLOUT_N}"
echo "  Learning Rate:     ${ACTOR_LR}"
echo "  Validation:        Every epoch"
echo "  Log file:          ${LOG_FILE}"
echo ""
# Auto-start (skip confirmation)
if [ "${CONFIRM_START:-0}" = "1" ]; then
    read -p "Start quick test? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
        echo "Test cancelled"
        exit 0
    fi
fi

echo ""
echo "Starting quick test..."
echo "Monitor val/accuracy in console output"
echo "Full log will be saved to: ${LOG_FILE}"
echo ""

# 使用tee命令同时输出到终端和日志文件
bash examples/number_game_agent/qwen2_5_vl_3b_numgame_grpo.sh 2>&1 | tee "${LOG_FILE}"

# 保存退出状态
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================================="
echo "  Quick Test Complete"
echo "=========================================================="
echo ""
echo "Training log saved to: ${LOG_FILE}"
echo ""
echo "Next steps:"
echo ""
echo "If val/accuracy reached 60-70%:"
echo "  ✅ Config is good! Run full training:"
echo "     bash examples/number_game_agent/qwen2_5_vl_3b_numgame_grpo.sh"
echo ""
echo "If val/accuracy < 50%:"
echo "  ⚠️  Check WandB logs for issues"
echo "  ⚠️  Verify data files are correct"
echo "  ⚠️  Review log file: ${LOG_FILE}"
echo ""
echo "If val/accuracy > 80%:"
echo "  🎉 Excellent! Task is easier than expected"
echo "  🎉 Consider reducing epochs or increasing difficulty"
echo ""
echo "To analyze the log file:"
echo "  # View full log"
echo "  cat ${LOG_FILE}"
echo ""
echo "  # Extract validation accuracy"
echo "  grep 'val/accuracy' ${LOG_FILE}"
echo ""
echo "  # Check for errors"
echo "  grep -i 'error\\|exception\\|failed' ${LOG_FILE}"
echo ""
echo "=========================================================="
echo ""

# 如果训练失败，返回非零退出码
exit ${TRAIN_EXIT_CODE}
