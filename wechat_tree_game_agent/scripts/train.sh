#!/bin/bash
# 微信砍树游戏 GRPO 训练启动脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "微信砍树游戏 GRPO 训练"
echo "=========================================="

# 检查数据集是否存在
TRAIN_DATA="wechat_tree_game_agent/data/tree_game_dataset.jsonl"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误: 训练数据集不存在: $TRAIN_DATA"
    echo "请先运行: python wechat_tree_game_agent/data/process_dataset.py"
    exit 1
fi

# 检查配置文件是否存在
CONFIG_FILE="wechat_tree_game_agent/config/tree_game_grpo.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 设置实验名称（带时间戳）
EXPERIMENT_NAME="tree_game_grpo_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "配置信息:"
echo "  训练数据: $TRAIN_DATA"
echo "  配置文件: $CONFIG_FILE"
echo "  实验名称: $EXPERIMENT_NAME"
echo ""

# 确认启动
read -p "按 Enter 键开始训练，或 Ctrl+C 取消..."

echo ""
echo "启动训练..."
echo "=========================================="

# 运行训练
python -m verl.trainer.main \
    config="$CONFIG_FILE" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger='["wandb", "file"]'

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "检查点保存在: wechat_tree_game_agent/checkpoints/$EXPERIMENT_NAME"
echo "日志保存在: outputs/$EXPERIMENT_NAME"
echo ""
