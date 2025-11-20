#!/bin/bash
# 测试在线训练脚本 - 单设备、最小配置

set -e

echo "=== 在线游戏训练测试脚本 ==="
echo "启动时间: $(date)"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate easyr1

# 进入项目目录
cd /home/ubuntu/EasyR1

# 设置环境变量
export PYTHONPATH=/home/ubuntu/EasyR1:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

echo -e "\n=== 检查 Android 设备连接 ==="
adb devices

echo -e "\n=== 启动 Ray 集群 ==="
ray stop 2>/dev/null || true
sleep 2
ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-gpus=1
sleep 3
ray status

echo -e "\n=== 启动训练 ==="
python3 -m verl.trainer.main \
  config=number_game_agent/configs/test_online.yaml \
  2>&1 | tee /tmp/training_test_$(date +%Y%m%d_%H%M%S).log

echo -e "\n=== 训练完成 ==="
echo "完成时间: $(date)"
