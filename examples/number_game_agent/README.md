# 启动训练

**注意**: 训练脚本必须从 EasyR1 根目录运行,因为所有路径配置都是相对于根目录的。

## 基础用法

### 启动训练
```bash
# 1. 切换到 EasyR1 根目录
cd /path/to/EasyR1

# 2. 运行训练脚本
bash examples/number_game_agent/qwen2_5_vl_3b_numgame_grpo.sh
```

### 可用的环境变量

**模型和数据**:
- `MODEL_PATH`: 模型路径 (默认: `/home/ubuntu/Qwen2.5-VL-3B-Instruct`)
- `TRAIN_FILES`: 训练数据路径
- `TEST_FILES`: 测试数据路径
- `CONFIG_PATH`: 配置文件路径

**训练超参数**:
- `TOTAL_EPOCHS`: 训练轮数 (默认: `1`)
- `MAX_STEPS`: 最大训练步数 (默认: `3`)
- `ROLLOUT_BATCH_SIZE`: Rollout批次大小 (默认: `4`)
- `GLOBAL_BATCH_SIZE`: 全局批次大小 (默认: `4`)
- `GPU_MEMORY_UTIL`: GPU显存利用率 (默认: `0.4`)

**集群配置**:
- `NNODES`: 节点数量 (默认: `1`)
- `GPUS_PER_NODE`: 每节点GPU数 (默认: `2`)
- `TENSOR_PARALLEL_SIZE`: 张量并行大小 (默认: `1`)

**实验配置**:
- `PROJECT_NAME`: 项目名称
- `EXPERIMENT_NAME`: 实验名称
- `LOGGER`: 日志记录器 (默认: `'["console","wandb"]'`)

**其他**:
- `AUTO_START=1`: 跳过确认直接开始训练
- `RAY_WAIT_TIMEOUT`: Ray集群等待超时时间(秒)

## 配置说明

本项目的训练配置已与 EasyR1 官方最佳实践对齐，主要特点：

### 完全对齐的参数
- 学习率: `1e-6` (RL 训练的保守学习率)
- GRPO 响应数: `n=5`
- 验证频率: 每 5 个 epoch
- 保存频率: 每 5 个 epoch
- KL 系数: `1e-2`

### 针对任务优化
- `max_prompt_length: 1024` (vs 官方 2048) - 游戏提示较短
- `max_response_length: 128` (vs 官方 2048) - 答案很短 (0/1/2)
- `global_batch_size: 64` (vs 官方 128) - 简单任务可用更小batch
- `tensor_parallel_size: 1` (vs 官方 2) - 3B 模型不需要 TP

详细的参数对照表和优化说明请参考: [PARAMETER_ALIGNMENT.md](PARAMETER_ALIGNMENT.md)

### 导出模型
```shell
python3 scripts/model_merger.py --local_dir /home/ubuntu/EasyR1/checkpoints/Mobile_Number_Game_RL/quick_test_20251120_110841/global_step_35/actor
```

### 启动推理
```shell
vllm serve 
  /home/ubuntu/EasyR1/checkpoints/Mobile_Number_Game_RL/quick_test_20251120_110841/global_step_35/actor/huggingface/ --host 0.0.0.0 --port 8000
```

### 测试模型
```shell
curl -X POST http://42.193.100.201:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "/home/ubuntu/EasyR1/checkpoints/Mobile_Number_Game_RL/quick_test_20251120_110841/global_step_35/actor/huggingface/",
      "messages": [
        {
          "role": "user",
          "content": "你好，请介绍一下你自己。"
        }
      ],
      "max_tokens": 512,
      "temperature": 0.7
    }'
```