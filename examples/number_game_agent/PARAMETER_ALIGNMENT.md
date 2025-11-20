# Number Game Agent 训练参数对标说明

本文档说明了 Number Game Agent 训练配置如何与 EasyR1 官方配置对标。

## 参数对照表

### 数据配置 (data)

| 参数 | Number Game | Official (3B) | Official (7B) | 说明 |
|------|-------------|---------------|---------------|------|
| `max_prompt_length` | 1024 | 2048 | 2048 | 数字游戏提示较短，减少长度 |
| `max_response_length` | 128 | 2048 | 2048 | 游戏答案很短 (0/1/2)，大幅减少 |
| `rollout_batch_size` | 128 | 512 | 512 | 为3B模型适当减小 |
| `val_batch_size` | 256 | 1024 | 1024 | 为3B模型适当减小 |
| `seed` | 1 | 1 | 1 | ✅ 对齐 |
| `min_pixels` | 262144 | 262144 | 262144 | ✅ 对齐 |
| `max_pixels` | 4194304 | 4194304 | 4194304 | ✅ 对齐 |
| `filter_overlong_prompts` | true | true | true | ✅ 对齐 |

### 算法配置 (algorithm)

| 参数 | Number Game | Official | 说明 |
|------|-------------|----------|------|
| `adv_estimator` | grpo | grpo | ✅ 对齐 |
| `disable_kl` | false | false | ✅ 对齐 |
| `use_kl_loss` | true | true | ✅ 对齐 |
| `kl_penalty` | low_var_kl | low_var_kl | ✅ 对齐 |
| `kl_coef` | 1.0e-2 | 1.0e-2 | ✅ 对齐 |
| `online_filtering` | false | false | ✅ 对齐 |

### Actor 配置 (worker.actor)

| 参数 | Number Game | Official (3B) | Official (7B) | 说明 |
|------|-------------|---------------|---------------|------|
| `global_batch_size` | 64 | 128 | 128 | 为简单任务适当减小 |
| `micro_batch_size_per_device_for_update` | 1 | 1 | 1 | ✅ 对齐 |
| `micro_batch_size_per_device_for_experience` | 2 | 2 | 2 | ✅ 对齐 |
| `max_grad_norm` | 1.0 | 1.0 | 1.0 | ✅ 对齐 |
| `padding_free` | true | true | true | ✅ 对齐 |
| `dynamic_batching` | true | true | true | ✅ 对齐 |
| `ulysses_size` | 1 | 1 | 1 | ✅ 对齐 |

### 优化器配置 (worker.actor.optim)

| 参数 | Number Game | Official | 说明 |
|------|-------------|----------|------|
| `lr` | 1.0e-6 | 1.0e-6 | ✅ 对齐 (RL常用保守学习率) |
| `weight_decay` | 1.0e-2 | 1.0e-2 | ✅ 对齐 |
| `strategy` | adamw | adamw | ✅ 对齐 |
| `lr_warmup_ratio` | 0.0 | 0.0 | ✅ 对齐 (RL不需要warmup) |

### FSDP 配置 (worker.actor.fsdp)

| 参数 | Number Game | Official | 说明 |
|------|-------------|----------|------|
| `enable_full_shard` | true | true | ✅ 对齐 |
| `enable_cpu_offload` | false | false | ✅ 对齐 |
| `enable_rank0_init` | true | true | ✅ 对齐 |

### Offload 配置 (worker.actor.offload)

| 参数 | Number Game | Official | 说明 |
|------|-------------|----------|------|
| `offload_params` | true | true | ✅ 对齐 (节省GPU内存) |
| `offload_optimizer` | true | true | ✅ 对齐 (节省GPU内存) |

### Rollout 配置 (worker.rollout)

| 参数 | Number Game | Official (3B) | Official (7B) | 说明 |
|------|-------------|---------------|---------------|------|
| `n` | 5 | 5 | 5 | ✅ 对齐 (GRPO生成5个响应) |
| `temperature` | 1.0 | 1.0 | 1.0 | ✅ 对齐 |
| `top_p` | 1.0 | 1.0 | 1.0 | ✅ 对齐 |
| `limit_images` | 1 | 0 | 0 | 游戏每个样本只有1张图 |
| `gpu_memory_utilization` | 0.6 | 0.6 | 0.6 | ✅ 对齐 |
| `tensor_parallel_size` | 1 | 2 | 2 | 3B模型不需要TP，7B需要 |
| `val_override_config.temperature` | 0.6 | 0.6 | 0.6 | ✅ 对齐 |
| `val_override_config.top_p` | 0.95 | 0.95 | 0.95 | ✅ 对齐 |
| `val_override_config.n` | 1 | 1 | 1 | ✅ 对齐 |

### 训练器配置 (trainer)

| 参数 | Number Game | Official | 说明 |
|------|-------------|----------|------|
| `total_epochs` | 15 | 15 | ✅ 对齐 |
| `max_steps` | null | null | ✅ 对齐 |
| `n_gpus_per_node` | 2 | 8 | 根据硬件调整 |
| `max_try_make_batch` | 20 | 20 | ✅ 对齐 |
| `val_freq` | 5 | 5 | ✅ 对齐 (每5个epoch验证) |
| `val_before_train` | true | true | ✅ 对齐 |
| `val_generations_to_log` | 3 | 3 | ✅ 对齐 |
| `save_freq` | 5 | 5 | ✅ 对齐 (每5个epoch保存) |
| `save_limit` | 3 | 3 | ✅ 对齐 (保留最新3个checkpoint) |
| `find_last_checkpoint` | true | true | ✅ 对齐 (自动恢复训练) |

## 关键调整说明

### 1. 序列长度优化
```yaml
# 官方配置
max_prompt_length: 2048
max_response_length: 2048

# Number Game 优化
max_prompt_length: 1024  # 游戏提示较短
max_response_length: 128  # 答案只是 0/1/2，大幅减少
```
**优势**: 减少内存占用，提高训练速度

### 2. 批次大小调整
```yaml
# 官方配置 (7B模型，8 GPU)
rollout_batch_size: 512
global_batch_size: 128

# Number Game 优化 (3B模型，2 GPU)
rollout_batch_size: 128  # 减小以适应更少GPU
global_batch_size: 64    # 简单任务可用更小batch
```
**优势**: 适配硬件资源，保持训练稳定性

### 3. Tensor Parallelism
```yaml
# 官方配置 (7B+模型)
tensor_parallel_size: 2

# Number Game (3B模型)
tensor_parallel_size: 1
```
**优势**: 3B模型单GPU可容纳，避免通信开销

### 4. 学习率策略
```yaml
# 保持与官方一致
lr: 1.0e-6           # RL训练的保守学习率
lr_warmup_ratio: 0.0  # RL不需要warmup
```
**原因**: RL训练对学习率敏感，使用经验证的保守值

### 5. GRPO 响应数量
```yaml
# 对齐官方配置
rollout.n: 5  # 每个prompt生成5个响应
```
**原因**: GRPO算法需要多个响应进行对比学习

## 训练脚本默认值

### 快速测试
```bash
# 仅运行1步快速验证
MAX_STEPS=1 bash examples/number_game_agent/qwen2_5_vl_3b_numgame_grpo.sh
```

### 完整训练 (默认)
```bash
# 15 epochs, 每5个epoch验证和保存
TOTAL_EPOCHS=15 bash examples/number_game_agent/qwen2_5_vl_3b_numgame_grpo.sh
```

### 自定义配置
```bash
# 使用7B模型，需要更多资源
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
GPUS_PER_NODE=4 \
TENSOR_PARALLEL_SIZE=2 \
ROLLOUT_BATCH_SIZE=256 \
GLOBAL_BATCH_SIZE=128 \
bash examples/number_game_agent/qwen2_5_vl_3b_numgame_grpo.sh
```

## 内存优化建议

### GPU内存受限时
```bash
# 减少批次大小
ROLLOUT_BATCH_SIZE=64 GLOBAL_BATCH_SIZE=32 bash ...

# 减少GPU内存利用率
GPU_MEMORY_UTIL=0.4 bash ...

# 使用梯度检查点 (默认已启用)
# enable_gradient_checkpointing: true
```

### CPU内存受限时
```bash
# 在config.yaml中修改:
worker.actor.offload.offload_params: false
worker.actor.offload.offload_optimizer: false
worker.ref.fsdp.enable_cpu_offload: false
```

## 性能优化建议

### 使用 BF16 (如果硬件支持)
```yaml
# 在config.yaml中:
worker.actor.fsdp.torch_dtype: bf16
worker.actor.optim.strategy: adamw_bf16
```

### 多GPU并行
```bash
# 使用更多GPU
GPUS_PER_NODE=4 bash ...

# 7B+ 模型使用 Tensor Parallelism
TENSOR_PARALLEL_SIZE=2 bash ...
```

## 对标总结

✅ **完全对齐**: 算法参数、优化器配置、验证/保存频率
✅ **适当调整**: 批次大小、序列长度(适配任务特点)
✅ **硬件适配**: GPU数量、Tensor Parallelism(适配可用资源)

Number Game Agent 的配置在保持与官方最佳实践对齐的同时，针对任务特点和硬件环境做了合理优化。
