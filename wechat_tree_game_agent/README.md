# 微信砍树小游戏 Agent GRPO 强化学习训练方案

> **场景**: 通过视觉理解和点击操作，训练 Agent 在微信小程序砍树游戏中最大化战斗力
>
> **目标**: 验证 GRPO 算法在真实 Android 环境下的视觉决策能力

---

## 📋 目录

1. [方案概述](#1-方案概述)
2. [场景设计与可行性论证](#2-场景设计与可行性论证)
3. [技术架构](#3-技术架构)
4. [数据集构建](#4-数据集构建)
5. [训练流程](#5-训练流程)
6. [评估体系](#6-评估体系)
7. [实施时间表](#7-实施时间表)
8. [风险控制](#8-风险控制)

---

## 1. 方案概述

### 1.1 核心设计

**游戏规则**:
- 玩家点击屏幕砍树（需点击 10 次）
- 每次砍树可能掉落装备
- 装备显示属性变化（攻击力、防御力、生命值等）
- 目标：通过选择合适的装备，最大化总战斗力

**Agent 任务**:
1. 识别当前屏幕状态（砍树按钮位置、装备属性、战斗力数值）
2. 决策：点击砍树按钮
3. 装备判断：识别装备属性变化（↑ 上升 / ↓ 下降 / → 不变）
4. 装备选择：决定是否装备（战斗力↑则装备，战斗力↓则放弃）

**Reward 设计**:
- **战斗力上升**: +1.0 分
- **战斗力不变**: 0 分（容错点击失误）
- **战斗力下降**: -1.0 分（惩罚错误决策）
- **完成 10 次砍树**: 额外 bonus +2.0 分

---

## 2. 场景设计与可行性论证

### 2.1 为何选择砍树游戏？

✅ **优势分析**:

| 维度 | 优势 | 可行性评分 |
|------|------|-----------|
| **技术简单性** | 只需识别数值和箭头符号，OCR 难度低 | ⭐⭐⭐⭐⭐ |
| **数据收集成本** | 20-30 张截图即可构建完整数据集 | ⭐⭐⭐⭐⭐ |
| **Reward 明确性** | 战斗力数值直接可读，无需复杂验证逻辑 | ⭐⭐⭐⭐⭐ |
| **调试友好性** | 每次交互结果立即可见，便于调试 | ⭐⭐⭐⭐⭐ |
| **训练效率** | 单次游戏 10 步完成，episode 短，训练快 | ⭐⭐⭐⭐⭐ |
| **学术价值** | 涉及多模态理解+序列决策，可发表 | ⭐⭐⭐⭐ |
| **无侵权风险** | 简单小游戏，无知识产权争议 | ⭐⭐⭐⭐⭐ |

### 2.2 与外卖场景对比

| 对比项 | 外卖任务 | 砍树游戏 | 选择 |
|--------|----------|----------|------|
| 数据收集 | 需 100+ 条轨迹 | 只需 20-30 张截图 | ✅ 砍树 |
| Reward 计算 | 需 OCR + 界面状态检测 | 直接读数值 | ✅ 砍树 |
| 调试难度 | 多步骤，难定位问题 | 单步反馈，易调试 | ✅ 砍树 |
| 训练时间 | ~10 天 | ~2-3 天 | ✅ 砍树 |
| POC 适用性 | 较复杂 | 完美适配 | ✅ 砍树 |

**结论**: 砍树游戏是理想的 POC 场景，验证完成后可扩展至复杂任务。

### 2.3 关键技术挑战及解决方案

| 挑战 | 解决方案 | 实现难度 |
|------|----------|----------|
| **识别装备属性变化** | 使用 PaddleOCR 提取数值 + 简单的箭头符号识别 | ⭐⭐ (简单) |
| **点击位置精准性** | 基于 UI Automator 或固定坐标 | ⭐ (极简单) |
| **Rollout 中集成 Android** | 修改 vLLM rollout，每次生成后调用 ADB 执行 | ⭐⭐⭐ (中等) |
| **Reward 函数稳定性** | 缓存 OCR 结果 + 异常处理 | ⭐⭐ (简单) |

---

## 3. 技术架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    训练主循环 (Ray Trainer)                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Actor       │    │  Rollout     │    │  Reward      │  │
│  │  (FSDP)      │◄───│  (vLLM)      │───►│  Function    │  │
│  │              │    │              │    │              │  │
│  │ Qwen2.5-VL-7B│    │ + Android    │    │ OCR + Parser │  │
│  └──────────────┘    │   Executor   │    └──────────────┘  │
│                      └───────┬──────┘                        │
│                              │                               │
│                              ▼                               │
│                    ┌─────────────────┐                       │
│                    │  Android Device  │                      │
│                    │                  │                      │
│                    │  ┌────────────┐  │                      │
│                    │  │ 微信小程序  │  │                      │
│                    │  │ 砍树游戏    │  │                      │
│                    │  └────────────┘  │                      │
│                    │                  │                      │
│                    │  ADB Connection  │                      │
│                    └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据流

```
1. Prompt + 当前截图 → Qwen2.5-VL
   ↓
2. 模型输出动作 → "<action>click(320, 480)</action>"
   ↓
3. Android Executor 执行点击
   ↓
4. 截图新界面
   ↓
5. OCR 提取战斗力数值
   ↓
6. Reward 函数计算奖励 → +1.0 / 0 / -1.0
   ↓
7. GRPO 更新策略
```

### 3.3 目录结构

```
wechat_tree_game_agent/
├── README.md                          # 本文档
├── IMPLEMENTATION_GUIDE.md            # 详细实施指南
│
├── config/
│   └── tree_game_grpo.yaml            # GRPO 训练配置
│
├── data/
│   ├── collect_screenshots.py         # 截图收集工具
│   ├── process_dataset.py             # 数据集生成脚本
│   ├── raw_screenshots/               # 原始截图目录
│   └── tree_game_dataset.jsonl        # 生成的训练数据
│
├── android_env/
│   ├── __init__.py
│   ├── adb_controller.py              # ADB 设备控制
│   ├── screenshot_capture.py          # 截图捕获
│   ├── action_executor.py             # 动作执行器
│   └── game_state_parser.py           # 游戏状态解析（OCR）
│
├── reward_function/
│   └── tree_game_reward.py            # 战斗力奖励函数
│
├── format_prompt/
│   └── tree_game_action.jinja         # 动作生成 Prompt 模板
│
├── tests/
│   ├── test_adb_connection.py         # 测试 ADB 连接
│   ├── test_ocr_parser.py             # 测试 OCR 解析
│   └── test_reward_function.py        # 测试 Reward 函数
│
├── scripts/
│   ├── setup_environment.sh           # 环境安装脚本
│   ├── collect_data.sh                # 数据收集脚本
│   └── train.sh                       # 训练启动脚本
│
└── checkpoints/                       # 模型检查点（训练时生成）
```

---

## 4. 数据集构建

### 4.1 数据收集流程

#### **Step 1: 准备工作**

```bash
# 1. 连接 Android 设备
adb devices

# 2. 安装微信（如未安装）
adb install wechat.apk

# 3. 手动打开砍树小游戏
# 记录包名和 Activity（用于自动化）
adb shell dumpsys window | grep mCurrentFocus
```

#### **Step 2: 收集关键截图**

只需收集 **20-30 张截图**，覆盖以下场景：

| 截图类型 | 数量 | 说明 |
|----------|------|------|
| 游戏首页（砍树按钮可见） | 3-5 张 | 不同战斗力状态 |
| 砍树后掉落装备界面 | 10-15 张 | 各种属性组合（攻击↑防御↓等） |
| 装备后的结果界面 | 5-10 张 | 战斗力变化后的状态 |

**收集命令**:

```bash
# 运行自动收集脚本
python data/collect_screenshots.py \
    --device emulator-5554 \
    --output data/raw_screenshots/ \
    --count 25
```

#### **Step 3: 数据标注**

使用半自动工具标注：

```json
// data/raw_screenshots/annotations.json
[
  {
    "image": "screenshot_001.jpg",
    "state": "tree_cutting",
    "combat_power": 1250,
    "action": "click",
    "click_coords": [360, 800],
    "description": "点击砍树按钮"
  },
  {
    "image": "screenshot_002.jpg",
    "state": "equipment_dropped",
    "equipment_stats": {
      "attack": "+50 ↑",
      "defense": "-10 ↓",
      "hp": "+20 ↑"
    },
    "estimated_power_change": "+60",
    "ground_truth_action": "equip",
    "description": "总体战斗力预期上升，应该装备"
  }
]
```

### 4.2 数据集生成

```bash
# 自动生成 JSONL 格式训练数据
python data/process_dataset.py \
    --input data/raw_screenshots/annotations.json \
    --output data/tree_game_dataset.jsonl
```

**生成格式**:

```jsonl
{"prompt": "你是一个砍树游戏助手，根据当前截图决定下一步操作。如果出现装备，判断是否应该装备（总战斗力上升则装备）。", "images": ["raw_screenshots/screenshot_001.jpg"], "answer": "<action>click(360, 800)</action>", "combat_power_before": 1250}
{"prompt": "你是一个砍树游戏助手，根据当前截图决定下一步操作。如果出现装备，判断是否应该装备（总战斗力上升则装备）。", "images": ["raw_screenshots/screenshot_002.jpg"], "answer": "<action>equip()</action>", "combat_power_before": 1250, "combat_power_after": 1310, "power_change": 60}
```

### 4.3 数据集规模

| 数据类型 | 数量 | 用途 |
|----------|------|------|
| 训练集 | 80% (16-24 条) | GRPO 训练 |
| 验证集 | 20% (4-6 条) | 评估指标 |

**重要**: 由于是 POC，小样本即可验证方案可行性。

---

## 5. 训练流程

### 5.1 环境配置

```bash
# 1. 安装依赖
cd /path/to/EasyR1
pip install -r requirements.txt
pip install paddleocr adb-shell pillow  # Android 交互依赖

# 2. 验证 GPU
nvidia-smi

# 3. 测试 ADB 连接
python wechat_tree_game_agent/tests/test_adb_connection.py

# 4. 测试 OCR
python wechat_tree_game_agent/tests/test_ocr_parser.py
```

### 5.2 预训练（可选）

如果从零开始训练效果不佳，可先用通用 GUI 数据集进行 SFT：

```bash
# 使用 AndroidControl 等数据集预训练
python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=android_control.jsonl \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.total_epochs=3 \
    trainer.experiment_name=sft_pretrain
```

### 5.3 GRPO 训练

```bash
# 主训练命令
bash wechat_tree_game_agent/scripts/train.sh

# 或直接运行
python -m verl.trainer.main \
    config=wechat_tree_game_agent/config/tree_game_grpo.yaml \
    trainer.logger='["wandb", "file"]' \
    trainer.experiment_name=tree_game_grpo_v1
```

**关键配置参数**:

```yaml
# wechat_tree_game_agent/config/tree_game_grpo.yaml

data:
  train_files: wechat_tree_game_agent/data/tree_game_dataset.jsonl
  rollout_batch_size: 16  # 小批次，因为需要真实设备交互

algorithm:
  adv_estimator: grpo
  kl_coef: 0.01

worker:
  rollout:
    n: 4  # 每个 prompt 生成 4 个候选动作
    temperature: 0.7
    android_device_id: "emulator-5554"
    enable_real_android: true  # 关键：启用真实 Android 交互

  reward:
    reward_function: wechat_tree_game_agent/reward_function/tree_game_reward.py:compute_score

trainer:
  total_epochs: 10
  n_gpus_per_node: 4
  val_freq: 2
```

### 5.4 训练监控

**WandB Dashboard**:
- `reward/overall`: 总体奖励（目标：持续上升）
- `reward/combat_power_gain`: 战斗力提升值
- `metrics/equipment_accuracy`: 装备决策准确率（目标 >80%）
- `metrics/action_format_correct`: 动作格式正确率（目标 >95%）

**日志示例**:

```
[Epoch 1] reward/overall: 0.35, equipment_accuracy: 0.62
[Epoch 3] reward/overall: 0.58, equipment_accuracy: 0.75
[Epoch 5] reward/overall: 0.72, equipment_accuracy: 0.83
[Epoch 8] reward/overall: 0.81, equipment_accuracy: 0.91
```

---

## 6. 评估体系

### 6.1 自动化评估指标

| 指标 | 定义 | 目标值 | 计算方式 |
|------|------|--------|----------|
| **任务完成率** | 完成 10 次砍树的比例 | >90% | 完成次数 / 总尝试次数 |
| **战斗力提升** | 平均战斗力增长 | >+200 | Avg(最终战斗力 - 初始战斗力) |
| **装备准确率** | 正确装备决策比例 | >85% | 正确决策数 / 总装备出现次数 |
| **动作格式正确率** | 输出符合格式的比例 | >95% | 格式正确数 / 总生成次数 |
| **无效点击率** | 点击失误的比例 | <5% | 失败点击数 / 总点击数 |

### 6.2 评估脚本

```python
# wechat_tree_game_agent/evaluate.py

def evaluate_model(model_path, test_episodes=20):
    results = {
        "completion_rate": 0.0,
        "avg_power_gain": 0.0,
        "equipment_accuracy": 0.0,
        "format_correct_rate": 0.0
    }

    for episode in range(test_episodes):
        # 重置游戏环境
        android_env.reset_game()

        initial_power = android_env.get_combat_power()
        correct_decisions = 0
        total_decisions = 0

        for step in range(10):  # 10 次砍树
            screenshot = android_env.capture_screenshot()
            action = model.generate(screenshot)

            # 验证格式
            if validate_action_format(action):
                results["format_correct_rate"] += 1

            # 执行并评估
            android_env.execute(action)

            if android_env.has_equipment():
                total_decisions += 1
                if is_correct_decision(action, android_env.get_equipment_stats()):
                    correct_decisions += 1

        final_power = android_env.get_combat_power()
        results["avg_power_gain"] += (final_power - initial_power)
        results["equipment_accuracy"] += correct_decisions / max(total_decisions, 1)

        if android_env.game_completed():
            results["completion_rate"] += 1

    # 平均化
    for key in results:
        results[key] /= test_episodes

    return results
```

### 6.3 对比基线

| 模型 | 完成率 | 平均战斗力提升 | 装备准确率 |
|------|--------|----------------|-----------|
| **Random Policy** | 90% | +50 | 50% (随机选择) |
| **Rule-based** | 95% | +180 | 78% (基于简单规则) |
| **SFT Baseline** | 92% | +150 | 72% |
| **GRPO (目标)** | >95% | >+200 | >85% |

---

## 7. 实施时间表

### 7.1 详细计划（7 天完成 POC）

| 日期 | 任务 | 交付物 | 负责人 | 状态 |
|------|------|--------|--------|------|
| **Day 1** | 环境搭建 + Android 连接测试 | `test_adb_connection.py` 通过 | - | ⏳ Pending |
| **Day 2** | 数据收集工具开发 + 收集 25 张截图 | `raw_screenshots/` 目录 | - | ⏳ Pending |
| **Day 3** | OCR 解析 + Reward 函数开发 | `tree_game_reward.py` | - | ⏳ Pending |
| **Day 4** | 数据集生成 + 配置文件完成 | `tree_game_dataset.jsonl` | - | ⏳ Pending |
| **Day 5** | Rollout 修改 + 集成 Android | 首次训练运行成功 | - | ⏳ Pending |
| **Day 6** | GRPO 训练 + 调试 | 训练 Loss 收敛 | - | ⏳ Pending |
| **Day 7** | 模型评估 + 撰写报告 | 最终评估结果 + 技术报告 | - | ⏳ Pending |

### 7.2 里程碑

- **Milestone 1** (Day 3): 完成数据收集和 Reward 函数，可独立测试
- **Milestone 2** (Day 5): 训练 Pipeline 打通，成功运行 1 个 epoch
- **Milestone 3** (Day 7): 模型在测试集上装备准确率 >80%

---

## 8. 风险控制

### 8.1 风险清单与缓解措施

| 风险 | 概率 | 影响 | 缓解方案 | 负责人 |
|------|------|------|----------|--------|
| **Android 设备连接不稳定** | 中 | 高 | 使用稳定的 emulator；实现自动重连机制 | - |
| **OCR 识别错误** | 高 | 中 | 多次识别取众数；手动校验 5% 样本 | - |
| **训练不收敛** | 中 | 高 | 调整学习率；增加数据增强；回退到 SFT | - |
| **小程序更新导致界面变化** | 低 | 中 | 保存游戏版本号；使用本地测试版本 | - |
| **GPU 资源不足** | 低 | 中 | 使用 7B 模型（单卡可训练）；降低 batch size | - |

### 8.2 应急预案

**场景 1: Android 交互失败**
- **Plan A**: 使用模拟环境（生成合成截图）
- **Plan B**: 切换到 iOS 设备（使用 tidevice）

**场景 2: GRPO 训练效果不佳**
- **Plan A**: 先用 SFT 训练基线，再进行 RL 微调
- **Plan B**: 尝试 RLOO 或 ReMax 算法
- **Plan C**: 增加人工演示数据至 50 条

**场景 3: OCR 完全不可用**
- **Plan A**: 使用 UI Automator 直接读取 UI 元素文本
- **Plan B**: 训练简单的数字识别模型（MNIST-like）

---

## 9. 预期成果

### 9.1 技术成果

1. **可复用的 Android RL 训练框架**
   - 支持任意微信小程序的 Agent 训练
   - 开源代码和配置文件

2. **完整的训练流程验证**
   - 证明 GRPO 在真实 Android 环境下的有效性
   - 提供小样本训练的最佳实践

3. **性能提升证明**
   - 装备准确率：Random 50% → GRPO 85%+
   - 战斗力提升：Random +50 → GRPO +200+

### 9.2 学术价值

**潜在论文贡献点**:
1. **真实环境 RL**: 不同于模拟器，使用真实 Android 设备进行 Rollout
2. **极小样本学习**: 仅用 20-30 张截图即可训练有效策略
3. **视觉决策**: 结合 OCR 和 VLM 的多模态决策框架
4. **快速原型验证**: 7 天完成完整 RL 训练闭环

**可投稿方向**:
- EMNLP Demo Track
- ICML Workshop on RL
- NeurIPS Dataset and Benchmark Track

---

## 10. 扩展方向

完成砍树游戏 POC 后，可扩展至：

1. **更复杂的微信游戏**
   - 跳一跳（需要力度控制）
   - 消消乐（需要组合优化）

2. **实用工具自动化**
   - 自动签到
   - 自动领取优惠券
   - 自动回复（客服机器人）

3. **迁移至其他平台**
   - iOS 设备（使用 tidevice）
   - 桌面应用（使用 pyautogui）
   - Web 应用（使用 Playwright）

---

## 11. 参考资料

### 11.1 学术论文

1. **GRPO**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
2. **Mobile-R1**: [Towards Interactive RL for VLM-Based Mobile Agent](https://arxiv.org/abs/2506.20332)
3. **UI-R1**: [Enhancing Action Prediction of GUI Agents by RL](https://arxiv.org/abs/2503.21620)

### 11.2 开源项目

1. **EasyR1**: https://github.com/hiyouga/EasyR1
2. **Qwen2.5-VL**: https://github.com/QwenLM/Qwen2-VL
3. **PaddleOCR**: https://github.com/PaddlePaddle/PaddleOCR
4. **Android UI Automator**: https://developer.android.com/training/testing/ui-automator

### 11.3 工具文档

1. **ADB Commands**: https://developer.android.com/tools/adb
2. **vLLM Documentation**: https://docs.vllm.ai/
3. **WandB Integration**: https://docs.wandb.ai/

---

## 12. FAQ

**Q1: 为什么只需要 20-30 张截图？**
> A: 砍树游戏状态空间小（只有"砍树"和"装备选择"两类），且 GRPO 通过 Rollout 在线采样，不依赖大量离线数据。

**Q2: 如果没有真实 Android 设备怎么办？**
> A: 可以使用 Android Emulator（推荐 Genymotion 或 Android Studio 自带），通过 ADB 连接方式完全相同。

**Q3: 训练需要多少 GPU？**
> A: Qwen2.5-VL-7B 在 4×40GB GPU 上即可训练，如果使用 BF16 优化，2×40GB 即可。

**Q4: 如何验证 OCR 识别准确性？**
> A: 运行 `tests/test_ocr_parser.py`，手动标注 10 张测试图，要求准确率 >95%。

**Q5: 训练不收敛怎么办？**
> A: 先用 SFT 训练基线，确保模型能输出正确格式，再进行 GRPO；降低学习率至 1e-7。

---

## 13. 联系方式

- **项目负责人**: [待填写]
- **技术支持**: 参考 EasyR1 Issues
- **邮件**: [待填写]

---

**版本**: v1.0
**最后更新**: 2025-01-17
**文档状态**: ✅ 已完成论证，待执行
