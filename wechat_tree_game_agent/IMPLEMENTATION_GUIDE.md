# 微信砍树游戏 Agent - 详细实施指南

> **预计完成时间**: 7 天
>
> **难度**: ⭐⭐⭐ (中等)

---

## 📅 Day 1: 环境搭建与设备连接

### 1.1 安装依赖

```bash
# 进入项目目录
cd /path/to/EasyR1

# 运行自动安装脚本
bash wechat_tree_game_agent/scripts/setup_environment.sh

# 或手动安装
pip install -r requirements.txt
pip install adb-shell pillow paddleocr paddlepaddle
```

### 1.2 配置 Android 设备

#### 选项 A: 使用真机

```bash
# 1. 启用开发者选项
# 设置 → 关于手机 → 连续点击「版本号」7次

# 2. 启用 USB 调试
# 设置 → 开发者选项 → USB 调试（开启）

# 3. 连接电脑并授权
adb devices
# List of devices attached
# XXXXXXXX    device

# 记录设备 ID
export DEVICE_ID="XXXXXXXX"
```

#### 选项 B: 使用模拟器（推荐）

```bash
# 使用 Android Studio 模拟器
# 1. 安装 Android Studio
# 2. AVD Manager → Create Virtual Device
# 3. 选择 Pixel 6 (1080x2400)
# 4. 启动模拟器

adb devices
# emulator-5554    device

export DEVICE_ID="emulator-5554"
```

### 1.3 测试连接

```bash
# 测试 ADB 控制器
python wechat_tree_game_agent/android_env/adb_controller.py $DEVICE_ID

# 预期输出:
# ✓ 设备 emulator-5554 已连接
# [测试 1] 获取屏幕分辨率
# 屏幕分辨率: 1080x2400
# [测试 2] 截取截图
# ✓ 截图已保存: test_screenshot.png
# [测试 3] 点击屏幕中心
# ✓ 点击坐标: (540, 1200)
```

### 1.4 安装微信和砍树游戏

```bash
# 安装微信（如果没有）
adb -s $DEVICE_ID install wechat.apk

# 手动步骤:
# 1. 打开微信
# 2. 搜索「砍树」小程序（根据实际游戏名称）
# 3. 记录小程序包名（用于自动化）
adb shell dumpsys window | grep mCurrentFocus
```

**✅ Day 1 检查点**:
- [ ] ADB 连接成功
- [ ] 能够截图和点击
- [ ] 微信和游戏已安装

---

## 📅 Day 2: 数据收集

### 2.1 手动游戏熟悉

先手动玩 5-10 局游戏，了解：
- 砍树按钮的位置
- 装备属性显示方式
- 战斗力数值位置
- 游戏流程

### 2.2 自动截图收集

```bash
# 启动自动收集工具
python wechat_tree_game_agent/data/collect_screenshots.py \
    --device $DEVICE_ID \
    --output wechat_tree_game_agent/data/raw_screenshots \
    --count 25 \
    --interval 3

# 运行过程中，手动操作游戏:
# 1. 点击砍树
# 2. 等待装备掉落
# 3. 查看装备属性
# 4. 装备或跳过
# 5. 重复 10 次完成一局
```

**工具会自动**:
- 每 3 秒截图一次
- 使用 OCR 识别战斗力和装备属性
- 生成 `annotations.json` 文件

### 2.3 手动标注

```bash
# 查看标注指南
python wechat_tree_game_agent/data/collect_screenshots.py --guide

# 编辑标注文件
vim wechat_tree_game_agent/data/raw_screenshots/annotations.json
# 或使用任意文本编辑器
```

**标注示例**:

```json
{
  "id": 1,
  "image": "screenshot_001.jpg",
  "state": "tree_cutting",
  "combat_power": 1250,
  "manual_annotation": {
    "action": "click(360, 800)",
    "description": "点击砍树按钮"
  }
},
{
  "id": 2,
  "image": "screenshot_002.jpg",
  "state": "equipment_selection",
  "combat_power": 1250,
  "equipment_stats": {"attack": "+50 ↑", "defense": "-10 ↓"},
  "estimated_power_change": 42,
  "manual_annotation": {
    "action": "equip()",
    "description": "总战斗力+42，应该装备"
  }
}
```

**标注质量检查**:
- [ ] 至少 20 张有效截图
- [ ] 覆盖「砍树」和「装备选择」两种状态
- [ ] 装备选择包含「应该装备」和「应该跳过」两类
- [ ] 每个截图的 `action` 字段已填写

### 2.4 生成训练数据集

```bash
# 生成 JSONL 格式数据集
python wechat_tree_game_agent/data/process_dataset.py \
    --input wechat_tree_game_agent/data/raw_screenshots/annotations.json \
    --output wechat_tree_game_agent/data/ \
    --val-ratio 0.2

# 输出:
# ✓ 训练集: 20 条
# ✓ 验证集: 5 条
```

**验证数据集**:

```bash
# 查看训练数据
head -n 2 wechat_tree_game_agent/data/tree_game_dataset.jsonl

# 预期格式:
# {"prompt": "...", "images": ["..."], "answer": "<action>click(360, 800)</action>", ...}
```

**✅ Day 2 检查点**:
- [ ] 收集 25+ 张截图
- [ ] 完成手动标注
- [ ] 生成训练数据集（20+ 条）

---

## 📅 Day 3: Reward 函数开发与测试

### 3.1 测试 OCR 解析器

```bash
# 运行单元测试
python wechat_tree_game_agent/android_env/game_state_parser.py

# 预期输出:
# [测试 1] 解析战斗力
# 识别到战斗力: 1250.0
# [测试 2] 解析装备属性
# 装备属性: {'attack': '+50 ↑', 'defense': '-10 ↓', 'hp': '+20 ↑'}
# [测试 3] 估算战斗力变化
# 估算战斗力变化: 52
```

### 3.2 测试 Reward 函数

```bash
# 运行 Reward 函数测试
python wechat_tree_game_agent/reward_function/tree_game_reward.py

# 预期输出:
# [测试 1] 战斗力上升 (正确装备)
#   总奖励: 2.50
#   - 战斗力变化: 2.50
#   - 格式正确: 1.00
#   预期: >1.0 ✓
```

### 3.3 使用真实截图测试

```bash
# 在 Python 交互环境中测试
python

>>> from wechat_tree_game_agent.android_env import GameStateParser
>>> from wechat_tree_game_agent.reward_function.tree_game_reward import compute_score
>>> from PIL import Image

>>> # 加载一张真实截图
>>> parser = GameStateParser(use_gpu=False)
>>> image = Image.open("wechat_tree_game_agent/data/raw_screenshots/screenshot_002.jpg")
>>> parsed = parser.parse_screenshot(image)

>>> # 查看解析结果
>>> print(parsed)
# {'state': 'equipment_selection', 'combat_power': 1250.0, 'equipment_stats': {...}}

>>> # 测试 Reward 计算
>>> reward_input = {
...     "response": "<action>equip()</action>",
...     "combat_power_before": 1250,
...     "combat_power_after": 1310
... }
>>> result = compute_score([reward_input])
>>> print(result[0])
# {'overall': 2.6, 'power_change': 2.6, ...}
```

### 3.4 调优 OCR 准确性

如果 OCR 识别不准确：

```python
# 调整 OCR 参数
parser = GameStateParser(use_gpu=True)  # 使用 GPU 提升速度和准确性

# 或使用图像预处理
from PIL import ImageEnhance

def preprocess_image(image):
    # 增强对比度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    return image
```

**✅ Day 3 检查点**:
- [ ] OCR 准确率 >90%（手动验证 5-10 张截图）
- [ ] Reward 函数所有测试通过
- [ ] 能够正确解析真实游戏截图

---

## 📅 Day 4: 训练配置验证

### 4.1 检查训练配置

```bash
# 查看配置文件
cat wechat_tree_game_agent/config/tree_game_grpo.yaml

# 关键参数验证:
# - data.train_files: 路径正确
# - worker.rollout.n: 4 (GRPO 要求)
# - trainer.n_gpus_per_node: 根据实际 GPU 数量调整
```

### 4.2 修改配置（可选）

如果 GPU 显存不足（<40GB）：

```yaml
# 降低批次大小
data:
  rollout_batch_size: 8  # 改为 8（默认 16）

worker:
  actor:
    global_batch_size: 32  # 改为 32（默认 64）
    fsdp:
      torch_dtype: bf16  # 使用 BF16 降低显存

  rollout:
    tensor_parallel_size: 2  # 使用张量并行（如果有多卡）
```

### 4.3 Dry Run（不实际训练）

```bash
# 验证数据加载和配置正确性
python -m verl.trainer.main \
    config=wechat_tree_game_agent/config/tree_game_grpo.yaml \
    trainer.val_only=true \
    trainer.val_before_train=true

# 预期输出:
# Loading dataset...
# ✓ Train: 20 samples
# ✓ Val: 5 samples
# Running validation...
# [Validation] reward/overall: 0.xx
```

**✅ Day 4 检查点**:
- [ ] 配置文件无语法错误
- [ ] 数据集能够正确加载
- [ ] 验证模式运行成功

---

## 📅 Day 5: 集成 Android Rollout（关键）

### 5.1 理解 Rollout 流程

GRPO 训练过程中，Rollout 阶段需要：

1. 模型生成动作：`<action>click(x,y)</action>`
2. **在 Android 设备上执行动作**（关键创新）
3. 截图获取新状态
4. OCR 识别战斗力变化
5. 计算 Reward

### 5.2 修改 vLLM Rollout

创建自定义 Rollout 类：

```bash
# 创建文件
vim wechat_tree_game_agent/android_env/android_rollout.py
```

**核心代码** (简化版):

```python
# wechat_tree_game_agent/android_env/android_rollout.py
from verl.workers.rollout.vllm_rollout_spmd import vLLMRollout
from .adb_controller import ADBController
from .game_state_parser import GameStateParser
import re

class AndroidGameRollout(vLLMRollout):
    """集成 Android 交互的 Rollout"""

    def __init__(self, *args, android_device_id="emulator-5554", **kwargs):
        super().__init__(*args, **kwargs)
        self.android = ADBController(device_id=android_device_id)
        self.parser = GameStateParser(use_gpu=False)

    def generate_sequences(self, data):
        # 1. 原有的 vLLM 生成
        outputs = super().generate_sequences(data)

        # 2. 对每个生成的动作序列，在 Android 上执行
        for i, output in enumerate(outputs):
            action_text = output.outputs[0].text

            # 解析动作
            action = self._parse_action(action_text)

            # 在 Android 上执行
            if action["type"] == "click":
                x, y = action["coords"]
                self.android.tap(x, y)

            # 截图并解析新状态
            screenshot = self.android.capture_screenshot()
            parsed = self.parser.parse_screenshot(screenshot)

            # 将结果添加到 data 中（供 Reward 函数使用）
            data.batch["combat_power_after"][i] = parsed["combat_power"]

        return outputs

    def _parse_action(self, text):
        # 解析 <action>click(x,y)</action>
        match = re.search(r"click\((\d+),\s*(\d+)\)", text)
        if match:
            return {
                "type": "click",
                "coords": [int(match.group(1)), int(match.group(2))]
            }
        return {"type": "unknown"}
```

### 5.3 测试 Android Rollout

```python
# 测试脚本
python

>>> from wechat_tree_game_agent.android_env.android_rollout import AndroidGameRollout
>>> rollout = AndroidGameRollout(
...     model_path="Qwen/Qwen2.5-VL-7B-Instruct",
...     config=...,  # Rollout 配置
...     android_device_id="emulator-5554"
... )

>>> # 测试生成和执行
>>> # (需要完整的训练环境)
```

**注意**: 这是 POC 的核心创新，需要仔细调试！

**✅ Day 5 检查点**:
- [ ] Android Rollout 类实现完成
- [ ] 能够在设备上执行生成的动作
- [ ] 能够获取动作后的截图和战斗力

---

## 📅 Day 6: GRPO 训练

### 6.1 启动训练

```bash
# 确保所有检查点都已通过
# 启动训练
bash wechat_tree_game_agent/scripts/train.sh

# 或直接运行
python -m verl.trainer.main \
    config=wechat_tree_game_agent/config/tree_game_grpo.yaml \
    trainer.experiment_name=tree_game_grpo_$(date +%Y%m%d_%H%M%S) \
    trainer.logger='["wandb", "file"]'
```

### 6.2 监控训练

#### WandB Dashboard

打开 https://wandb.ai，查看：

- **reward/overall**: 总体奖励（目标：持续上升）
  - Epoch 1: ~0.3
  - Epoch 5: ~0.6
  - Epoch 10: >0.7

- **reward/power_change**: 战斗力提升奖励
- **metrics/equipment_accuracy**: 装备决策准确率
- **loss/policy_loss**: 策略损失（应该下降）

#### 本地日志

```bash
# 查看训练日志
tail -f outputs/tree_game_grpo_*/train.log

# 预期输出:
# [Epoch 1/10] Step 10/20: reward/overall=0.35, loss=1.23
# [Epoch 1/10] Step 20/20: reward/overall=0.42, loss=1.15
# [Validation] equipment_accuracy=0.65
```

### 6.3 训练调试

如果遇到问题：

#### 问题 1: Reward 始终为负
```bash
# 检查 Reward 函数
python wechat_tree_game_agent/reward_function/tree_game_reward.py

# 检查数据标注是否正确
head wechat_tree_game_agent/data/tree_game_dataset.jsonl
```

#### 问题 2: 内存不足
```yaml
# 降低批次大小
data:
  rollout_batch_size: 8  # 降低
worker:
  actor:
    global_batch_size: 32  # 降低
```

#### 问题 3: Android 连接断开
```bash
# 重新连接
adb reconnect

# 检查设备状态
adb devices
```

### 6.4 中途保存

训练会自动保存 checkpoint:
```
wechat_tree_game_agent/checkpoints/tree_game_grpo_TIMESTAMP/
├── checkpoint-epoch-2/
├── checkpoint-epoch-4/
└── checkpoint-epoch-6/
```

**✅ Day 6 检查点**:
- [ ] 训练成功运行至少 5 个 epoch
- [ ] Reward 曲线上升
- [ ] 无频繁报错

---

## 📅 Day 7: 评估与报告

### 7.1 模型评估

```bash
# 使用最佳 checkpoint 进行评估
python wechat_tree_game_agent/evaluate.py \
    --model wechat_tree_game_agent/checkpoints/tree_game_grpo_*/checkpoint-epoch-best \
    --test-episodes 20 \
    --device $DEVICE_ID
```

**预期结果**:

| 指标 | SFT Baseline | GRPO (目标) | 实际结果 |
|------|--------------|-------------|----------|
| 完成率 | 92% | >95% | ___ % |
| 战斗力提升 | +150 | >+200 | +___ |
| 装备准确率 | 72% | >85% | ___ % |

### 7.2 案例分析

手动测试 5 局游戏：

```bash
# 启动交互式测试
python wechat_tree_game_agent/interactive_test.py \
    --model checkpoint-epoch-best \
    --device $DEVICE_ID
```

记录：
- 每局的决策过程
- 错误案例（为什么失败？）
- 成功案例（为什么成功？）

### 7.3 撰写技术报告

模板：

```markdown
# 微信砍树游戏 Agent GRPO 训练报告

## 1. 项目概述
- 目标：训练 Agent 在砍树游戏中最大化战斗力
- 方法：GRPO + Qwen2.5-VL-7B + 真实 Android 交互

## 2. 数据集
- 训练集：XX 条
- 验证集：XX 条
- 数据来源：人工演示 + 自动标注

## 3. 训练过程
- 训练时长：XX 小时
- 硬件：XX GPU (XX GB)
- 最佳 Epoch：XX

## 4. 评估结果
- 完成率：XX%
- 战斗力提升：+XX
- 装备准确率：XX%

## 5. 案例分析
[插入成功和失败案例的截图和分析]

## 6. 创新点
1. 首次在 GRPO 训练中集成真实 Android 交互
2. 仅用 XX 张截图实现有效训练
3. 多维度 Reward 函数设计

## 7. 未来改进
- 扩展至更复杂的游戏/应用
- 自动数据收集流程
- 多设备并行训练
```

**✅ Day 7 检查点**:
- [ ] 完成模型评估
- [ ] 装备准确率达到目标（>80%）
- [ ] 撰写完整技术报告

---

## 🚀 扩展方向

完成 POC 后，可以：

### 方向 1: 迁移至更复杂场景
- 外卖订购（原始计划）
- 微信自动回复
- 跳一跳游戏

### 方向 2: 优化训练效率
- 自动数据增强
- 多设备并行 Rollout
- 使用更小的模型（Qwen2-VL-3B）

### 方向 3: 发表论文
- 整理实验数据
- 对比其他方法（Random, Rule-based, SFT）
- 投稿 EMNLP Demo / NeurIPS Workshop

---

## ❓ 常见问题

### Q1: 如果没有 GPU 怎么办？
A: 可以使用 Google Colab 或 AWS/阿里云的 GPU 实例。Qwen2.5-VL-7B 需要至少 4×40GB GPU。

### Q2: Android 设备必须是真机吗？
A: 不是，推荐使用模拟器（更稳定，易于自动化）。

### Q3: 训练需要多久？
A: 小数据集（20 条），约 2-4 小时（取决于 GPU 性能）。

### Q4: 如果 OCR 识别不准确怎么办？
A:
- 使用 PaddleOCR GPU 版本
- 增强图像对比度
- 或直接使用 UI Automator 读取文本

### Q5: 训练不收敛怎么办？
A:
- 先用 SFT 训练基线
- 降低学习率（5e-7 → 1e-7）
- 增加训练数据至 50 条

---

## 📞 技术支持

- **EasyR1 Issues**: https://github.com/hiyouga/EasyR1/issues
- **Qwen2-VL Docs**: https://github.com/QwenLM/Qwen2-VL
- **PaddleOCR Docs**: https://github.com/PaddlePaddle/PaddleOCR

---

**祝训练顺利！明天开始执行吧！** 🎉
