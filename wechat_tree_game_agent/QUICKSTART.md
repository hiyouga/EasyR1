# 快速开始指南

> **10 分钟快速验证方案可行性**

---

## 🚀 快速验证步骤

### Step 1: 安装依赖（2 分钟）

```bash
# 安装基础依赖
pip install adb-shell pillow

# 可选：安装 OCR（如果需要真实解析）
pip install paddleocr paddlepaddle
```

### Step 2: 测试 ADB 连接（3 分钟）

```bash
# 1. 连接 Android 设备或启动模拟器
adb devices

# 2. 运行连接测试
python wechat_tree_game_agent/tests/test_adb_connection.py

# 预期输出:
# ✓ 设备连接成功
# ✓ 屏幕分辨率: 1080x2400
# ✓ 截图成功
# ✓ 点击成功
```

### Step 3: 测试 Reward 函数（2 分钟）

```bash
# 运行 Reward 函数单元测试
python wechat_tree_game_agent/tests/test_reward_function.py

# 预期输出:
# [测试 1] 战斗力上升 (正确装备)
#   总奖励: 2.50 ✓
# [测试 2] 战斗力下降 (错误装备)
#   总奖励: -1.50 ✓
# ...
# 所有测试通过！✅
```

### Step 4: 收集 5 张测试截图（3 分钟）

```bash
# 手动打开砍树游戏
# 运行截图收集工具
python wechat_tree_game_agent/data/collect_screenshots.py \
    --device emulator-5554 \
    --count 5 \
    --interval 3

# 在收集过程中手动操作游戏
```

---

## ✅ 验证完成

如果以上 4 步全部通过，说明方案完全可行！

**下一步**:
- 阅读 [README.md](./README.md) 了解完整方案
- 阅读 [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) 开始 7 天实施计划

---

## 🆘 如果遇到问题

### 问题 1: `adb: command not found`

```bash
# macOS
brew install android-platform-tools

# Linux (Ubuntu/Debian)
sudo apt install adb

# Windows
# 下载 Android SDK Platform-Tools
```

### 问题 2: `设备未连接`

```bash
# 重启 ADB
adb kill-server
adb start-server

# 检查设备
adb devices
```

### 问题 3: `PaddleOCR 安装失败`

```bash
# 暂时跳过 OCR，使用模拟数据测试
# 后续可以使用 UI Automator 直接读取文本
```

---

## 📁 项目结构

```
wechat_tree_game_agent/
├── README.md                    # 完整方案文档（已读）
├── IMPLEMENTATION_GUIDE.md      # 详细实施指南（必读）
├── QUICKSTART.md               # 本文件（快速验证）
│
├── config/
│   └── tree_game_grpo.yaml     # GRPO 训练配置
│
├── android_env/                # Android 交互模块
│   ├── adb_controller.py       # ADB 控制器
│   └── game_state_parser.py    # OCR 解析器
│
├── reward_function/            # Reward 函数
│   └── tree_game_reward.py
│
├── data/                       # 数据收集工具
│   ├── collect_screenshots.py
│   └── process_dataset.py
│
├── scripts/                    # 启动脚本
│   ├── setup_environment.sh
│   └── train.sh
│
└── tests/                      # 单元测试
    ├── test_adb_connection.py
    └── test_reward_function.py
```

---

## 🎯 关键文件说明

| 文件 | 作用 | 何时使用 |
|------|------|----------|
| `README.md` | 完整方案文档 | 了解项目全貌 |
| `IMPLEMENTATION_GUIDE.md` | 7 天实施计划 | 开始执行前必读 |
| `config/tree_game_grpo.yaml` | 训练配置 | 训练前检查/调整 |
| `reward_function/tree_game_reward.py` | Reward 逻辑 | 核心逻辑，需理解 |
| `scripts/train.sh` | 训练启动脚本 | 训练时运行 |

---

**准备好了吗？让我们开始 Day 1！** 🚀

👉 打开 [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) 查看详细步骤
