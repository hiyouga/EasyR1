# Number Game Agent Cookbook

这个目录包含了数字选择游戏相关的工具和脚本。

## 📁 目录结构

```
cookbook/
├── README.md              # 本文件
├── adb_controller.py      # ADB设备控制器（共用）
├── vlm_client.py          # VLM模型客户端
├── play_agent.py          # 自动玩游戏的Agent（入口脚本）
└── collect_data.py        # 数据收集脚本（入口脚本）
```

## 🔧 模块说明

### `adb_controller.py`
Android设备控制器，提供ADB操作的封装。

**主要功能：**
- 设备连接检查
- 屏幕截图
- 屏幕点击
- 获取屏幕分辨率

**被使用于：**
- `play_agent.py`
- `collect_data.py`

### `vlm_client.py`
VLM（视觉语言模型）客户端，支持Ollama和vLLM两种模型服务。

**主要功能：**
- 图像转base64编码
- 查询Ollama API
- 查询vLLM API

**被使用于：**
- `play_agent.py`

### `play_agent.py` ⭐
自动玩数字选择游戏的Agent（入口脚本）。

**主要功能：**
- 使用VLM识别游戏状态和指示灯
- 自动做出决策并执行操作
- 记录游戏结果和截图

**使用方法：**
```bash
cd examples/number_game_agent/cookbook

# 基本用法
python play_agent.py \
    --model-type ollama \
    --api-url http://localhost:11434 \
    --model-name qwen2.5vl:3b \
    --devices 101.43.137.83:5555

# 完整参数
python play_agent.py \
    --model-type vllm \
    --api-url http://localhost:8000 \
    --model-name Qwen/Qwen2.5-VL-3B \
    --devices 101.43.137.83:5555 192.168.1.100:5555 \
    --screenshot-dir game_screenshots \
    --episodes 3 \
    --debug
```

### `collect_data.py` ⭐
数据收集脚本（入口脚本），用于离线训练数据集构建。

**主要功能：**
- 支持多设备并发收集游戏截图
- 截图命名格式规范
- 只收集截图，不调用VLM（节省时间和资源）
- 自动重试失败的轮次
- 记录每局游戏的元数据

**使用方法：**
```bash
cd examples/number_game_agent/cookbook

# 基本用法
python collect_data.py \
    --devices 101.43.137.83:5555

# 并发收集
python collect_data.py \
    --devices 101.43.137.83:5555 192.168.1.100:5555 \
    --episodes 20 \
    --output-dir game_data_raw \
    --parallel \
    --max-workers 4 \
    --debug
```

## 📝 参数说明

### `play_agent.py` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-type` | str | ollama | 模型服务类型: ollama 或 vllm |
| `--api-url` | str | http://localhost:11434 | 模型API地址 |
| `--model-name` | str | qwen2.5vl:3b | 模型名称 |
| `--devices` | list | 101.43.137.83:5555 | Android设备地址列表 |
| `--screenshot-dir` | str | game_screenshots | 截图保存目录 |
| `--episodes` | int | 1 | 每个设备运行几局游戏 |
| `--parallel` | flag | False | 并发处理多个设备 |
| `--debug` | flag | False | 开启调试模式 |

### `collect_data.py` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--devices` | list | **必需** | Android设备地址列表 |
| `--episodes` | int | 10 | 每个设备收集多少局游戏 |
| `--output-dir` | str | game_data_raw | 输出目录 |
| `--parallel` | flag | False | 并发执行多个设备 |
| `--max-workers` | int | 4 | 并发执行时的最大线程数 |
| `--debug` | flag | False | 开启调试模式 |

## 🚀 快速开始

### 1. 准备环境
```bash
# 安装依赖
pip install pillow requests

# 确保ADB可用
adb devices
```

### 2. 收集数据
```bash
cd examples/number_game_agent/cookbook
python collect_data.py --devices YOUR_DEVICE_IP:5555 --episodes 5
```

### 3. 使用Agent玩游戏
```bash
cd examples/number_game_agent/cookbook
python play_agent.py \
    --model-type ollama \
    --model-name qwen2.5vl:3b \
    --devices YOUR_DEVICE_IP:5555
```

## 📦 依赖项

- Python 3.8+
- PIL (Pillow)
- requests
- Android Debug Bridge (ADB)
- Ollama 或 vLLM (仅 play_agent.py 需要)

## 🔄 迁移说明

如果你之前使用的是旧的目录结构，这些文件已经整合到新结构中：

- `android_env/adb_controller.py` → `adb_controller.py`
- `agent/number_game_play_agent.py` → `play_agent.py` (包含 VLMClient)
- `collect_data_from_android/collect_game_data_from_android.py` → `collect_data.py`

所有代码逻辑保持不变，只是目录结构更加清晰简洁。
