#!/bin/bash
# 环境安装脚本

set -e

echo "=========================================="
echo "微信砍树游戏 Agent 环境安装"
echo "=========================================="

# 检查 Python 版本
echo ""
echo "[1/5] 检查 Python 版本..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "当前 Python 版本: $PYTHON_VERSION"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "错误: 需要 Python 3.9+"
    exit 1
fi

# 安装 EasyR1 依赖
echo ""
echo "[2/5] 安装 EasyR1 依赖..."
pip install -r requirements.txt

# 安装 Android 交互依赖
echo ""
echo "[3/5] 安装 Android 交互依赖..."
pip install adb-shell pillow

# 安装 PaddleOCR (可选，如果需要 OCR)
echo ""
echo "[4/5] 安装 PaddleOCR (可选)..."
read -p "是否安装 PaddleOCR？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install paddleocr paddlepaddle
    echo "✓ PaddleOCR 安装完成"
else
    echo "跳过 PaddleOCR 安装（可使用模拟数据测试）"
fi

# 测试 ADB 连接
echo ""
echo "[5/5] 测试 ADB 连接..."
if command -v adb &> /dev/null; then
    echo "✓ ADB 已安装"
    echo ""
    echo "可用设备:"
    adb devices
else
    echo "⚠ ADB 未安装"
    echo "请安装 Android SDK Platform-Tools"
    echo "  macOS: brew install android-platform-tools"
    echo "  Linux: apt install adb"
fi

echo ""
echo "=========================================="
echo "环境安装完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 连接 Android 设备: adb devices"
echo "  2. 测试连接: python wechat_tree_game_agent/android_env/adb_controller.py"
echo "  3. 收集数据: python wechat_tree_game_agent/data/collect_screenshots.py"
echo ""
