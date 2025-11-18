"""
测试 ADB 连接和游戏控制

运行此脚本验证:
1. ADB 设备连接是否正常
2. 截图功能是否工作
3. 点击功能是否工作
4. OCR 识别是否正常
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wechat_tree_game_agent.android_env.adb_controller import ADBController
from wechat_tree_game_agent.android_env.game_state_parser import GameStateParser


def test_adb_connection(device_id: str):
    """测试 ADB 连接"""
    print("\n" + "=" * 60)
    print("[测试 1/4] ADB 设备连接")
    print("=" * 60)

    try:
        controller = ADBController(device_id=device_id)
        print("✓ ADB 连接成功")
        return controller
    except Exception as e:
        print(f"✗ ADB 连接失败: {e}")
        return None


def test_screenshot(controller: ADBController):
    """测试截图功能"""
    print("\n" + "=" * 60)
    print("[测试 2/4] 截图功能")
    print("=" * 60)

    try:
        screenshot = controller.capture_screenshot(save_path="test_screenshot.png")
        print(f"✓ 截图成功")
        print(f"  - 尺寸: {screenshot.size}")
        print(f"  - 保存路径: test_screenshot.png")
        return screenshot
    except Exception as e:
        print(f"✗ 截图失败: {e}")
        return None


def test_screen_info(controller: ADBController):
    """测试屏幕信息获取"""
    print("\n" + "=" * 60)
    print("[测试 3/4] 屏幕信息")
    print("=" * 60)

    try:
        width, height = controller.get_screen_resolution()
        print(f"✓ 屏幕分辨率: {width}x{height}")

        # 计算关键位置
        axe_x = width // 2
        axe_y = int(height * 5 / 6)
        print(f"  - 斧子位置（预估）: ({axe_x}, {axe_y})")

        replace_x = int(width * 0.65)
        replace_y = int(height * 0.65)
        print(f"  - 替换按钮（预估）: ({replace_x}, {replace_y})")

        decompose_x = int(width * 0.35)
        decompose_y = int(height * 0.65)
        print(f"  - 分解按钮（预估）: ({decompose_x}, {decompose_y})")

        return True
    except Exception as e:
        print(f"✗ 获取屏幕信息失败: {e}")
        return False


def test_ocr(screenshot):
    """测试 OCR 识别"""
    print("\n" + "=" * 60)
    print("[测试 4/4] OCR 识别")
    print("=" * 60)

    try:
        parser = GameStateParser(use_gpu=False)
        parsed = parser.parse_screenshot(screenshot)

        print(f"✓ OCR 解析成功")
        print(f"  - 游戏状态: {parsed['state']}")
        print(f"  - 妖力数值: {parsed['demon_power']}")

        if parsed['equipment_stats']:
            print(f"  - 装备属性:")
            for stat, value in parsed['equipment_stats'].items():
                print(f"    - {stat}: {value}")
            print(f"  - 估算妖力变化: {parsed['estimated_power_change']}")

        # 打印原始 OCR 文本（调试用）
        if parsed['raw_texts']:
            print(f"\n  原始 OCR 文本:")
            for item in parsed['raw_texts'][:10]:  # 只显示前10条
                print(f"    - {item['text']} (置信度: {item['confidence']:.2f})")

        return True

    except Exception as e:
        print(f"⚠ OCR 解析失败: {e}")
        print("  提示: 如果 PaddleOCR 未安装，这是正常的")
        print("  可选: pip install paddleocr paddlepaddle")
        return False


def test_tap(controller: ADBController):
    """测试点击功能（可选）"""
    print("\n" + "=" * 60)
    print("[可选测试] 点击功能")
    print("=" * 60)

    response = input("是否测试点击功能？这将在设备屏幕上点击一次 (y/n): ")

    if response.lower() == 'y':
        try:
            width, height = controller.get_screen_resolution()
            center_x, center_y = width // 2, height // 2

            print(f"将点击屏幕中心: ({center_x}, {center_y})")
            controller.tap(center_x, center_y, delay=1.0)
            print("✓ 点击成功")
            return True
        except Exception as e:
            print(f"✗ 点击失败: {e}")
            return False
    else:
        print("跳过点击测试")
        return True


def main():
    parser = argparse.ArgumentParser(description="测试 ADB 连接和游戏控制")

    parser.add_argument(
        "--device",
        type=str,
        default="101.43.137.83:5555",
        help="Android 设备 ID (通过 'adb devices' 查看)"
    )

    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="跳过 OCR 测试"
    )

    parser.add_argument(
        "--test-tap",
        action="store_true",
        help="测试点击功能"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ADB 连接测试工具 - 微信砍树游戏")
    print("=" * 60)
    print(f"\n设备 ID: {args.device}")

    # 测试 1: ADB 连接
    controller = test_adb_connection(args.device)
    if not controller:
        print("\n✗ 测试失败：无法连接设备")
        print("\n解决方案:")
        print("1. 确认设备已连接: adb devices")
        print("2. 确认设备 ID 正确")
        print("3. 如果是远程设备，确认端口转发正常")
        sys.exit(1)

    # 测试 2: 截图
    screenshot = test_screenshot(controller)
    if not screenshot:
        print("\n✗ 测试失败：无法截图")
        sys.exit(1)

    # 测试 3: 屏幕信息
    test_screen_info(controller)

    # 测试 4: OCR（可选）
    if not args.no_ocr:
        test_ocr(screenshot)

    # 可选: 测试点击
    if args.test_tap:
        test_tap(controller)

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print("\n✓ 所有基础测试通过！")
    print("\n下一步:")
    print("1. 确认游戏已打开并在主界面")
    print("2. 运行游戏 Agent:")
    print(f"   python wechat_tree_game_agent/play_game.py --device {args.device}")
    print("\n提示:")
    print("- 首次运行建议使用 --debug 参数，保存每步截图")
    print("- 如果没有训练好的模型，Agent 会使用规则策略")
    print("=" * 60)


if __name__ == "__main__":
    main()
