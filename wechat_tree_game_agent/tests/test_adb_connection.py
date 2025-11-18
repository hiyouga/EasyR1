"""
测试 ADB 连接

运行: python wechat_tree_game_agent/tests/test_adb_connection.py [device_id]
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from wechat_tree_game_agent.android_env import ADBController


def test_adb_connection(device_id: str = "emulator-5554"):
    """测试 ADB 连接和基本功能"""

    print("=" * 60)
    print("ADB 连接测试")
    print("=" * 60)

    try:
        # 1. 初始化控制器
        print("\n[测试 1] 初始化 ADB 控制器")
        controller = ADBController(device_id=device_id)
        print(f"✓ 设备连接成功: {device_id}")

        # 2. 获取屏幕分辨率
        print("\n[测试 2] 获取屏幕分辨率")
        width, height = controller.get_screen_resolution()
        print(f"✓ 屏幕分辨率: {width}x{height}")

        # 3. 截图
        print("\n[测试 3] 截取屏幕截图")
        screenshot = controller.capture_screenshot(
            save_path="wechat_tree_game_agent/tests/test_screenshot.png"
        )
        print(f"✓ 截图成功，尺寸: {screenshot.size}")

        # 4. 点击测试（点击屏幕中心）
        print("\n[测试 4] 点击屏幕中心")
        center_x, center_y = width // 2, height // 2
        controller.tap(center_x, center_y)
        print(f"✓ 点击成功: ({center_x}, {center_y})")

        print("\n" + "=" * 60)
        print("所有测试通过！✅")
        print("=" * 60)

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"测试失败！❌")
        print(f"错误: {e}")
        print("=" * 60)

        print("\n故障排查:")
        print("1. 检查设备是否连接: adb devices")
        print("2. 检查设备 ID 是否正确")
        print("3. 检查 USB 调试是否开启")
        print("4. 尝试重启 ADB: adb kill-server && adb start-server")

        return False


if __name__ == "__main__":
    # 从命令行参数获取设备 ID
    device_id = sys.argv[1] if len(sys.argv) > 1 else "emulator-5554"

    success = test_adb_connection(device_id)
    sys.exit(0 if success else 1)
