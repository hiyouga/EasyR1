"""
ADB 设备控制器

功能:
- 连接 Android 设备
- 执行点击、滑动、输入等操作
- 截图获取
- 设备状态检测
"""

import subprocess
import time
from typing import Tuple, Optional
from PIL import Image
import io


class ADBController:
    """Android Debug Bridge 控制器"""

    def __init__(self, device_id: str = "emulator-5554"):
        """
        初始化 ADB 控制器

        Args:
            device_id: 设备 ID (通过 `adb devices` 查看)
        """
        self.device_id = device_id
        self._check_connection()

    def _check_connection(self):
        """检查设备连接"""
        try:
            result = subprocess.run(
                ["adb", "devices"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if self.device_id not in result.stdout:
                raise ConnectionError(
                    f"设备 {self.device_id} 未连接。"
                    f"请运行 'adb devices' 查看可用设备。"
                )

            print(f"✓ 设备 {self.device_id} 已连接")

        except FileNotFoundError:
            raise RuntimeError(
                "ADB 未安装或未添加到 PATH。"
                "请安装 Android SDK Platform-Tools。"
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError("ADB 连接超时，请检查设备状态。")

    def execute_command(self, command: str, timeout: int = 10) -> str:
        """
        执行 ADB 命令

        Args:
            command: ADB 命令 (不包含 'adb -s device_id' 前缀)
            timeout: 超时时间 (秒)

        Returns:
            命令输出结果
        """
        full_command = f"adb -s {self.device_id} {command}"

        try:
            result = subprocess.run(
                full_command.split(),
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"命令执行失败: {result.stderr}")

            return result.stdout

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"命令执行超时: {command}")

    def capture_screenshot(self, save_path: Optional[str] = None) -> Image.Image:
        """
        截取屏幕截图

        Args:
            save_path: 保存路径 (可选)

        Returns:
            PIL Image 对象
        """
        try:
            # 使用 screencap 命令
            cmd = f"adb -s {self.device_id} exec-out screencap -p"
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                timeout=15  # 增加超时时间，特别是远程设备或并发时
            )

            if result.returncode != 0:
                raise RuntimeError("截图失败")

            # 将字节流转换为 PIL Image
            image = Image.open(io.BytesIO(result.stdout))

            if save_path:
                image.save(save_path)
                print(f"✓ 截图已保存: {save_path}")

            return image

        except Exception as e:
            raise RuntimeError(f"截图失败: {e}")

    def tap(self, x: int, y: int, delay: float = 0.5) -> bool:
        """
        点击屏幕坐标

        Args:
            x: X 坐标
            y: Y 坐标
            delay: 点击后等待时间 (秒)

        Returns:
            是否成功
        """
        try:
            self.execute_command(f"shell input tap {x} {y}")
            time.sleep(delay)
            print(f"✓ 点击坐标: ({x}, {y})")
            return True

        except Exception as e:
            print(f"✗ 点击失败: {e}")
            return False

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> bool:
        """
        滑动屏幕

        Args:
            x1, y1: 起始坐标
            x2, y2: 结束坐标
            duration: 滑动持续时间 (毫秒)

        Returns:
            是否成功
        """
        try:
            self.execute_command(f"shell input swipe {x1} {y1} {x2} {y2} {duration}")
            print(f"✓ 滑动: ({x1}, {y1}) → ({x2}, {y2})")
            return True

        except Exception as e:
            print(f"✗ 滑动失败: {e}")
            return False

    def input_text(self, text: str) -> bool:
        """
        输入文本 (需要输入框已聚焦)

        Args:
            text: 要输入的文本

        Returns:
            是否成功
        """
        try:
            # 转义空格
            escaped_text = text.replace(" ", "%s")
            self.execute_command(f"shell input text '{escaped_text}'")
            print(f"✓ 输入文本: {text}")
            return True

        except Exception as e:
            print(f"✗ 输入失败: {e}")
            return False

    def press_back(self) -> bool:
        """按返回键"""
        try:
            self.execute_command("shell input keyevent KEYCODE_BACK")
            print("✓ 按返回键")
            return True
        except Exception as e:
            print(f"✗ 返回键失败: {e}")
            return False

    def press_home(self) -> bool:
        """按主屏幕键"""
        try:
            self.execute_command("shell input keyevent KEYCODE_HOME")
            print("✓ 按主屏幕键")
            return True
        except Exception as e:
            print(f"✗ 主屏幕键失败: {e}")
            return False

    def get_screen_resolution(self) -> Tuple[int, int]:
        """
        获取屏幕分辨率

        Returns:
            (width, height)
        """
        try:
            output = self.execute_command("shell wm size")
            # 输出格式: Physical size: 1080x2400
            size_str = output.split(":")[-1].strip()
            width, height = map(int, size_str.split("x"))
            return width, height

        except Exception as e:
            print(f"⚠ 无法获取分辨率，使用默认值 (1080, 2400): {e}")
            return 1080, 2400

    def start_app(self, package_name: str, activity: Optional[str] = None) -> bool:
        """
        启动应用

        Args:
            package_name: 包名 (例如: com.tencent.mm)
            activity: Activity 名称 (可选)

        Returns:
            是否成功
        """
        try:
            if activity:
                cmd = f"shell am start -n {package_name}/{activity}"
            else:
                cmd = f"shell monkey -p {package_name} -c android.intent.category.LAUNCHER 1"

            self.execute_command(cmd)
            time.sleep(2)  # 等待应用启动
            print(f"✓ 启动应用: {package_name}")
            return True

        except Exception as e:
            print(f"✗ 启动失败: {e}")
            return False

    def stop_app(self, package_name: str) -> bool:
        """
        停止应用

        Args:
            package_name: 包名

        Returns:
            是否成功
        """
        try:
            self.execute_command(f"shell am force-stop {package_name}")
            print(f"✓ 停止应用: {package_name}")
            return True

        except Exception as e:
            print(f"✗ 停止失败: {e}")
            return False


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ADB 控制器测试")
    print("=" * 60)

    # 从命令行参数获取设备 ID，或使用默认值
    device_id = sys.argv[1] if len(sys.argv) > 1 else "emulator-5554"

    try:
        # 初始化控制器
        controller = ADBController(device_id=device_id)

        # 测试 1: 获取屏幕分辨率
        print("\n[测试 1] 获取屏幕分辨率")
        width, height = controller.get_screen_resolution()
        print(f"屏幕分辨率: {width}x{height}")

        # 测试 2: 截图
        print("\n[测试 2] 截取截图")
        screenshot = controller.capture_screenshot(save_path="test_screenshot.png")
        print(f"截图尺寸: {screenshot.size}")

        # 测试 3: 点击屏幕中心
        print("\n[测试 3] 点击屏幕中心")
        center_x, center_y = width // 2, height // 2
        controller.tap(center_x, center_y)

        print("\n" + "=" * 60)
        print("所有测试通过！ADB 控制器工作正常。")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        sys.exit(1)
