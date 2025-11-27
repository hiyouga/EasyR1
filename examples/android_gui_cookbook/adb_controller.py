"""
ADB 设备控制器

功能:
- 连接 Android 设备
- 执行点击、滑动、输入等操作
- 截图获取
- 设备状态检测
"""

import io
import subprocess
import time
from typing import Optional, Tuple

from PIL import Image


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
            result = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)

            if self.device_id not in result.stdout:
                raise ConnectionError(f"设备 {self.device_id} 未连接。请运行 'adb devices' 查看可用设备。")

            print(f"✓ 设备 {self.device_id} 已连接")

        except FileNotFoundError:
            raise RuntimeError("ADB 未安装或未添加到 PATH。请安装 Android SDK Platform-Tools。")
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
            result = subprocess.run(full_command.split(), capture_output=True, text=True, timeout=timeout)

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
                timeout=15,  # 增加超时时间，特别是远程设备或并发时
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
