"""
自动玩数字选择游戏的 Agent

功能:
1. 支持 Ollama 和 vLLM 两种模型服务
2. 支持多个 Android 设备并发/顺序执行
3. 自动识别游戏状态、指示灯、数字
4. 自动做出决策并执行操作
5. 记录游戏结果和截图
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from adb_controller import ADBController
from PIL import Image
from vlm_client import VLMClient


class GameAgent:
    """数字选择游戏 Agent"""

    def __init__(
        self, device_id: str, vlm_client: VLMClient, screenshot_dir: str = "game_screenshots", debug: bool = False
    ):
        self.device_id = device_id
        self.vlm_client = vlm_client
        self.debug = debug

        # 创建截图目录
        self.screenshot_dir = Path(screenshot_dir) / device_id.replace(":", "_")
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 ADB 控制器
        print(f"[{device_id}] 连接 Android 设备...")
        self.controller = ADBController(device_id=device_id)

        # 获取屏幕分辨率
        self.screen_width, self.screen_height = self.controller.get_screen_resolution()
        print(f"[{device_id}] 屏幕分辨率: {self.screen_width}x{self.screen_height}")

        # 游戏状态
        self.current_round = 0
        self.game_results = []

    def calculate_card_positions(self) -> List[Tuple[int, int]]:
        """计算 3 个卡片的点击位置"""
        # 根据实际测试，720x1280屏幕下的精确坐标
        y = 905  # 实测坐标

        positions = [
            (135, y),  # 左边数字
            (360, y),  # 中间数字
            (585, y),  # 右边数字
        ]
        return positions

    def calculate_next_button_position(self) -> Tuple[int, int]:
        """计算"下一轮"按钮的位置"""
        # 下一轮按钮坐标（实测坐标）
        x = 360
        y = 1070
        return (x, y)

    def recognize_score(self, screenshot: Image.Image) -> Optional[int]:
        """
        使用VLM识别截图中的游戏分数

        Args:
            screenshot: 游戏截图

        Returns:
            识别的分数值，如果识别失败返回 None
        """
        prompt = """请仔细观察这张游戏截图，识别粉色/红色渐变卡片中间的大数字（当前分数）是多少？

只需要回答数字即可，不需要其他说明。"""

        try:
            response = self.vlm_client.query(screenshot, prompt)

            if self.debug:
                print(f"[{self.device_id}] 分数识别VLM输出: {response}")

            # 从响应中提取数字
            # 尝试多种模式匹配
            patterns = [
                r"(\d+)",  # 任何数字
                r"分数[是为:：]\s*(\d+)",
                r"数字[是为:：]\s*(\d+)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, response)
                if matches:
                    score = int(matches[0])
                    print(f"[{self.device_id}] 识别到的分数: {score}")
                    return score

            print(f"⚠ [{self.device_id}] 无法从VLM输出中提取分数")
            return None

        except Exception as e:
            print(f"⚠ [{self.device_id}] 分数识别失败: {e}")
            return None

    def capture_screenshot(self, round_num: int) -> Image.Image:
        """截取屏幕并保存"""
        screenshot = self.controller.capture_screenshot()

        # 保存截图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"round_{round_num:02d}_{timestamp}.png"
        filepath = self.screenshot_dir / filename
        screenshot.save(filepath)

        if self.debug:
            print(f"[{self.device_id}] 截图已保存: {filepath}")

        return screenshot

    def check_click_success(self, screenshot: Image.Image) -> bool:
        """
        检查点击后是否成功（卡片颜色是否改变）

        Args:
            screenshot: 点击后的截图

        Returns:
            True表示点击成功，False表示未成功
        """
        prompt = """请观察这张游戏截图，判断三个数字卡片中是否有卡片的颜色发生了变化（不再是原来的紫色/蓝色渐变，而是变成了红色、绿色或黄色）？

如果有卡片颜色改变了，回答"是"；如果所有卡片颜色都没变，回答"否"。"""

        try:
            response = self.vlm_client.query(screenshot, prompt)

            if self.debug:
                print(f"[{self.device_id}] 点击成功检查VLM输出: {response}")

            # 判断是否点击成功
            if "是" in response or "yes" in response.lower() or "成功" in response or "改变" in response:
                return True
            else:
                return False

        except Exception as e:
            print(f"⚠ [{self.device_id}] 检查点击成功失败: {e}")
            return False

    def parse_vlm_response(self, response: str) -> Optional[int]:
        """
        解析 VLM 输出，提取选择的索引

        Returns:
            选择的索引 (0-2)，如果解析失败返回 None
        """
        # 策略 1: 提取 <action>select(N)</action>
        pattern = r"<action>select\((\d+)\)</action>"
        matches = re.findall(pattern, response, re.IGNORECASE)

        if matches:
            index = int(matches[0])
            if 0 <= index <= 2:
                return index

        # 策略 2: 提取 "选择的索引: N" 或类似文本
        index_patterns = [
            r"选择的?索引[:：]\s*(\d+)",
            r"索引[:：]\s*(\d+)",
            r"select\((\d+)\)",
            r"选择\s*(\d+)",
            r"第\s*(\d+)\s*个",
            r"answer[:：]\s*(\d+)",
            r"选项\s*([abc])",
        ]

        for pattern in index_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                match_str = matches[0]
                # 处理选项a/b/c
                if match_str.lower() in ["a", "b", "c"]:
                    index = ord(match_str.lower()) - ord("a")
                else:
                    index = int(match_str)

                if 0 <= index <= 2:
                    return index

        print(f"⚠ [{self.device_id}] VLM 输出解析失败，无法提取索引")
        return None

    def make_decision(self, screenshot: Image.Image) -> Optional[int]:
        """
        基于截图做出决策

        Returns:
            选择的索引 (0-2)，如果失败返回 None
        """
        prompt = """这是一个数字选择游戏的截图。

游戏规则:
- 屏幕上方有3个指示灯（圆形）
- 绿色灯亮：选择最大的数字 (+10分)
- 红色灯亮：选择最小的数字 (+10分)
- 黄色灯亮：选择中间的数字 (+10分)
- 屏幕中央有3个数字卡片（从左到右排列），分别标记为"选项 a"、"选项 b"、"选项 c"

任务:
1. 识别哪个指示灯是亮的（绿色/红色/黄色）
2. 识别3个数字卡片的数字（选项a、选项b、选项c）
3. 根据亮灯的规则选择正确的数字

请按照以下格式回答:
1. 亮灯颜色: [绿色/红色/黄色]
2. 识别的数字: [选项a的数字, 选项b的数字, 选项c的数字]
3. 应选择: [最大/最小/中间]
4. 选择的索引: N (0=选项a/左边, 1=选项b/中间, 2=选项c/右边)

最后用以下格式输出你的选择:
<action>select(N)</action>

其中 N 是 0、1 或 2。
"""

        print(f"[{self.device_id}] VLM 推理中...")
        response = self.vlm_client.query(screenshot, prompt)

        if self.debug:
            print("\n--- VLM 输出 ---")
            print(response)
            print("--- 输出结束 ---\n")

        # 解析响应
        selected_index = self.parse_vlm_response(response)

        if selected_index is not None:
            print(f"[{self.device_id}] VLM 决策: 选择索引 {selected_index}")

        return selected_index

    def play_one_round(self, round_num: int) -> bool:
        """
        玩一轮游戏

        Returns:
            是否成功完成该轮
        """
        print(f"\n[{self.device_id}] ========== Round {round_num}/10 ==========")

        # 1. 截图
        print(f"[{self.device_id}] [1/5] 截图...")
        screenshot = self.capture_screenshot(round_num)

        # 2. VLM 决策
        print(f"[{self.device_id}] [2/5] VLM 决策...")
        selected_index = self.make_decision(screenshot)

        if selected_index is None:
            print(f"⚠ [{self.device_id}] 决策失败，跳过此轮")
            return False

        # 3. 执行动作：点击卡片，并验证是否成功
        print(f"[{self.device_id}] [3/5] 点击卡片...")
        positions = self.calculate_card_positions()
        x, y = positions[selected_index]

        max_retry = 3
        click_success = False

        for retry in range(max_retry):
            # 点击卡片
            success = self.controller.tap(x, y, delay=1.5)
            if not success:
                print(f"⚠ [{self.device_id}] 点击卡片失败")
                return False

            print(f"[{self.device_id}] 已点击位置: ({x}, {y})")

            # 等待一下，让界面反应
            time.sleep(1.0)

            # 截图检查点击是否成功，并保存截图
            check_screenshot = self.controller.capture_screenshot()

            # 保存点击后的截图
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"round_{round_num:02d}_after_click_{timestamp}.png"
            filepath = self.screenshot_dir / filename
            check_screenshot.save(filepath)
            if self.debug:
                print(f"[{self.device_id}] 点击后截图已保存: {filepath}")

            click_success = self.check_click_success(check_screenshot)

            if click_success:
                print(f"[{self.device_id}] ✓ 点击成功，卡片颜色已改变")
                break
            else:
                print(f"⚠ [{self.device_id}] 点击未成功（第{retry + 1}次尝试），重试...")
                time.sleep(0.5)

        if not click_success:
            print(f"⚠ [{self.device_id}] 多次点击均未成功，跳过此轮")
            return False

        # 4. 等待反馈显示
        print(f"[{self.device_id}] [4/5] 等待反馈...")
        time.sleep(1.5)

        # 5. 点击"下一轮"按钮
        print(f"[{self.device_id}] [5/5] 点击下一轮...")
        next_x, next_y = self.calculate_next_button_position()
        success = self.controller.tap(next_x, next_y, delay=1.5)

        if not success:
            print(f"⚠ [{self.device_id}] 点击下一轮按钮失败")
            return False

        print(f"[{self.device_id}] 已点击下一轮按钮: ({next_x}, {next_y})")

        return True

    def capture_final_score(self) -> Image.Image:
        """捕获游戏结束时的最终得分截图"""
        print(f"[{self.device_id}] 截取最终得分...")
        time.sleep(2.0)  # 等待游戏结束动画

        screenshot = self.controller.capture_screenshot()

        # 保存最终截图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"final_score_{timestamp}.png"
        filepath = self.screenshot_dir / filename
        screenshot.save(filepath)

        print(f"[{self.device_id}] 最终得分截图已保存: {filepath}")

        return screenshot

    def refresh_browser(self):
        """刷新浏览器，准备下一局游戏"""
        print(f"[{self.device_id}] 刷新浏览器...")

        # 方法1: 先按返回键关闭可能弹出的键盘
        os.system(f"adb -s {self.device_id} shell input keyevent 4")  # KEYCODE_BACK
        time.sleep(0.5)

        # 方法2: 点击右上角的刷新按钮（三个点按钮旁边）
        # 根据截图，刷新按钮在地址栏右侧，大约在 x=540 的位置
        # 对于 720x1280 的屏幕，刷新按钮大约在 (540, 140)
        refresh_button_x = 380  # 地址栏右侧的前进按钮位置
        refresh_button_y = 130

        self.controller.tap(refresh_button_x, refresh_button_y, delay=1.0)

        time.sleep(3.0)  # 等待页面加载
        print(f"[{self.device_id}] 浏览器已刷新")

    def save_results(self, final_screenshot: Image.Image, final_score: Optional[int] = None):
        """保存游戏结果到 JSON 文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result = {
            "device_id": self.device_id,
            "timestamp": timestamp,
            "total_rounds": 10,
            "final_score": final_score,
            "screenshot_dir": str(self.screenshot_dir),
            "final_screenshot": str(self.screenshot_dir / f"final_score_{timestamp}.png"),
            "model_type": self.vlm_client.model_type,
            "model_name": self.vlm_client.model_name,
        }

        # 保存结果
        result_file = self.screenshot_dir / f"result_{timestamp}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[{self.device_id}] 游戏结果已保存: {result_file}")
        if final_score is not None:
            print(f"[{self.device_id}] 最终得分: {final_score}")

        self.game_results.append(result)

    def run_game(self) -> Dict:
        """
        运行完整的游戏（10轮）

        Returns:
            游戏结果字典
        """
        print(f"\n{'=' * 60}")
        print(f"[{self.device_id}] 开始游戏")
        print(f"{'=' * 60}\n")

        # 玩10轮，确保每轮都成功
        completed_rounds = 0
        max_total_attempts = 20  # 最多尝试20次，避免无限循环
        total_attempts = 0

        while completed_rounds < 10 and total_attempts < max_total_attempts:
            total_attempts += 1

            print(
                f"\n[{self.device_id}] ========== 尝试第 {completed_rounds + 1} 轮 (总尝试次数: {total_attempts}) =========="
            )

            success = self.play_one_round(completed_rounds + 1)

            if success:
                completed_rounds += 1
                print(f"✓ [{self.device_id}] 第 {completed_rounds} 轮完成！")
            else:
                print(f"⚠ [{self.device_id}] 本轮执行失败，将重新尝试...")
                time.sleep(2.0)  # 失败后等待longer一点
                continue

            # 轮次间等待
            if completed_rounds < 10:
                time.sleep(1.5)

        if completed_rounds < 10:
            print(f"\n⚠ [{self.device_id}] 警告: 未能完成10轮游戏，只完成了 {completed_rounds} 轮")

        # 游戏结束，截取最终得分
        final_screenshot = self.capture_final_score()

        # 使用VLM识别最终分数
        print(f"[{self.device_id}] 识别最终分数...")
        final_score = self.recognize_score(final_screenshot)

        if final_score is not None:
            print(f"\n{'=' * 60}")
            print(f"[{self.device_id}] 🎉 最终得分: {final_score}")
            print(f"{'=' * 60}\n")
        else:
            print(f"⚠ [{self.device_id}] 未能识别最终分数")

        # 保存结果
        self.save_results(final_screenshot, final_score)

        print(f"\n{'=' * 60}")
        print(f"[{self.device_id}] 游戏完成！")
        print(f"[{self.device_id}] 完成轮数: {completed_rounds}/10")
        print(f"[{self.device_id}] 请手动刷新浏览器以开始下一局游戏")
        print(f"{'=' * 60}\n")

        return self.game_results[-1] if self.game_results else {}


def main():
    parser = argparse.ArgumentParser(description="自动玩数字选择游戏的 Agent")

    # 模型配置
    parser.add_argument(
        "--model-type", type=str, choices=["ollama", "vllm"], default="ollama", help="模型服务类型: ollama 或 vllm"
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:11434",
        help="模型 API 地址 (Ollama: http://localhost:11434, vLLM: http://localhost:8000)",
    )

    parser.add_argument("--model-name", type=str, default="qwen2.5vl:3b", help="模型名称")

    # Android 设备配置
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["101.43.137.83:5555"],
        help="Android 设备地址列表，如: 101.43.137.83:5555 192.168.1.100:5555",
    )

    # 其他配置
    parser.add_argument("--screenshot-dir", type=str, default="game_screenshots", help="截图保存目录")

    parser.add_argument("--episodes", type=int, default=1, help="每个设备运行几局游戏")

    parser.add_argument("--parallel", action="store_true", help="并发处理多个设备（默认顺序处理）")

    parser.add_argument("--debug", action="store_true", help="开启调试模式")

    args = parser.parse_args()

    # 初始化 VLM 客户端
    print("=" * 60)
    print("初始化 VLM 客户端")
    print("=" * 60)
    print(f"模型类型: {args.model_type}")
    print(f"API 地址: {args.api_url}")
    print(f"模型名称: {args.model_name}")
    print()

    vlm_client = VLMClient(model_type=args.model_type, api_url=args.api_url, model_name=args.model_name)

    # 处理多个设备
    all_results = []

    if args.parallel:
        # TODO: 实现并发处理（使用 threading 或 multiprocessing）
        print("⚠ 并发模式暂未实现，使用顺序模式")
        args.parallel = False

    # 顺序处理
    for device_id in args.devices:
        print(f"\n{'=' * 60}")
        print(f"处理设备: {device_id}")
        print(f"{'=' * 60}\n")

        # 创建 Agent
        agent = GameAgent(
            device_id=device_id, vlm_client=vlm_client, screenshot_dir=args.screenshot_dir, debug=args.debug
        )

        # 运行多局游戏
        for episode in range(1, args.episodes + 1):
            if args.episodes > 1:
                print(f"\n--- Episode {episode}/{args.episodes} ---\n")

            result = agent.run_game()
            all_results.append(result)

            # 局间等待
            if episode < args.episodes:
                print("\n等待 5 秒后开始下一局...\n")
                time.sleep(5)

    # 打印总结
    print("\n" + "=" * 60)
    print("所有游戏完成！")
    print("=" * 60)
    print(f"总共完成: {len(all_results)} 局游戏")
    print(f"涉及设备: {len(args.devices)} 个")
    print("\n结果文件已保存到各自的截图目录中")


if __name__ == "__main__":
    main()
