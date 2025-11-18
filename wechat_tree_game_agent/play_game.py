"""
自动玩游戏 Agent

工作流程:
1. 连接到 Android 设备（通过 ADB）
2. 循环执行:
   - 截取当前游戏画面
   - 使用训练好的模型推理下一步动作
   - 解析并执行动作
   - 等待游戏响应
3. 记录妖力变化和决策历史
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wechat_tree_game_agent.android_env.adb_controller import ADBController
from wechat_tree_game_agent.android_env.game_state_parser import GameStateParser


class TreeGameAgent:
    """砍树游戏自动玩家"""

    def __init__(
        self,
        device_id: str,
        model_path: Optional[str] = None,
        use_ocr: bool = True,
        debug: bool = False
    ):
        """
        初始化 Agent

        Args:
            device_id: Android 设备 ID
            model_path: 训练好的模型路径（如果为 None，使用规则策略）
            use_ocr: 是否使用 OCR 识别妖力
            debug: 是否开启调试模式（保存每步截图）
        """
        self.device_id = device_id
        self.model_path = model_path
        self.use_ocr = use_ocr
        self.debug = debug

        # 初始化 ADB 控制器
        print("[初始化] 连接 Android 设备...")
        self.controller = ADBController(device_id=device_id)

        # 获取屏幕分辨率
        self.screen_width, self.screen_height = self.controller.get_screen_resolution()
        print(f"[初始化] 屏幕分辨率: {self.screen_width}x{self.screen_height}")

        # 初始化 OCR 解析器
        if self.use_ocr:
            print("[初始化] 加载 OCR 解析器...")
            self.parser = GameStateParser(use_gpu=False)
        else:
            self.parser = None

        # 初始化模型（如果提供）
        if self.model_path:
            print(f"[初始化] 加载模型: {self.model_path}")
            self.model = self._load_model(model_path)
        else:
            print("[初始化] 使用规则策略（无需模型）")
            self.model = None

        # 游戏状态
        self.step_count = 0
        self.demon_power_history = []
        self.action_history = []
        self.initial_power = None

        # 调试目录
        if self.debug:
            self.debug_dir = Path("debug_screenshots")
            self.debug_dir.mkdir(exist_ok=True)
            print(f"[调试] 截图保存目录: {self.debug_dir}")

    def _load_model(self, model_path: str):
        """
        加载训练好的模型

        TODO: 实现模型加载逻辑
        这里需要根据你的训练框架（vLLM, HuggingFace等）来实现
        """
        # 示例代码（需要根据实际情况修改）
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForVision2Seq.from_pretrained(model_path)

            print(f"✓ 模型加载成功: {model_path}")
            return {"model": model, "processor": processor}

        except Exception as e:
            print(f"⚠ 模型加载失败: {e}")
            print("使用规则策略代替")
            return None

    def calculate_axe_position(self) -> Tuple[int, int]:
        """
        计算斧子按钮位置（屏幕下方中央）

        根据屏幕分辨率动态调整坐标
        """
        # 斧子通常在屏幕底部 1/6 处，水平居中
        x = self.screen_width // 2
        y = int(self.screen_height * 5 / 6)

        return x, y

    def calculate_button_positions(self) -> Dict[str, Tuple[int, int]]:
        """
        计算装备界面按钮位置

        Returns:
            {
                "replace": (x, y),    # 替换按钮（绿色，右侧）
                "decompose": (x, y)   # 分解按钮（红色，左侧）
            }
        """
        # 装备弹窗通常在屏幕中央
        # 替换按钮在右侧，分解按钮在左侧
        center_y = int(self.screen_height * 0.65)  # 弹窗底部按钮区域

        return {
            "decompose": (int(self.screen_width * 0.35), center_y),  # 左侧 35%
            "replace": (int(self.screen_width * 0.65), center_y)     # 右侧 65%
        }

    def capture_and_parse(self) -> Dict:
        """
        截图并解析游戏状态

        Returns:
            {
                "image": PIL.Image,
                "state": "tree_cutting" | "equipment_selection",
                "demon_power": float,
                "equipment_stats": dict (如果是装备界面)
            }
        """
        # 截图
        screenshot = self.controller.capture_screenshot()

        # 保存调试截图
        if self.debug:
            debug_path = self.debug_dir / f"step_{self.step_count:03d}.png"
            screenshot.save(debug_path)

        # 解析截图（如果启用 OCR）
        parsed = {"image": screenshot, "state": "unknown", "demon_power": -1}

        if self.parser:
            parsed_result = self.parser.parse_screenshot(screenshot)
            parsed.update(parsed_result)

        return parsed

    def rule_based_decision(self, game_state: Dict) -> str:
        """
        基于规则的决策（不使用模型）

        规则:
        1. 如果是砍树界面 → click(斧子位置)
        2. 如果是装备界面:
           - 估算妖力变化 > 0 → replace()
           - 估算妖力变化 <= 0 → decompose()

        Args:
            game_state: capture_and_parse() 返回的结果

        Returns:
            动作字符串，例如: "click(180, 1000)", "replace()", "decompose()"
        """
        state = game_state.get("state", "unknown")

        if state == "tree_cutting":
            # 砍树界面 → 点击斧子
            x, y = self.calculate_axe_position()
            return f"click({x}, {y})"

        elif state == "equipment_selection":
            # 装备界面 → 判断妖力变化
            estimated_change = game_state.get("estimated_power_change", 0)

            if estimated_change > 0:
                return "replace()"
            else:
                return "decompose()"

        else:
            # 未知状态 → 默认点击砍树
            x, y = self.calculate_axe_position()
            return f"click({x}, {y})"

    def model_based_decision(self, game_state: Dict) -> str:
        """
        基于模型的决策

        TODO: 实现模型推理逻辑

        Args:
            game_state: capture_and_parse() 返回的结果

        Returns:
            动作字符串
        """
        if not self.model:
            return self.rule_based_decision(game_state)

        try:
            # 准备输入
            image = game_state["image"]
            prompt = "根据游戏截图，判断当前状态并输出最优动作，最大化妖力。"

            # 调用模型推理（示例代码，需要根据实际模型修改）
            processor = self.model["processor"]
            model = self.model["model"]

            inputs = processor(images=image, text=prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=50)
            response = processor.decode(outputs[0], skip_special_tokens=True)

            # 解析模型输出
            action = self._parse_model_output(response)
            return action

        except Exception as e:
            print(f"⚠ 模型推理失败: {e}")
            print("使用规则策略")
            return self.rule_based_decision(game_state)

    def _parse_model_output(self, response: str) -> str:
        """
        解析模型输出

        预期格式: <action>click(180, 1000)</action>

        Returns:
            动作字符串（去掉 <action> 标签）
        """
        pattern = r"<action>(.*?)</action>"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()
        else:
            print(f"⚠ 模型输出格式错误: {response}")
            # 默认点击砍树
            x, y = self.calculate_axe_position()
            return f"click({x}, {y})"

    def execute_action(self, action: str) -> bool:
        """
        执行动作

        Args:
            action: 动作字符串，例如:
                - "click(180, 1000)"
                - "replace()"
                - "decompose()"

        Returns:
            是否执行成功
        """
        # 解析 click 动作
        click_pattern = r"click\((\d+),\s*(\d+)\)"
        click_match = re.match(click_pattern, action)

        if click_match:
            x = int(click_match.group(1))
            y = int(click_match.group(2))
            return self.controller.tap(x, y, delay=1.5)

        # 解析 replace 动作
        elif action in ["replace()", "equip()"]:
            positions = self.calculate_button_positions()
            x, y = positions["replace"]
            print(f"[执行] 点击替换按钮: ({x}, {y})")
            return self.controller.tap(x, y, delay=2.0)

        # 解析 decompose 动作
        elif action in ["decompose()", "skip()"]:
            positions = self.calculate_button_positions()
            x, y = positions["decompose"]
            print(f"[执行] 点击分解按钮: ({x}, {y})")
            return self.controller.tap(x, y, delay=2.0)

        else:
            print(f"⚠ 未知动作: {action}")
            return False

    def run_episode(self, max_steps: int = 50) -> Dict:
        """
        运行一个 episode（一局游戏）

        Args:
            max_steps: 最大步数

        Returns:
            统计信息
        """
        print("\n" + "=" * 60)
        print(f"开始游戏 Episode (最大步数: {max_steps})")
        print("=" * 60)

        self.step_count = 0
        self.demon_power_history = []
        self.action_history = []

        for step in range(max_steps):
            self.step_count = step + 1
            print(f"\n--- Step {self.step_count} ---")

            # 1. 截图并解析
            print("[1/4] 截图并解析游戏状态...")
            game_state = self.capture_and_parse()

            state = game_state.get("state", "unknown")
            demon_power = game_state.get("demon_power", -1)

            print(f"  状态: {state}")
            if demon_power > 0:
                print(f"  妖力: {demon_power}")
                self.demon_power_history.append(demon_power)

                if self.initial_power is None:
                    self.initial_power = demon_power

            # 2. 决策
            print("[2/4] 决策下一步动作...")
            if self.model:
                action = self.model_based_decision(game_state)
            else:
                action = self.rule_based_decision(game_state)

            print(f"  动作: {action}")
            self.action_history.append({
                "step": self.step_count,
                "state": state,
                "action": action,
                "demon_power": demon_power
            })

            # 3. 执行动作
            print("[3/4] 执行动作...")
            success = self.execute_action(action)

            if not success:
                print("⚠ 动作执行失败，跳过此步")
                continue

            # 4. 等待游戏响应
            print("[4/4] 等待游戏响应...")
            time.sleep(2.0)  # 等待动画和界面更新

            # 检查是否完成
            if self.step_count >= 10 and state == "equipment_selection":
                print("\n已完成 10 次砍树，Episode 结束")
                break

        # 统计信息
        stats = self._calculate_stats()
        self._print_summary(stats)

        return stats

    def _calculate_stats(self) -> Dict:
        """计算统计信息"""
        stats = {
            "total_steps": self.step_count,
            "initial_power": self.initial_power or 0,
            "final_power": self.demon_power_history[-1] if self.demon_power_history else 0,
            "power_gain": 0,
            "action_counts": {}
        }

        # 计算妖力增长
        if self.initial_power and self.demon_power_history:
            stats["power_gain"] = stats["final_power"] - stats["initial_power"]

        # 统计动作类型
        for record in self.action_history:
            action_type = record["action"].split("(")[0]
            stats["action_counts"][action_type] = stats["action_counts"].get(action_type, 0) + 1

        return stats

    def _print_summary(self, stats: Dict):
        """打印总结"""
        print("\n" + "=" * 60)
        print("Episode 总结")
        print("=" * 60)

        print(f"\n总步数: {stats['total_steps']}")
        print(f"初始妖力: {stats['initial_power']:.0f}")
        print(f"最终妖力: {stats['final_power']:.0f}")
        print(f"妖力增长: {stats['power_gain']:+.0f}")

        print("\n动作统计:")
        for action_type, count in sorted(stats["action_counts"].items()):
            print(f"  {action_type:15s}: {count:3d} 次")

        print("=" * 60)

        # 保存历史记录
        if self.debug:
            history_file = self.debug_dir / "action_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.action_history, f, ensure_ascii=False, indent=2)
            print(f"\n✓ 动作历史已保存: {history_file}")


def main():
    parser = argparse.ArgumentParser(description="自动玩砍树游戏")

    parser.add_argument(
        "--device",
        type=str,
        default="101.43.137.83:5555",
        help="Android 设备 ID (通过 'adb devices' 查看)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="训练好的模型路径（如不提供，使用规则策略）"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="最大步数"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="运行几局游戏"
    )

    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="不使用 OCR（仅依赖模型）"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="开启调试模式（保存截图和历史）"
    )

    args = parser.parse_args()

    # 创建 Agent
    agent = TreeGameAgent(
        device_id=args.device,
        model_path=args.model,
        use_ocr=not args.no_ocr,
        debug=args.debug
    )

    # 运行游戏
    all_stats = []
    for episode in range(args.episodes):
        if args.episodes > 1:
            print(f"\n\n{'=' * 60}")
            print(f"Episode {episode + 1} / {args.episodes}")
            print('=' * 60)

        stats = agent.run_episode(max_steps=args.max_steps)
        all_stats.append(stats)

        # 多局游戏之间等待
        if episode < args.episodes - 1:
            print("\n等待 5 秒后开始下一局...")
            time.sleep(5)

    # 打印总体统计
    if args.episodes > 1:
        print("\n\n" + "=" * 60)
        print(f"总体统计 ({args.episodes} 局)")
        print("=" * 60)

        avg_power_gain = sum(s["power_gain"] for s in all_stats) / len(all_stats)
        print(f"\n平均妖力增长: {avg_power_gain:+.0f}")

        print("\n各局妖力增长:")
        for i, s in enumerate(all_stats, 1):
            print(f"  Episode {i}: {s['power_gain']:+.0f}")


if __name__ == "__main__":
    main()
