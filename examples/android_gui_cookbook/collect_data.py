"""
数据收集脚本 - 用于离线训练数据集构建

功能:
1. 支持多设备并发收集游戏截图
2. 截图命名格式规范，方便后续批量标注
3. 只收集截图，不调用VLM（节省时间和资源）
4. 自动重试失败的轮次
5. 记录每局游戏的元数据
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from adb_controller import ADBController


class DataCollector:
    """游戏数据收集器"""

    def __init__(self, device_id: str, output_dir: str = "game_data_raw", debug: bool = False):
        self.device_id = device_id
        self.debug = debug

        # 创建输出目录（使用安全的文件名）
        safe_device_id = device_id.replace(":", "_").replace(".", "_")
        self.output_dir = Path(output_dir) / safe_device_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 检查已有的episode，自动续集
        self.start_episode_id = self._find_next_episode_id()
        if self.start_episode_id > 1:
            print(f"[{device_id}] 检测到已有数据，从 Episode {self.start_episode_id} 继续收集")

        # 初始化 ADB 控制器
        print(f"[{device_id}] 连接 Android 设备...")
        self.controller = ADBController(device_id=device_id)

        # 获取屏幕分辨率
        self.screen_width, self.screen_height = self.controller.get_screen_resolution()
        print(f"[{device_id}] 屏幕分辨率: {self.screen_width}x{self.screen_height}")

        # 游戏元数据
        self.episodes = []

    def _find_next_episode_id(self) -> int:
        """
        查找下一个可用的episode_id（避免覆盖已有数据）

        Returns:
            下一个episode_id（从1开始）
        """
        existing_episodes = list(self.output_dir.glob("episode_*_metadata.json"))

        if not existing_episodes:
            return 1

        # 提取所有已有的episode_id
        episode_ids = []
        for metadata_file in existing_episodes:
            # 文件名格式: episode_001_metadata.json
            match = re.match(r"episode_(\d+)_metadata\.json", metadata_file.name)
            if match:
                episode_ids.append(int(match.group(1)))

        if episode_ids:
            return max(episode_ids) + 1
        else:
            return 1

    def calculate_card_positions(self) -> List[Tuple[int, int]]:
        """计算 3 个选项按钮的点击位置"""
        # 根据截图分析，选项按钮在屏幕约65-68%高度处
        y = 860  # 调整后的坐标（720x1280屏幕，避免触发键盘）
        positions = [
            (135, y),  # 左边（选项a）
            (360, y),  # 中间（选项b）
            (585, y),  # 右边（选项c）
        ]
        return positions

    def calculate_next_button_position(self) -> Tuple[int, int]:
        """计算"下一轮"按钮的位置"""
        # 下一轮按钮在屏幕约80-82%高度处
        return (360, 1040)

    def random_choice(self) -> int:
        """随机选择一个索引（0, 1, 2）"""
        import random

        return random.choice([0, 1, 2])

    def capture_and_save(self, episode_id: int, round_num: int, suffix: str = "") -> Optional[str]:
        """
        截图并保存，使用标准化的文件名

        文件名格式: episode_{ep}_round_{rd}_{suffix}.png
        例如: episode_001_round_03_question.png, episode_001_round_03_result.png

        Args:
            episode_id: 局数
            round_num: 轮数
            suffix: 文件名后缀，如 "question" 或 "result"

        Returns:
            保存的文件路径（相对路径），失败返回 None
        """
        try:
            screenshot = self.controller.capture_screenshot()

            # 标准化文件名
            if suffix:
                filename = f"episode_{episode_id:03d}_round_{round_num:02d}_{suffix}.png"
            else:
                filename = f"episode_{episode_id:03d}_round_{round_num:02d}.png"
            filepath = self.output_dir / filename

            screenshot.save(filepath)

            if self.debug:
                print(f"[{self.device_id}] 截图已保存: {filepath}")

            return str(filepath.relative_to(self.output_dir.parent))

        except Exception as e:
            print(f"⚠ [{self.device_id}] 截图失败: {e}")
            return None

    def check_card_color_changed(self) -> bool:
        """
        简单检查：等待一下后重新截图，看是否有颜色变化
        这里用简化的方法：如果点击后界面没报错，就认为成功
        """
        time.sleep(0.8)
        return True  # 简化处理，假设点击总是成功

    def play_one_round(self, episode_id: int, round_num: int) -> Optional[Dict]:
        """
        玩一轮游戏并收集数据

        收集两张截图：
        1. question.png - 操作前的状态（灯光+数字选项）
        2. result.png - 操作后的反馈（显示正确答案）

        Returns:
            该轮的元数据字典，失败返回 None
        """
        print(f"[{self.device_id}] Round {round_num}/10")

        # 短暂延迟，避免并发时ADB冲突
        time.sleep(0.3)

        # 1. 截图1：question（操作前状态）
        question_screenshot = self.capture_and_save(episode_id, round_num, "question")
        if question_screenshot is None:
            return None

        # 2. 随机选择一个卡片点击
        selected_index = self.random_choice()

        if self.debug:
            print(f"[{self.device_id}] 随机选择索引: {selected_index}")

        # 3. 点击卡片
        positions = self.calculate_card_positions()
        x, y = positions[selected_index]

        max_retry = 3
        click_success = False

        for retry in range(max_retry):
            success = self.controller.tap(x, y, delay=1.0)
            if not success:
                if retry < max_retry - 1:
                    print(f"⚠ [{self.device_id}] 点击失败，重试 {retry + 1}/{max_retry}")
                    time.sleep(0.5)
                    continue
                else:
                    print(f"⚠ [{self.device_id}] 点击失败，跳过此轮")
                    return None

            # 检查点击是否成功
            if self.check_card_color_changed():
                click_success = True
                break
            else:
                if retry < max_retry - 1:
                    print(f"⚠ [{self.device_id}] 点击未生效，重试 {retry + 1}/{max_retry}")
                    time.sleep(0.5)

        if not click_success:
            print(f"⚠ [{self.device_id}] 多次点击均未成功")
            return None

        # 4. 等待反馈显示
        time.sleep(1.5)

        # 5. 截图2：result（操作后反馈，包含正确答案）
        # 增加短暂延迟避免并发冲突
        time.sleep(0.2)
        result_screenshot = self.capture_and_save(episode_id, round_num, "result")
        if result_screenshot is None:
            print(f"⚠ [{self.device_id}] result截图失败")
            return None

        # 6. 点击"下一轮"按钮
        next_x, next_y = self.calculate_next_button_position()
        success = self.controller.tap(next_x, next_y, delay=1.0)

        if not success:
            print(f"⚠ [{self.device_id}] 点击下一轮失败")
            return None

        # 7. 返回该轮的元数据
        metadata = {
            "round": round_num,
            "question_screenshot": question_screenshot,
            "result_screenshot": result_screenshot,
            "selected_index": selected_index,
            "click_position": [x, y],
            "timestamp": datetime.now().isoformat(),
        }

        return metadata

    def collect_one_episode(self, episode_id: int) -> Dict:
        """
        收集一局游戏的数据（10轮）

        Returns:
            该局的元数据字典
        """
        print(f"\n{'=' * 60}")
        print(f"[{self.device_id}] Episode {episode_id} 开始")
        print(f"{'=' * 60}\n")

        episode_metadata = {
            "episode_id": episode_id,
            "device_id": self.device_id,
            "start_time": datetime.now().isoformat(),
            "rounds": [],
            "completed_rounds": 0,
            "success": False,
        }

        # 收集10轮数据
        completed_rounds = 0
        attempt_count = 0
        max_attempts = 20  # 最多尝试20次

        while completed_rounds < 10 and attempt_count < max_attempts:
            attempt_count += 1
            round_num = completed_rounds + 1

            round_metadata = self.play_one_round(episode_id, round_num)

            if round_metadata is not None:
                episode_metadata["rounds"].append(round_metadata)
                completed_rounds += 1
                print(f"✓ [{self.device_id}] Round {completed_rounds}/10 完成")

                # 轮次间等待
                if completed_rounds < 10:
                    time.sleep(1.0)
            else:
                print(f"⚠ [{self.device_id}] Round {round_num} 失败，重试...")
                time.sleep(1.5)

        episode_metadata["completed_rounds"] = completed_rounds
        episode_metadata["success"] = completed_rounds == 10
        episode_metadata["end_time"] = datetime.now().isoformat()

        # 截取最终得分界面
        time.sleep(2.0)
        final_screenshot_path = self.capture_and_save(episode_id, 99, "final")  # 用99表示final
        if final_screenshot_path:
            episode_metadata["final_screenshot"] = final_screenshot_path

        # 保存该局的元数据
        metadata_file = self.output_dir / f"episode_{episode_id:03d}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(episode_metadata, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print(f"[{self.device_id}] Episode {episode_id} 完成")
        print(f"[{self.device_id}] 成功轮数: {completed_rounds}/10")
        print(f"[{self.device_id}] 元数据已保存: {metadata_file}")
        print(f"{'=' * 60}\n")

        self.episodes.append(episode_metadata)
        return episode_metadata

    def refresh_browser(self):
        """刷新浏览器页面，准备下一局"""
        print(f"[{self.device_id}] 刷新浏览器...")

        # 点击刷新按钮
        refresh_button_x = 380
        refresh_button_y = 130
        self.controller.tap(refresh_button_x, refresh_button_y, delay=1.0)

        time.sleep(3.0)  # 等待页面加载
        print(f"[{self.device_id}] 浏览器已刷新")

    def collect_data(self, num_episodes: int) -> List[Dict]:
        """
        收集多局游戏数据

        Args:
            num_episodes: 要收集的局数

        Returns:
            所有局的元数据列表
        """
        # 从续集ID开始
        for i in range(num_episodes):
            episode_id = self.start_episode_id + i
            self.collect_one_episode(episode_id)

            # 局间刷新浏览器（最后一局不需要）
            if i < num_episodes - 1:
                self.refresh_browser()
                time.sleep(2.0)

        # 保存汇总信息
        summary = {
            "device_id": self.device_id,
            "total_episodes": num_episodes,
            "successful_episodes": sum(1 for ep in self.episodes if ep["success"]),
            "total_rounds_collected": sum(ep["completed_rounds"] for ep in self.episodes),
            "collection_time": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
            "episodes": self.episodes,
        }

        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print(f"[{self.device_id}] 数据收集完成！")
        print(f"[{self.device_id}] 总局数: {num_episodes}")
        print(f"[{self.device_id}] 成功局数: {summary['successful_episodes']}")
        print(f"[{self.device_id}] 总轮数: {summary['total_rounds_collected']}")
        print(f"[{self.device_id}] 汇总文件: {summary_file}")
        print(f"{'=' * 60}\n")

        return self.episodes


def collect_from_device(device_id: str, num_episodes: int, output_dir: str, debug: bool) -> Dict:
    """
    从单个设备收集数据（用于并发执行）

    Returns:
        收集汇总信息
    """
    try:
        collector = DataCollector(device_id=device_id, output_dir=output_dir, debug=debug)

        collector.collect_data(num_episodes)

        # 读取汇总文件
        summary_file = collector.output_dir / "collection_summary.json"
        with open(summary_file, encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print(f"⚠ 设备 {device_id} 收集失败: {e}")
        import traceback

        traceback.print_exc()
        return {"device_id": device_id, "error": str(e), "success": False}


def main():
    parser = argparse.ArgumentParser(description="游戏数据收集脚本（用于离线训练）")

    # 设备配置
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        required=True,
        help="Android 设备地址列表，如: 101.43.137.83:5555 192.168.1.100:5555",
    )

    # 收集配置
    parser.add_argument("--episodes", type=int, default=10, help="每个设备收集多少局游戏（默认10局）")

    parser.add_argument("--output-dir", type=str, default="game_data_raw", help="输出目录（默认 game_data_raw）")

    # 执行模式
    parser.add_argument("--parallel", action="store_true", help="并发执行多个设备（默认顺序执行）")

    parser.add_argument("--max-workers", type=int, default=4, help="并发执行时的最大线程数（默认4）")

    parser.add_argument("--debug", action="store_true", help="开启调试模式")

    args = parser.parse_args()

    print("=" * 60)
    print("游戏数据收集脚本")
    print("=" * 60)
    print(f"设备数量: {len(args.devices)}")
    print(f"每设备局数: {args.episodes}")
    print(f"预计总轮数: {len(args.devices) * args.episodes * 10}")
    print(f"输出目录: {args.output_dir}")
    print(f"执行模式: {'并发' if args.parallel else '顺序'}")
    print("=" * 60)
    print()

    start_time = time.time()
    all_summaries = []

    if args.parallel and len(args.devices) > 1:
        # 并发执行
        print(f"使用 {min(args.max_workers, len(args.devices))} 个线程并发收集数据...\n")

        with ThreadPoolExecutor(max_workers=min(args.max_workers, len(args.devices))) as executor:
            # 提交所有任务
            future_to_device = {
                executor.submit(collect_from_device, device_id, args.episodes, args.output_dir, args.debug): device_id
                for device_id in args.devices
            }

            # 等待完成
            for future in as_completed(future_to_device):
                device_id = future_to_device[future]
                try:
                    summary = future.result()
                    all_summaries.append(summary)
                    print(f"✓ 设备 {device_id} 数据收集完成")
                except Exception as e:
                    print(f"⚠ 设备 {device_id} 发生异常: {e}")
    else:
        # 顺序执行
        for device_id in args.devices:
            print(f"\n处理设备: {device_id}")
            print("-" * 60)

            summary = collect_from_device(device_id, args.episodes, args.output_dir, args.debug)
            all_summaries.append(summary)

    # 生成总汇总
    elapsed_time = time.time() - start_time
    total_summary = {
        "total_devices": len(args.devices),
        "episodes_per_device": args.episodes,
        "total_episodes_collected": sum(s.get("total_episodes", 0) for s in all_summaries),
        "successful_episodes": sum(s.get("successful_episodes", 0) for s in all_summaries),
        "total_rounds_collected": sum(s.get("total_rounds_collected", 0) for s in all_summaries),
        "collection_time_seconds": elapsed_time,
        "output_dir": args.output_dir,
        "timestamp": datetime.now().isoformat(),
        "device_summaries": all_summaries,
    }

    # 保存总汇总
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_summary_file = output_path / "total_summary.json"
    with open(total_summary_file, "w", encoding="utf-8") as f:
        json.dump(total_summary, f, ensure_ascii=False, indent=2)

    # 打印最终结果
    print("\n" + "=" * 60)
    print("所有数据收集完成！")
    print("=" * 60)
    print(f"总设备数: {total_summary['total_devices']}")
    print(f"总局数: {total_summary['total_episodes_collected']}")
    print(f"成功局数: {total_summary['successful_episodes']}")
    print(f"总轮数: {total_summary['total_rounds_collected']}")
    print(f"耗时: {elapsed_time:.1f} 秒 ({elapsed_time / 60:.1f} 分钟)")
    print(f"输出目录: {args.output_dir}")
    print(f"总汇总文件: {total_summary_file}")
    print("=" * 60)
    print("\n下一步: 使用标注脚本对收集的截图进行批量标注")


if __name__ == "__main__":
    main()
