"""
截图收集工具

用于收集砍树游戏的训练数据
"""

import argparse
import json
import time
from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from wechat_tree_game_agent.android_env import ADBController, GameStateParser


def collect_screenshots(
    device_id: str,
    output_dir: str,
    count: int = 25,
    interval: float = 3.0
):
    """
    自动收集截图

    Args:
        device_id: Android 设备 ID
        output_dir: 输出目录
        count: 截图数量
        interval: 截图间隔 (秒)
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 初始化控制器和解析器
    controller = ADBController(device_id=device_id)
    parser = GameStateParser(use_gpu=False)

    print("=" * 60)
    print("截图收集工具")
    print("=" * 60)
    print(f"设备: {device_id}")
    print(f"输出目录: {output_dir}")
    print(f"目标数量: {count} 张")
    print(f"间隔: {interval} 秒")
    print("=" * 60)

    annotations = []

    print("\n请手动操作游戏，工具将自动截图并解析...")
    print("按 Ctrl+C 提前结束\n")

    try:
        for i in range(count):
            print(f"[{i+1}/{count}] 截图中...")

            # 截图
            screenshot_filename = f"screenshot_{i+1:03d}.jpg"
            screenshot_path = output_path / screenshot_filename

            image = controller.capture_screenshot(save_path=str(screenshot_path))

            # 解析截图
            parsed = parser.parse_screenshot(image)

            # 记录标注
            annotation = {
                "id": i + 1,
                "image": screenshot_filename,
                "state": parsed["state"],
                "combat_power": parsed["combat_power"],
                "equipment_stats": parsed["equipment_stats"],
                "estimated_power_change": parsed["estimated_power_change"],
                "manual_annotation": {
                    "action": None,  # 待手动标注
                    "description": None  # 待手动标注
                }
            }

            annotations.append(annotation)

            # 打印解析结果
            print(f"  状态: {parsed['state']}")
            print(f"  战斗力: {parsed['combat_power']}")
            if parsed["equipment_stats"]:
                print(f"  装备属性: {parsed['equipment_stats']}")
                print(f"  估算战斗力变化: {parsed['estimated_power_change']}")

            # 等待下一次截图
            if i < count - 1:
                time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n用户中断，提前结束收集")

    # 保存标注文件
    annotation_file = output_path / "annotations.json"
    with open(annotation_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"✓ 收集完成！共 {len(annotations)} 张截图")
    print(f"✓ 标注文件: {annotation_file}")
    print("=" * 60)

    print("\n下一步:")
    print("1. 检查截图和 OCR 识别结果")
    print("2. 编辑 annotations.json，填写 manual_annotation 字段")
    print("3. 运行 process_dataset.py 生成训练数据")


def manual_annotation_guide():
    """打印手动标注指南"""
    guide = """
手动标注指南
============================================================

打开 annotations.json，为每个截图填写:

{
  "manual_annotation": {
    "action": "click(360, 800)" | "equip()" | "skip()",
    "description": "点击砍树按钮" | "装备（总战斗力+60）" | "跳过（战斗力-30）"
  }
}

标注原则:
1. 如果是砍树界面，标注点击坐标
2. 如果是装备界面:
   - 总战斗力上升 → action: "equip()"
   - 总战斗力不变或下降 → action: "skip()"
3. description 简要描述决策理由

示例:
--------------------------------------------------------------
// 砍树界面
{
  "image": "screenshot_001.jpg",
  "state": "tree_cutting",
  "combat_power": 1250,
  "manual_annotation": {
    "action": "click(360, 800)",
    "description": "点击砍树按钮（屏幕下方中央）"
  }
}

// 装备界面（应该装备）
{
  "image": "screenshot_002.jpg",
  "state": "equipment_selection",
  "combat_power": 1250,
  "equipment_stats": {"attack": "+50 ↑", "defense": "-10 ↓"},
  "estimated_power_change": 42,
  "manual_annotation": {
    "action": "equip()",
    "description": "总战斗力预期+42，应该装备"
  }
}

// 装备界面（应该跳过）
{
  "image": "screenshot_003.jpg",
  "state": "equipment_selection",
  "combat_power": 1250,
  "equipment_stats": {"attack": "+10 ↑", "defense": "-50 ↓"},
  "estimated_power_change": -30,
  "manual_annotation": {
    "action": "skip()",
    "description": "总战斗力预期-30，应该跳过"
  }
}
============================================================
"""
    print(guide)


if __name__ == "__main__":
    parser_args = argparse.ArgumentParser(description="砍树游戏截图收集工具")

    parser_args.add_argument(
        "--device",
        type=str,
        default="emulator-5554",
        help="Android 设备 ID (通过 'adb devices' 查看)"
    )

    parser_args.add_argument(
        "--output",
        type=str,
        default="wechat_tree_game_agent/data/raw_screenshots",
        help="输出目录"
    )

    parser_args.add_argument(
        "--count",
        type=int,
        default=25,
        help="截图数量"
    )

    parser_args.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="截图间隔 (秒)"
    )

    parser_args.add_argument(
        "--guide",
        action="store_true",
        help="显示手动标注指南"
    )

    args = parser_args.parse_args()

    if args.guide:
        manual_annotation_guide()
    else:
        collect_screenshots(
            device_id=args.device,
            output_dir=args.output,
            count=args.count,
            interval=args.interval
        )
