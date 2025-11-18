"""
数据集生成脚本

将标注好的截图转换为 EasyR1 训练格式 (JSONL)
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict


def load_annotations(annotation_file: str) -> List[Dict]:
    """加载标注文件"""
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    print(f"✓ 加载标注: {len(annotations)} 条")
    return annotations


def validate_annotation(annotation: Dict) -> bool:
    """验证标注是否完整"""
    manual = annotation.get("manual_annotation", {})

    if not manual.get("action"):
        print(f"⚠ 跳过未标注: {annotation['image']}")
        return False

    return True


def convert_to_training_format(
    annotations: List[Dict],
    raw_screenshots_dir: str,
    system_prompt: str = "你是一个砍树游戏助手，根据截图决定下一步操作。"
) -> List[Dict]:
    """
    转换为训练格式

    EasyR1 要求的格式:
    {
        "prompt": str,
        "images": List[str],
        "answer": str,
        "combat_power_before": float,
        "combat_power_after": float (可选)
    }
    """
    training_data = []

    for i, ann in enumerate(annotations):
        if not validate_annotation(ann):
            continue

        # 构造完整的图片路径
        image_path = str(Path(raw_screenshots_dir) / ann["image"])

        # 构造 prompt
        prompt = system_prompt

        # 构造 answer (模型应该输出的内容)
        action = ann["manual_annotation"]["action"]
        answer = f"<action>{action}</action>"

        # 基础数据
        data_item = {
            "prompt": prompt,
            "images": [image_path],
            "answer": answer,
            "combat_power_before": ann.get("combat_power", -1),
            "state": ann.get("state", "unknown"),
            "description": ann["manual_annotation"].get("description", "")
        }

        # 如果有装备信息，添加额外字段
        if ann.get("equipment_stats"):
            data_item["equipment_stats"] = ann["equipment_stats"]
            data_item["estimated_power_change"] = ann.get("estimated_power_change", 0)

        training_data.append(data_item)

    return training_data


def split_train_val(data: List[Dict], val_ratio: float = 0.2) -> tuple:
    """划分训练集和验证集"""
    import random

    # 打乱数据
    random.seed(42)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # 划分
    val_size = int(len(shuffled_data) * val_ratio)
    val_data = shuffled_data[:val_size]
    train_data = shuffled_data[val_size:]

    return train_data, val_data


def save_jsonl(data: List[Dict], output_file: str):
    """保存为 JSONL 格式"""
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✓ 保存: {output_file} ({len(data)} 条)")


def main():
    parser = argparse.ArgumentParser(description="生成训练数据集")

    parser.add_argument(
        "--input",
        type=str,
        default="wechat_tree_game_agent/data/raw_screenshots/annotations.json",
        help="标注文件路径"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="wechat_tree_game_agent/data/",
        help="输出目录"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("数据集生成工具")
    print("=" * 60)

    # 1. 加载标注
    annotations = load_annotations(args.input)

    # 2. 转换格式
    raw_screenshots_dir = str(Path(args.input).parent)
    training_data = convert_to_training_format(annotations, raw_screenshots_dir)

    print(f"✓ 转换完成: {len(training_data)} 条有效数据")

    # 3. 划分训练集和验证集
    train_data, val_data = split_train_val(training_data, args.val_ratio)

    print(f"✓ 训练集: {len(train_data)} 条")
    print(f"✓ 验证集: {len(val_data)} 条")

    # 4. 保存
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_data, str(output_dir / "tree_game_dataset.jsonl"))
    save_jsonl(val_data, str(output_dir / "tree_game_val.jsonl"))

    print("\n" + "=" * 60)
    print("数据集生成完成！")
    print("=" * 60)

    # 打印数据统计
    print("\n数据统计:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")

    # 统计各状态数量
    state_counts = {}
    for item in training_data:
        state = item.get("state", "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1

    print("\n状态分布:")
    for state, count in state_counts.items():
        print(f"  {state}: {count} 条")

    print("\n下一步:")
    print("  运行训练: bash wechat_tree_game_agent/scripts/train.sh")


if __name__ == "__main__":
    main()
