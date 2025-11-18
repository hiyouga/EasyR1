"""
构建训练数据集

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
    if not annotation.get("action"):
        print(f"⚠ 跳过未标注: {annotation['image']}")
        return False

    # 跳过结果界面（用于验证，不用于训练）
    if annotation.get("state") == "result":
        return False

    return True


def convert_to_training_format(
    annotations: List[Dict],
    screenshot_dir: str,
    system_prompt: str = "根据游戏截图，判断当前状态并输出最优动作，最大化妖力。"
) -> List[Dict]:
    """
    转换为训练格式

    EasyR1 要求的格式:
    {
        "prompt": str,
        "images": List[str],
        "answer": str
    }
    """
    training_data = []

    for ann in annotations:
        if not validate_annotation(ann):
            continue

        # 构造完整的图片路径
        image_path = str(Path(screenshot_dir) / ann["image"])

        # 构造 prompt
        if ann["state"] == "tree_cutting":
            prompt = "当前是砍树界面，请点击屏幕下方中央的斧子开始砍树。"
        elif ann["state"] == "equipment_selection":
            if ann.get("expected_change") == "positive":
                prompt = "装备掉落了！新装备属性整体提升，妖力会上升。"
            elif ann.get("expected_change") == "negative":
                prompt = "装备掉落了！新装备属性整体下降，妖力会下降。"
            elif ann.get("expected_change") == "mixed":
                prompt = "装备掉落了！新装备属性有升有降，需要判断总妖力变化。"
            else:
                prompt = "装备掉落了！请判断是替换还是分解。"
        else:
            prompt = system_prompt

        # 构造 answer (模型应该输出的内容)
        action = ann["action"]
        answer = f"<action>{action}</action>"

        # 基础数据
        data_item = {
            "prompt": prompt,
            "images": [image_path],
            "answer": answer,
            "state": ann.get("state", "unknown"),
            "description": ann.get("description", "")
        }

        # 添加元数据（可选，用于调试）
        if ann.get("expected_change"):
            data_item["expected_change"] = ann["expected_change"]

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


def print_dataset_info(train_data: List[Dict], val_data: List[Dict]):
    """打印数据集信息"""
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)

    print(f"\n训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")

    # 统计各状态数量
    all_data = train_data + val_data
    state_counts = {}
    action_counts = {}

    for item in all_data:
        state = item.get("state", "unknown")
        action = item.get("answer", "").replace("<action>", "").replace("</action>", "")

        state_counts[state] = state_counts.get(state, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1

    print("\n状态分布:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state:25s}: {count:3d} 条")

    print("\n动作分布:")
    for action, count in sorted(action_counts.items()):
        print(f"  {action:25s}: {count:3d} 条")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="生成训练数据集")

    parser.add_argument(
        "--input",
        type=str,
        default="wechat_tree_game_agent/data/annotations.json",
        help="标注文件路径"
    )

    parser.add_argument(
        "--screenshot-dir",
        type=str,
        default="/Users/zhangyuehua/Desktop/tree_cutting",
        help="截图目录路径"
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
    print("数据集生成工具 - 微信砍树游戏")
    print("=" * 60)

    # 1. 加载标注
    annotations = load_annotations(args.input)

    # 2. 转换格式
    training_data = convert_to_training_format(annotations, args.screenshot_dir)
    print(f"✓ 转换完成: {len(training_data)} 条有效数据")

    # 3. 划分训练集和验证集
    train_data, val_data = split_train_val(training_data, args.val_ratio)
    print(f"✓ 训练集: {len(train_data)} 条")
    print(f"✓ 验证集: {len(val_data)} 条")

    # 4. 保存
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_data, str(output_dir / "tree_game_train.jsonl"))
    save_jsonl(val_data, str(output_dir / "tree_game_val.jsonl"))

    # 5. 打印统计信息
    print_dataset_info(train_data, val_data)

    print("\n" + "=" * 60)
    print("数据集生成完成！")
    print("=" * 60)

    print("\n生成的文件:")
    print(f"  - 训练集: {output_dir / 'tree_game_train.jsonl'}")
    print(f"  - 验证集: {output_dir / 'tree_game_val.jsonl'}")

    print("\n下一步:")
    print("  1. 检查生成的 JSONL 文件")
    print("  2. 更新训练配置文件 config/tree_game_grpo.yaml")
    print("  3. 运行训练: bash wechat_tree_game_agent/scripts/train.sh")


if __name__ == "__main__":
    main()
