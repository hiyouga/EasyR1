"""
自动标注脚本

基于文件名和简单规则自动生成训练数据标注
"""

import json
import os
from pathlib import Path
from typing import List, Dict


def categorize_screenshots(screenshot_dir: str) -> Dict[str, List[str]]:
    """
    根据文件名自动分类截图

    Returns:
        {
            "tree_cutting": ["主界面_001.png", ...],
            "equipment_positive": ["装备掉落_001.png", ...],
            "equipment_negative": ["装备掉落_降低战力_001.png", ...],
            "equipment_mixed": ["装备掉落_混合_001.png", ...],
            "result": ["装备替换结果_001.png", ...]
        }
    """
    screenshot_path = Path(screenshot_dir)
    files = sorted([f for f in os.listdir(screenshot_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    categorized = {
        "tree_cutting": [],
        "equipment_positive": [],
        "equipment_negative": [],
        "equipment_mixed": [],
        "result": []
    }

    for filename in files:
        if filename.startswith("主界面") or filename.startswith("砍树的位置"):
            categorized["tree_cutting"].append(filename)
        elif "降低战力" in filename or "降低妖力" in filename:
            categorized["equipment_negative"].append(filename)
        elif "混合" in filename:
            categorized["equipment_mixed"].append(filename)
        elif filename.startswith("装备掉落"):
            categorized["equipment_positive"].append(filename)
        elif filename.startswith("装备替换结果"):
            categorized["result"].append(filename)

    return categorized


def generate_annotations(categorized: Dict[str, List[str]], screenshot_dir: str) -> List[Dict]:
    """
    生成自动标注

    标注规则:
    1. 主界面/砍树界面 -> click(180, 1000) (屏幕下方中央)
    2. 装备掉落_正向 -> replace() (妖力上升)
    3. 装备掉落_负向 -> decompose() (妖力下降)
    4. 装备掉落_混合 -> 需要手动判断（暂时标记为 replace）
    5. 结果界面 -> 跳过（用于验证）
    """
    annotations = []
    idx = 1

    # 处理砍树界面
    for filename in categorized["tree_cutting"]:
        annotations.append({
            "id": idx,
            "image": filename,
            "state": "tree_cutting",
            "action": "click(180, 1000)",
            "description": "点击屏幕下方中央的斧子按钮进行砍树",
            "demon_power_before": None,  # 需要OCR提取
            "demon_power_after": None
        })
        idx += 1

    # 处理装备掉落_正向（妖力上升）
    for filename in categorized["equipment_positive"]:
        annotations.append({
            "id": idx,
            "image": filename,
            "state": "equipment_selection",
            "action": "replace()",
            "description": "装备属性整体提升，妖力上升，应该替换",
            "demon_power_before": None,
            "demon_power_after": None,
            "expected_change": "positive"
        })
        idx += 1

    # 处理装备掉落_负向（妖力下降）
    for filename in categorized["equipment_negative"]:
        annotations.append({
            "id": idx,
            "image": filename,
            "state": "equipment_selection",
            "action": "decompose()",
            "description": "装备属性整体下降，妖力下降，应该分解",
            "demon_power_before": None,
            "demon_power_after": None,
            "expected_change": "negative"
        })
        idx += 1

    # 处理装备掉落_混合（需要仔细计算）
    for filename in categorized["equipment_mixed"]:
        annotations.append({
            "id": idx,
            "image": filename,
            "state": "equipment_selection",
            "action": "replace()",  # 默认为替换，需要手动验证
            "description": "装备属性有升有降，需要计算总妖力变化",
            "demon_power_before": None,
            "demon_power_after": None,
            "expected_change": "mixed",
            "manual_review_required": True
        })
        idx += 1

    # 处理结果界面（用于验证）
    for filename in categorized["result"]:
        annotations.append({
            "id": idx,
            "image": filename,
            "state": "result",
            "action": None,
            "description": "装备替换后的结果界面，用于验证妖力变化",
            "demon_power_before": None,
            "demon_power_after": None
        })
        idx += 1

    return annotations


def save_annotations(annotations: List[Dict], output_file: str):
    """保存标注文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print(f"✓ 标注文件已保存: {output_file}")
    print(f"✓ 总计 {len(annotations)} 条标注")


def print_statistics(categorized: Dict[str, List[str]]):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("截图分类统计")
    print("=" * 60)

    for category, files in categorized.items():
        print(f"{category:25s}: {len(files):3d} 张")

    total = sum(len(files) for files in categorized.values())
    print("-" * 60)
    print(f"{'总计':25s}: {total:3d} 张")
    print("=" * 60)


def main():
    """主函数"""
    # 配置路径
    screenshot_dir = "/Users/zhangyuehua/Desktop/tree_cutting"
    output_file = "/Users/zhangyuehua/Documents/my_fork/EasyR1/wechat_tree_game_agent/data/annotations.json"

    print("=" * 60)
    print("自动标注工具 - 微信砍树游戏")
    print("=" * 60)

    # 1. 分类截图
    print("\n[步骤 1/3] 分类截图...")
    categorized = categorize_screenshots(screenshot_dir)
    print_statistics(categorized)

    # 2. 生成标注
    print("\n[步骤 2/3] 生成自动标注...")
    annotations = generate_annotations(categorized, screenshot_dir)

    # 3. 保存标注
    print("\n[步骤 3/3] 保存标注文件...")
    save_annotations(annotations, output_file)

    # 打印需要手动检查的项
    manual_review = [ann for ann in annotations if ann.get("manual_review_required")]
    if manual_review:
        print(f"\n⚠ 需要手动检查的标注: {len(manual_review)} 条")
        for ann in manual_review:
            print(f"  - {ann['image']}: {ann['description']}")

    print("\n" + "=" * 60)
    print("自动标注完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 检查生成的 annotations.json 文件")
    print("2. 如果需要，手动修正标注")
    print("3. 运行 process_dataset.py 生成训练数据集")


if __name__ == "__main__":
    main()
