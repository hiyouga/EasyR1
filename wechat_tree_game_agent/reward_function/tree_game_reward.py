"""
微信砍树游戏 Reward 函数

奖励逻辑（基于妖力变化）:
- 妖力上升: +1.0 ~ +2.0 (根据上升幅度)
- 妖力不变: 0.0 (容错)
- 妖力下降: -1.0 ~ -2.0 (根据下降幅度)
- 完成 10 次砍树: 额外 bonus +2.0
- 动作格式错误: -0.5
"""

import re
from typing import Any

# Metadata (EasyR1 框架要求)
REWARD_NAME = "tree_game"
REWARD_TYPE = "batch"


def parse_action(response: str) -> dict:
    """
    解析模型输出的动作

    预期格式:
    - <action>click(360, 800)</action>  # 点击砍树按钮
    - <action>replace()</action>        # 点击"替换"按钮（装备）
    - <action>decompose()</action>      # 点击"分解"按钮（跳过）

    Returns:
        dict: {"type": "click", "coords": [360, 800], "valid": True}
    """
    # 匹配 <action>...</action> 标签
    pattern = r"<action>(.*?)</action>"
    matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        return {"type": "unknown", "valid": False, "error": "No action tag found"}

    action_str = matches[0].strip()

    # 解析 click 动作（点击砍树按钮）
    click_pattern = r"click\((\d+),\s*(\d+)\)"
    click_match = re.match(click_pattern, action_str)
    if click_match:
        return {
            "type": "click",
            "coords": [int(click_match.group(1)), int(click_match.group(2))],
            "valid": True
        }

    # 解析 replace 动作（替换装备，妖力上升）
    if action_str in ["replace()", "equip()"]:  # 兼容旧格式
        return {"type": "replace", "valid": True}

    # 解析 decompose 动作（分解装备，妖力不变）
    if action_str in ["decompose()", "skip()"]:  # 兼容旧格式
        return {"type": "decompose", "valid": True}

    return {"type": "unknown", "valid": False, "error": f"Unknown action: {action_str}"}


def extract_combat_power(ocr_result: dict) -> float:
    """
    从 OCR 结果中提取战斗力数值

    Args:
        ocr_result: {"text": "战斗力: 1250", "confidence": 0.95}

    Returns:
        float: 战斗力数值，提取失败返回 -1
    """
    if not ocr_result or "text" not in ocr_result:
        return -1.0

    text = ocr_result["text"]

    # 匹配数字 (支持中文和英文标识)
    # 例如: "战斗力: 1250" 或 "Combat Power: 1250"
    pattern = r"(\d+)"
    matches = re.findall(pattern, text)

    if matches:
        # 取最大的数字 (通常是战斗力)
        return float(max(matches, key=lambda x: int(x)))

    return -1.0


def compute_power_change_reward(power_before: float, power_after: float) -> float:
    """
    计算妖力变化奖励

    Args:
        power_before: 动作前妖力
        power_after: 动作后妖力

    Returns:
        float: 奖励值
    """
    if power_before < 0 or power_after < 0:
        # 无法识别妖力，给予轻微惩罚
        return -0.2

    delta = power_after - power_before

    if delta > 0:
        # 妖力上升，正奖励 (奖励与提升幅度正相关)
        # 妖力通常提升几百到几千，归一化到合理范围
        normalized_delta = min(delta / 1000.0, 1.0)
        return 1.0 + normalized_delta  # 最高 2.0
    elif delta == 0:
        # 妖力不变，零奖励 (容错，例如点击"分解")
        return 0.0
    else:
        # 妖力下降，负奖励 (惩罚与下降幅度正相关)
        normalized_delta = max(delta / 1000.0, -1.0)
        return -1.0 + normalized_delta  # 最低 -2.0


def compute_score(reward_inputs: list[dict[str, Any]], **kwargs) -> list[dict[str, float]]:
    """
    批量计算 Reward

    Args:
        reward_inputs: List of dicts, each containing:
            - response: 模型输出的动作序列
            - ground_truth: 标准答案 (可选)
            - combat_power_before: 动作前战斗力
            - combat_power_after: 动作后战斗力 (可选)
            - ocr_result: OCR 识别结果 (可选)
            - step_count: 当前步数 (可选)

    Returns:
        List of dicts with reward components
    """
    scores = []

    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        power_before = reward_input.get("combat_power_before", -1)
        power_after = reward_input.get("combat_power_after", -1)
        step_count = reward_input.get("step_count", 0)

        # 1. 动作格式奖励
        action = parse_action(response)
        format_score = 1.0 if action["valid"] else 0.0
        format_penalty = 0.0 if action["valid"] else -0.5

        # 2. 战斗力变化奖励
        power_change_score = compute_power_change_reward(power_before, power_after)

        # 3. 任务完成 Bonus
        completion_bonus = 0.0
        if step_count >= 10:  # 完成 10 次砍树
            completion_bonus = 2.0

        # 4. 效率奖励 (步数越少越好)
        efficiency_score = 0.0
        if step_count > 0:
            # 理想步数 10-15，超过则轻微惩罚
            if step_count <= 15:
                efficiency_score = 0.2
            elif step_count <= 20:
                efficiency_score = 0.0
            else:
                efficiency_score = -0.1 * (step_count - 20) / 10.0

        # 5. 总体奖励
        overall = (
            power_change_score +
            format_penalty +
            completion_bonus +
            efficiency_score
        )

        scores.append({
            "overall": overall,  # 必须字段，用于 GRPO 优化
            "power_change": power_change_score,
            "format": format_score,
            "completion_bonus": completion_bonus,
            "efficiency": efficiency_score,
        })

    return scores


# ============================================================
# 单元测试 (可独立运行验证)
# ============================================================

if __name__ == "__main__":
    # 测试用例
    test_cases = [
        {
            "name": "战斗力上升 (正确装备)",
            "input": {
                "response": "<action>equip()</action>",
                "combat_power_before": 1000,
                "combat_power_after": 1150,
                "step_count": 5
            },
            "expected_overall": ">1.0"
        },
        {
            "name": "战斗力下降 (错误装备)",
            "input": {
                "response": "<action>equip()</action>",
                "combat_power_before": 1000,
                "combat_power_after": 950,
                "step_count": 5
            },
            "expected_overall": "<0"
        },
        {
            "name": "战斗力不变 (跳过装备)",
            "input": {
                "response": "<action>skip()</action>",
                "combat_power_before": 1000,
                "combat_power_after": 1000,
                "step_count": 5
            },
            "expected_overall": "≈0"
        },
        {
            "name": "完成任务 (Bonus)",
            "input": {
                "response": "<action>click(360, 800)</action>",
                "combat_power_before": 1000,
                "combat_power_after": 1050,
                "step_count": 10
            },
            "expected_overall": ">3.0"
        },
        {
            "name": "格式错误",
            "input": {
                "response": "我认为应该点击",
                "combat_power_before": 1000,
                "combat_power_after": 1000,
                "step_count": 5
            },
            "expected_overall": "<0"
        }
    ]

    print("=" * 60)
    print("微信砍树游戏 Reward 函数单元测试")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[测试 {i}] {test['name']}")
        print(f"输入: {test['input']['response']}")
        print(f"战斗力: {test['input']['combat_power_before']} → {test['input']['combat_power_after']}")

        result = compute_score([test["input"]])[0]

        print(f"总奖励: {result['overall']:.2f}")
        print(f"  - 战斗力变化: {result['power_change']:.2f}")
        print(f"  - 格式正确: {result['format']:.2f}")
        print(f"  - 完成奖励: {result['completion_bonus']:.2f}")
        print(f"  - 效率: {result['efficiency']:.2f}")
        print(f"预期: {test['expected_overall']}")

    print("\n" + "=" * 60)
    print("测试完成！请检查结果是否符合预期。")
    print("=" * 60)
