"""
数字选择游戏 Reward 函数（条件反转版本）

奖励逻辑:
- 根据指示灯规则选择正确数字: +1.0
  - 绿灯（max）: 选最大数字
  - 红灯（min）: 选最小数字
  - 黄灯（mid）: 选中间数字
- 选择错误: -1.0
- 动作格式错误: -0.5
- 完成 10 轮: 额外 bonus +2.0
"""

import re
from typing import Any, List, Dict

# Metadata (EasyR1 框架要求)
REWARD_NAME = "number_game"
REWARD_TYPE = "batch"


def parse_action(response: str) -> dict:
    """
    解析模型输出的动作

    预期格式:
    - <action>select(0)</action>  # 选择左边 (索引 0)
    - <action>select(1)</action>  # 选择中间 (索引 1)
    - <action>select(2)</action>  # 选择右边 (索引 2)

    Returns:
        dict: {"type": "select", "index": 0, "valid": True}
    """
    # 匹配 <action>...</action> 标签
    pattern = r"<action>(.*?)</action>"
    matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        return {"type": "unknown", "valid": False, "error": "No action tag found"}

    action_str = matches[0].strip()

    # 解析 select 动作
    select_pattern = r"select\((\d+)\)"
    select_match = re.match(select_pattern, action_str)

    if select_match:
        index = int(select_match.group(1))

        # 验证索引范围
        if 0 <= index <= 2:
            return {
                "type": "select",
                "index": index,
                "valid": True
            }
        else:
            return {
                "type": "select",
                "index": index,
                "valid": False,
                "error": f"Index out of range: {index}"
            }

    return {"type": "unknown", "valid": False, "error": f"Unknown action: {action_str}"}


def compute_selection_reward(numbers: List[int], selected_index: int, rule: str = "max") -> float:
    """
    计算选择奖励（支持条件反转）

    Args:
        numbers: 3 个数字列表，例如 [5, 3, 8]
        selected_index: 选择的索引 (0, 1, 或 2)
        rule: 游戏规则 ("max", "min", "mid")

    Returns:
        float: 奖励值
            +1.0: 选择正确
            -1.0: 选择错误
    """
    if not numbers or len(numbers) != 3:
        return -0.5  # 数据无效

    if not (0 <= selected_index < 3):
        return -0.5  # 索引无效

    selected_value = numbers[selected_index]
    
    # 根据规则确定正确答案
    if rule == "max":
        correct_value = max(numbers)
    elif rule == "min":
        correct_value = min(numbers)
    elif rule == "mid":
        correct_value = sorted(numbers)[1]
    else:
        return -0.5  # 规则无效

    if selected_value == correct_value:
        return 1.0  # 选择正确
    else:
        return -1.0  # 选择错误


def compute_score_delta_reward(score_before: int, score_after: int) -> float:
    """
    计算分数变化奖励（辅助奖励）

    Args:
        score_before: 动作前分数
        score_after: 动作后分数

    Returns:
        float: 归一化奖励
    """
    if score_before < 0 or score_after < 0:
        return 0.0

    delta = score_after - score_before

    # 归一化到 [-1, 1] 范围
    # 游戏中分数变化通常是 +10, +5, -10
    normalized = delta / 10.0

    return max(min(normalized, 1.0), -1.0)


def compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    """
    批量计算 Reward

    Args:
        reward_inputs: List of dicts, each containing:
            - response: 模型输出的动作序列
            - numbers: 3个数字列表 [5, 3, 8]
            - rule: 游戏规则 ("max", "min", "mid")
            - score_before: 动作前分数
            - score_after: 动作后分数 (可选)
            - round: 当前回合 (可选)

    Returns:
        List of dicts with reward components
    """
    scores = []

    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        numbers = reward_input.get("numbers", [])
        rule = reward_input.get("rule", "max")
        score_before = reward_input.get("score_before", -1)
        score_after = reward_input.get("score_after", -1)
        round_num = reward_input.get("round", 1)

        # 1. 动作格式奖励
        action = parse_action(response)
        format_score = 1.0 if action["valid"] else 0.0
        format_penalty = 0.0 if action["valid"] else -0.5

        # 2. 选择奖励（核心奖励）
        selection_reward = 0.0
        if action["valid"] and action["type"] == "select":
            selection_reward = compute_selection_reward(numbers, action["index"], rule)

        # 3. 分数变化奖励（辅助奖励，用于验证）
        score_delta_reward = 0.0
        if score_after >= 0:
            score_delta_reward = compute_score_delta_reward(score_before, score_after)

        # 4. 完成 bonus
        completion_bonus = 0.0
        if round_num >= 10:
            completion_bonus = 2.0

        # 5. 总体奖励
        overall = (
            selection_reward +
            format_penalty +
            completion_bonus
        )

        scores.append({
            "overall": overall,  # 必须字段，用于 GRPO 优化
            "selection": selection_reward,
            "format": format_score,
            "score_delta": score_delta_reward,
            "completion_bonus": completion_bonus,
        })

    return scores


# ============================================================
# 单元测试 (可独立运行验证)
# ============================================================

if __name__ == "__main__":
    # 测试用例
    test_cases = [
        {
            "name": "绿灯-选最大 (正确)",
            "input": {
                "response": "<action>select(2)</action>",
                "numbers": [5, 3, 8],
                "rule": "max",
                "score_before": 100,
                "score_after": 110,
                "round": 5
            },
            "expected_overall": 1.0
        },
        {
            "name": "红灯-选最小 (正确)",
            "input": {
                "response": "<action>select(1)</action>",
                "numbers": [5, 3, 8],
                "rule": "min",
                "score_before": 100,
                "score_after": 110,
                "round": 5
            },
            "expected_overall": 1.0
        },
        {
            "name": "黄灯-选中间 (正确)",
            "input": {
                "response": "<action>select(0)</action>",
                "numbers": [5, 3, 8],
                "rule": "mid",
                "score_before": 100,
                "score_after": 110,
                "round": 5
            },
            "expected_overall": 1.0
        },
        {
            "name": "绿灯-选最小 (错误)",
            "input": {
                "response": "<action>select(1)</action>",
                "numbers": [5, 3, 8],
                "rule": "max",
                "score_before": 100,
                "score_after": 90,
                "round": 5
            },
            "expected_overall": -1.0
        },
        {
            "name": "完成游戏 (Bonus)",
            "input": {
                "response": "<action>select(2)</action>",
                "numbers": [5, 3, 8],
                "rule": "max",
                "score_before": 100,
                "score_after": 110,
                "round": 10
            },
            "expected_overall": 3.0  # 1.0 (选择) + 2.0 (bonus)
        },
        {
            "name": "格式错误",
            "input": {
                "response": "我认为应该选择最大的",
                "numbers": [5, 3, 8],
                "rule": "max",
                "score_before": 100,
                "score_after": 100,
                "round": 5
            },
            "expected_overall": -0.5
        }
    ]

    print("=" * 60)
    print("数字选择游戏 Reward 函数单元测试（条件反转版本）")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[测试 {i}] {test['name']}")
        print(f"输入: {test['input']['response']}")
        print(f"数字: {test['input']['numbers']}")
        print(f"规则: {test['input']['rule']}")

        result = compute_score([test["input"]])[0]

        print(f"总奖励: {result['overall']:.2f}")
        print(f"  - 选择奖励: {result['selection']:.2f}")
        print(f"  - 格式正确: {result['format']:.2f}")
        print(f"  - 分数变化: {result['score_delta']:.2f}")
        print(f"  - 完成奖励: {result['completion_bonus']:.2f}")
        print(f"预期: {test['expected_overall']}")

        # 验证
        if abs(result['overall'] - test['expected_overall']) < 0.01:
            print("✓ 通过")
        else:
            print("✗ 失败")

    print("\n" + "=" * 60)
    print("测试完成！请检查结果是否符合预期。")
    print("=" * 60)
