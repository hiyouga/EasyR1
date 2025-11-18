"""
测试 Reward 函数

运行: python wechat_tree_game_agent/tests/test_reward_function.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from wechat_tree_game_agent.reward_function.tree_game_reward import compute_score


def test_reward_function():
    """测试 Reward 函数的各种场景"""

    print("=" * 60)
    print("Reward 函数测试")
    print("=" * 60)

    test_cases = [
        {
            "name": "战斗力上升 (正确装备)",
            "input": {
                "response": "<action>equip()</action>",
                "combat_power_before": 1000,
                "combat_power_after": 1150,
                "step_count": 5
            },
            "expected": {
                "overall": ">1.0",
                "power_change": ">1.0"
            }
        },
        {
            "name": "战斗力下降 (错误装备)",
            "input": {
                "response": "<action>equip()</action>",
                "combat_power_before": 1000,
                "combat_power_after": 950,
                "step_count": 5
            },
            "expected": {
                "overall": "<0",
                "power_change": "<0"
            }
        },
        {
            "name": "战斗力不变 (跳过装备)",
            "input": {
                "response": "<action>skip()</action>",
                "combat_power_before": 1000,
                "combat_power_after": 1000,
                "step_count": 5
            },
            "expected": {
                "overall": "≈0",
                "power_change": "=0"
            }
        },
        {
            "name": "完成任务 (10 步)",
            "input": {
                "response": "<action>click(360, 800)</action>",
                "combat_power_before": 1000,
                "combat_power_after": 1050,
                "step_count": 10
            },
            "expected": {
                "overall": ">2.0",  # 战斗力奖励 + 完成奖励
                "completion_bonus": "=2.0"
            }
        },
        {
            "name": "格式错误",
            "input": {
                "response": "我认为应该点击",
                "combat_power_before": 1000,
                "combat_power_after": 1000,
                "step_count": 5
            },
            "expected": {
                "overall": "<0",
                "format": "=0"
            }
        }
    ]

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\n[测试 {i}] {test['name']}")
        print(f"输入:")
        print(f"  Response: {test['input']['response']}")
        print(f"  战斗力: {test['input']['combat_power_before']} → {test['input']['combat_power_after']}")

        result = compute_score([test["input"]])[0]

        print(f"\n结果:")
        print(f"  总奖励: {result['overall']:.2f}")
        print(f"  战斗力变化: {result['power_change']:.2f}")
        print(f"  格式正确: {result['format']:.2f}")
        print(f"  完成奖励: {result['completion_bonus']:.2f}")
        print(f"  效率: {result['efficiency']:.2f}")

        # 简单验证
        success = True
        for key, expected_value in test["expected"].items():
            actual_value = result[key]

            if expected_value.startswith(">"):
                threshold = float(expected_value[1:])
                if not (actual_value > threshold):
                    success = False
                    print(f"  ✗ {key} 应该 > {threshold}，实际 {actual_value}")
            elif expected_value.startswith("<"):
                threshold = float(expected_value[1:])
                if not (actual_value < threshold):
                    success = False
                    print(f"  ✗ {key} 应该 < {threshold}，实际 {actual_value}")
            elif expected_value.startswith("="):
                threshold = float(expected_value[1:])
                if not (abs(actual_value - threshold) < 0.1):
                    success = False
                    print(f"  ✗ {key} 应该 ≈ {threshold}，实际 {actual_value}")

        if success:
            print(f"  ✓ 测试通过")
            passed += 1
        else:
            print(f"  ✗ 测试失败")
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试完成！通过: {passed}/{len(test_cases)}")
    if failed == 0:
        print("所有测试通过！✅")
    else:
        print(f"失败: {failed} 个测试 ❌")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_reward_function()
    sys.exit(0 if success else 1)
