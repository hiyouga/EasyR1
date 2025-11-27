"""
Number Game Reward Function

评分规则：
- 选择正确的数字: +1.0
- 选择错误的数字: 0.0

输入格式:
reward_input = {
    "response": "1",  # 模型输出的答案 (0/1/2)
    "response_length": 10,  # 响应长度（token数）
    "ground_truth": "1"  # 正确答案 (0/1/2)
}

输出格式:
{
    "overall": 1.0,  # 总分（必需字段）
    "accuracy": 1.0  # 准确率（可选，用于监控）
}
"""

import re
from typing import Any


# Metadata - EasyR1框架要求
REWARD_NAME = "number_game"
REWARD_TYPE = "batch"  # 批量处理模式


def extract_answer(response: str) -> str:
    """
    从模型响应中提取答案索引

    Args:
        response: 模型的原始响应

    Returns:
        "0", "1", "2" 或 ""（提取失败）
    """
    # 情况1: 响应本身就是单个数字
    response = response.strip()
    if response in ["0", "1", "2"]:
        return response

    # 情况2: 响应包含多余文字，提取第一个出现的0/1/2
    match = re.search(r"[012]", response)
    if match:
        return match.group(0)

    # 提取失败
    return ""


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """
    计算一批样本的得分

    Args:
        reward_inputs: 包含多个样本的列表，每个样本包含:
            - response: 模型的响应
            - response_length: 响应长度
            - ground_truth: 正确答案

    Returns:
        每个样本的得分字典列表，包含:
            - overall: 总分（1.0表示正确，0.0表示错误）
            - accuracy: 准确率（同overall，用于监控）
    """
    scores = []

    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")

        # 提取答案
        predicted = extract_answer(response)

        # 计算得分
        if predicted == ground_truth:
            score = 1.0
        else:
            score = 0.0

        # 返回格式：必须包含overall字段
        scores.append({"overall": score, "accuracy": score})

    return scores


# 测试用例
if __name__ == "__main__":
    test_cases = [
        # 完美匹配
        {"response": "0", "response_length": 1, "ground_truth": "0"},
        {"response": "1", "response_length": 1, "ground_truth": "1"},
        {"response": "2", "response_length": 1, "ground_truth": "2"},
        # 响应包含额外文字
        {"response": "The answer is 1", "response_length": 15, "ground_truth": "1"},
        {"response": "I choose option 2", "response_length": 18, "ground_truth": "2"},
        # 错误答案
        {"response": "0", "response_length": 1, "ground_truth": "1"},
        {"response": "2", "response_length": 1, "ground_truth": "0"},
        # 提取失败
        {"response": "I don't know", "response_length": 12, "ground_truth": "1"},
        {"response": "", "response_length": 0, "ground_truth": "2"},
    ]

    scores = compute_score(test_cases)

    print("Reward Function Test Results:")
    print("=" * 60)
    for i, (test, score) in enumerate(zip(test_cases, scores), 1):
        print(f"{i}. Response: {test['response']!r}")
        print(f"   Ground Truth: {test['ground_truth']!r}")
        print(f"   Score: {score}")
        print()
