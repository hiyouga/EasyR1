import re

from mathruler.grader import grade_answer


def r1v_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0


def r1v_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        ground_truth = ground_truth.strip()
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        pred_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if grade_answer(pred_answer, ground_truth):
            return 1.0
    except Exception:
        pass

    return 0.0


def r1v_compute_score(predict_str: str, ground_truth: str) -> float:
    acc_reward = r1v_accuracy_reward(predict_str, ground_truth)
    format_reward = r1v_format_reward(predict_str)
    reward = acc_reward + format_reward
    reward /= 2
    return reward
