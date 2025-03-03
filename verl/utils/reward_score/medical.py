from mathruler.grader import extract_boxed_content


def parse_conditions(text):
    # Remove any boxing notation if present
    text = text.replace("\\boxed{", "").replace("}", "")

    # Split by common separators
    for sep in [", ", " and ", " & ", ",", "&"]:
        if sep in text:
            return set(cond.strip() for cond in text.split(sep))

    # If no separator found, treat as single condition
    return {text.strip()}


def medical_compute_score(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    if answer == "None":
        return 0.0  # no answer

    # Parse both prediction and ground truth into sets of conditions
    predicted_conditions = parse_conditions(answer)
    ground_truth_conditions = parse_conditions(ground_truth)

    # Calculate true positives, false positives, and false negatives
    true_positives = len(predicted_conditions.intersection(ground_truth_conditions))
    false_positives = len(predicted_conditions - ground_truth_conditions)
    false_negatives = len(ground_truth_conditions - predicted_conditions)

    # Calculate F1 score components
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Calculate F1 score (harmonic mean of precision and recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1