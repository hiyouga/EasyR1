from collections import defaultdict
import numpy as np
import torch
from typing import Dict, List, Set, Tuple, Union


def parse_conditions(text: str) -> Set[str]:
    """
    Parse medical conditions from text, handling various separators.

    Args:
        text (str): Text containing medical conditions.

    Returns:
        Set[str]: Set of individual medical conditions.
    """
    # Remove any boxing notation if present
    text = text.replace("\\boxed{", "").replace("}", "")

    # Split by common separators
    for sep in [", ", " and ", " & ", ",", "&"]:
        if sep in text:
            return set(cond.strip() for cond in text.split(sep))

    # If no separator found, treat as single condition
    return {text.strip()}


def extract_boxed_content(text: str) -> str:
    """
    Extract content within \boxed{} or similar boxing notations.

    Args:
        text (str): Text containing potentially boxed content.

    Returns:
        str: Extracted boxed content or the original text if no box found.
    """
    import re

    # Look for LaTeX \boxed{} notation
    boxed_match = re.search(r'\\boxed{([^}]*)}', text)
    if boxed_match:
        return boxed_match.group(1)

    # Look for markdown boxed notation (e.g., [boxed content])
    markdown_match = re.search(r'\[(.*?)\]', text)
    if markdown_match:
        return markdown_match.group(1)

    # Return the text as is if no boxed content is found
    return text


def compute_classification_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """
    Compute classification metrics for medical diagnosis evaluation.

    Args:
        predictions (List[str]): List of model predictions.
        ground_truths (List[str]): List of ground truth labels.

    Returns:
        Dict[str, float]: Dictionary of metrics.
    """
    # Initialize counters for each condition
    all_conditions = set()
    condition_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "count": 0})

    # Global counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_conditions = 0

    # Process each prediction-ground truth pair
    for pred, gt in zip(predictions, ground_truths):
        pred_answer = extract_boxed_content(pred)
        if pred_answer == "None":
            pred_conditions = set()
        else:
            pred_conditions = parse_conditions(pred_answer)

        gt_conditions = parse_conditions(gt)

        # Update set of all conditions seen
        all_conditions.update(gt_conditions)
        all_conditions.update(pred_conditions)

        # For each ground truth condition
        for condition in gt_conditions:
            condition_metrics[condition]["count"] += 1
            if condition in pred_conditions:
                # True positive for this condition
                condition_metrics[condition]["tp"] += 1
                total_tp += 1
            else:
                # False negative for this condition
                condition_metrics[condition]["fn"] += 1
                total_fn += 1

        # For each predicted condition
        for condition in pred_conditions:
            if condition not in gt_conditions:
                # False positive for this condition
                condition_metrics[condition]["fp"] += 1
                total_fp += 1

        # Calculate true negatives (conditions that are correctly not predicted)
        other_conditions = all_conditions - (gt_conditions | pred_conditions)
        for condition in other_conditions:
            condition_metrics[condition]["tn"] += 1
            total_tn += 1

        total_conditions += len(gt_conditions)

    # Calculate metrics per condition
    # condition_results = {}
    # for condition, metrics in condition_metrics.items():
    #     if metrics["count"] == 0:
    #         continue  # Skip conditions that never appear in ground truth
    #
    #     tp = metrics["tp"]
    #     fp = metrics["fp"]
    #     fn = metrics["fn"]
    #     tn = metrics["tn"]
    #
    #     # Calculate metrics (avoid division by zero)
    #     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    #     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    #     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    #     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    #     accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    #
    #     # Store per-condition metrics
    #     condition_results[f"condition/{condition}/precision"] = precision
    #     condition_results[f"condition/{condition}/recall"] = recall
    #     condition_results[f"condition/{condition}/specificity"] = specificity
    #     condition_results[f"condition/{condition}/f1"] = f1
    #     condition_results[f"condition/{condition}/accuracy"] = accuracy

    # Calculate global metrics
    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    global_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (
                                                                                                         global_precision + global_recall) > 0 else 0
    global_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (
                                                                                                         total_tp + total_tn + total_fp + total_fn) > 0 else 0

    # Store global metrics
    metrics = {
        "precision": global_precision,
        "recall": global_recall,
        "sensitivity": global_recall,  # sensitivity is the same as recall
        "specificity": global_specificity,
        "f1": global_f1,
        "accuracy": global_accuracy,
    }

    # Combine global and condition-specific metrics
    # metrics.update(condition_results)

    return metrics


def medical_compute_score(predict_str: str, ground_truth: str) -> Tuple[float, Dict[str, int]]:
    """
    Compute F1 score for medical diagnosis and return detailed metrics.

    Args:
        predict_str (str): Model's prediction.
        ground_truth (str): Ground truth answer.

    Returns:
        Tuple[float, Dict[str, int]]: F1 score and detailed metrics.
    """
    answer = extract_boxed_content(predict_str)
    if answer == "None":
        return 0.0, {"tp": 0, "fp": 0, "fn": len(parse_conditions(ground_truth))}

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

    metrics = {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives
    }

    return f1, metrics


def compute_metrics_by_data_source(
        predictions: List[str],
        ground_truths: List[str],
        data_sources: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics grouped by data source.

    Args:
        predictions (List[str]): List of model predictions.
        ground_truths (List[str]): List of ground truth labels.
        data_sources (List[str]): List of data sources for each example.

    Returns:
        Dict[str, Dict[str, float]]: Metrics grouped by data source.
    """
    # Group examples by data source
    source_groups = defaultdict(lambda: {"preds": [], "gts": []})

    for pred, gt, source in zip(predictions, ground_truths, data_sources):
        source_groups[source]["preds"].append(pred)
        source_groups[source]["gts"].append(gt)

    # Compute metrics for each data source
    source_metrics = {}

    for source, data in source_groups.items():
        metrics = compute_classification_metrics(data["preds"], data["gts"])
        source_metrics[source] = metrics

    return source_metrics