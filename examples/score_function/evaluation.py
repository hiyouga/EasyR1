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


def compute_class_metrics(class_name: str, confusion_matrix: Dict[str, int]) -> Dict[str, float]:
    """
    Compute metrics for a single class based on its confusion matrix.

    Args:
        class_name (str): Name of the class.
        confusion_matrix (Dict[str, int]): Confusion matrix with tp, fp, fn, tn.

    Returns:
        Dict[str, float]: Dictionary of metrics for this class.
    """
    tp = confusion_matrix["tp"]
    fp = confusion_matrix["fp"]
    fn = confusion_matrix["fn"]
    tn = confusion_matrix["tn"]

    # Calculate metrics (avoid division by zero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivity = recall  # sensitivity is the same as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "count": confusion_matrix["count"],
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
    }


def compute_confusion_matrices(predictions: List[str], ground_truths: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Compute confusion matrices for each class.

    Args:
        predictions (List[str]): List of model predictions.
        ground_truths (List[str]): List of ground truth labels.

    Returns:
        Dict[str, Dict[str, int]]: Confusion matrices for each class.
    """
    # Initialize counters for each condition
    all_conditions = set()
    condition_matrices = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "count": 0})

    # First pass: identify all unique conditions
    for gt in ground_truths:
        gt_conditions = parse_conditions(gt)
        all_conditions.update(gt_conditions)

    for pred in predictions:
        pred_answer = extract_boxed_content(pred)
        if pred_answer != "None":
            pred_conditions = parse_conditions(pred_answer)
            all_conditions.update(pred_conditions)

    # Second pass: compute confusion matrices
    for pred, gt in zip(predictions, ground_truths):
        pred_answer = extract_boxed_content(pred)
        if pred_answer == "None":
            pred_conditions = set()
        else:
            pred_conditions = parse_conditions(pred_answer)

        gt_conditions = parse_conditions(gt)

        # For each possible condition
        for condition in all_conditions:
            condition_present_in_gt = condition in gt_conditions
            condition_present_in_pred = condition in pred_conditions

            if condition_present_in_gt:
                condition_matrices[condition]["count"] += 1

            if condition_present_in_gt and condition_present_in_pred:
                # True positive
                condition_matrices[condition]["tp"] += 1
            elif condition_present_in_gt and not condition_present_in_pred:
                # False negative
                condition_matrices[condition]["fn"] += 1
            elif not condition_present_in_gt and condition_present_in_pred:
                # False positive
                condition_matrices[condition]["fp"] += 1
            else:
                # True negative
                condition_matrices[condition]["tn"] += 1

    return condition_matrices


def compute_dataset_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, Dict]:
    """
    Compute metrics for a single dataset, with class-wise averaging.

    Args:
        predictions (List[str]): List of model predictions for this dataset.
        ground_truths (List[str]): List of ground truth labels for this dataset.

    Returns:
        Dict[str, Dict]: Class metrics and averaged dataset metrics.
    """
    # Compute confusion matrices for each class
    class_matrices = compute_confusion_matrices(predictions, ground_truths)

    # Compute metrics for each class
    class_metrics = {}
    active_classes = 0

    # Accumulators for dataset-level metrics
    dataset_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "f1": 0.0,
        "accuracy": 0.0
    }

    # Compute metrics for each class and accumulate for dataset average
    for class_name, matrix in class_matrices.items():
        # Skip classes that never appear in ground truth
        if matrix["count"] == 0:
            continue

        active_classes += 1
        metrics = compute_class_metrics(class_name, matrix)
        class_metrics[class_name] = metrics

        # Accumulate for dataset average (equal class weighting)
        for metric_name in dataset_metrics.keys():
            dataset_metrics[metric_name] += metrics[metric_name]

    # Calculate dataset average (equal class weighting)
    if active_classes > 0:
        for metric_name in dataset_metrics.keys():
            dataset_metrics[metric_name] /= active_classes

    # Add class metrics to the result
    result = {
        "class_metrics": class_metrics,
        "dataset_metrics": dataset_metrics,
        "active_classes": active_classes
    }

    return result


def compute_metrics_by_data_source(
        predictions: List[str],
        ground_truths: List[str],
        data_sources: List[str],
        datasets: List[str]
) -> Dict[str, float]:
    """
    Compute hierarchical metrics: class -> dataset -> data source -> global.

    Args:
        predictions (List[str]): List of model predictions.
        ground_truths (List[str]): List of ground truth labels.
        data_sources (List[str]): List of data sources for each example.
        datasets (List[str]): List of dataset identifiers for each example.

    Returns:
        Dict[str, float]: Flattened dictionary of metrics at all levels with keys:
            - "val/{metric}" for global metrics
            - "{data_source}/{metric}" for data source metrics
            - "{data_source}/{dataset}/{metric}" for dataset metrics
    """
    # Group examples by data source and dataset
    grouped_data = defaultdict(lambda: defaultdict(lambda: {"preds": [], "gts": []}))

    for pred, gt, source, dataset in zip(predictions, ground_truths, data_sources, datasets):
        grouped_data[source][dataset]["preds"].append(pred)
        grouped_data[source][dataset]["gts"].append(gt)

    # Initialize the flattened result dictionary
    result = {}

    # Initialize global metrics accumulators
    global_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "f1": 0.0,
        "accuracy": 0.0
    }

    # Compute metrics for each dataset within each data source
    total_data_sources = 0

    for source_name, source_datasets in grouped_data.items():
        # Initialize metrics accumulators for this data source
        source_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "accuracy": 0.0
        }

        total_datasets_in_source = 0

        for dataset_name, dataset_data in source_datasets.items():
            # Compute metrics for this dataset
            dataset_result = compute_dataset_metrics(
                dataset_data["preds"],
                dataset_data["gts"]
            )

            # Store dataset-level metrics with the format "data_source/dataset/metric"
            for metric_name, metric_value in dataset_result["dataset_metrics"].items():
                result[f"{source_name}/{dataset_name}/{metric_name}"] = metric_value

            # Skip empty datasets
            if dataset_result["active_classes"] == 0:
                continue

            total_datasets_in_source += 1

            # Accumulate metrics for data source average (equal dataset weighting)
            for metric_name in source_metrics.keys():
                source_metrics[metric_name] += dataset_result["dataset_metrics"][metric_name]

        # Calculate data source average (equal dataset weighting)
        if total_datasets_in_source > 0:
            for metric_name in source_metrics.keys():
                source_metrics[metric_name] /= total_datasets_in_source

            # Store data source metrics with the format "data_source/metric"
            for metric_name, metric_value in source_metrics.items():
                result[f"{source_name}/{metric_name}"] = metric_value

            total_data_sources += 1

            # Accumulate for global metrics (equal data source weighting)
            for metric_name in global_metrics.keys():
                global_metrics[metric_name] += source_metrics[metric_name]

    # Calculate global average (equal data source weighting)
    if total_data_sources > 0:
        for metric_name in global_metrics.keys():
            global_metrics[metric_name] /= total_data_sources

        # Store global metrics with the format "val/metric"
        for metric_name, metric_value in global_metrics.items():
            result[f"val/{metric_name}"] = metric_value

    return result


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