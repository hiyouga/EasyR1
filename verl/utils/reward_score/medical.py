import re
import json
import torch
import numpy as np
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


def parse_json(json_output):
    """
    Parsing out the markdown fencing from JSON code blocks.
    """
    # Look for content between ```json and ```
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json" or line.strip() == "```":
            json_output = "\n".join(lines[i + 1:])  # Remove everything before ```json
            if "```" in json_output:
                json_output = json_output.split("```")[0]  # Remove everything after the closing ```
            break  # Exit the loop once code block marker is found
    return json_output


def extract_json_from_response(text):
    """
    Extract JSON content from markdown code blocks in the response.

    Args:
        text: The model's response text

    Returns:
        Parsed JSON object or None if no valid JSON found
    """
    # Find content between ```json and ```
    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_pattern, text)

    if not matches:
        return None

    # Try to parse each match as JSON
    for match in matches:
        try:
            parsed_json = json.loads(match.strip())
            return parsed_json
        except json.JSONDecodeError:
            continue

    # If we couldn't parse any match as valid JSON, try with ast.literal_eval
    import ast
    for match in matches:
        try:
            # Clean up the match a bit
            cleaned = match.strip().replace("'", "\"")
            parsed_json = ast.literal_eval(cleaned)
            return parsed_json
        except (SyntaxError, ValueError):
            continue

    return None


def bbox_to_mask(bbox, height, width):
    """
    Convert bounding box to binary mask.

    Args:
        bbox: Bounding box in format [x1, y1, x2, y2]
        height: Height of the mask
        width: Width of the mask

    Returns:
        Binary mask of shape (height, width)
    """
    mask = torch.zeros((height, width), dtype=torch.float32)

    # Ensure bbox coordinates are within image boundaries
    x1 = max(0, min(int(bbox[0]), width - 1))
    y1 = max(0, min(int(bbox[1]), height - 1))
    x2 = max(0, min(int(bbox[2]), width - 1))
    y2 = max(0, min(int(bbox[3]), height - 1))

    # Handle cases where x1>x2 or y1>y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # Set the box region to 1
    if x1 < x2 and y1 < y2:  # Ensure valid box dimensions
        mask[y1:y2 + 1, x1:x2 + 1] = 1.0

    return mask


def calculate_bbox_iou(pred_bboxes, seg_mask=None, gt_bbox=None):
    """
    Calculate IoU between predicted bounding boxes and ground truth (segmentation mask or bbox).

    Args:
        pred_bboxes: List of predicted bounding boxes in format [x1, y1, x2, y2]
        seg_mask: Ground truth segmentation mask tensor
        gt_bbox: Ground truth bounding box in format [x1, y1, x2, y2]

    Returns:
        Mean IoU score across all bounding boxes
    """
    if not pred_bboxes:
        return 0.0

    if seg_mask is not None:
        # Get mask dimensions
        if len(seg_mask.shape) == 3:  # Channel dimension
            height, width = seg_mask.shape[1], seg_mask.shape[2]
        else:
            height, width = seg_mask.shape[0], seg_mask.shape[1]

        # Convert segmentation mask to binary (1 for any positive value)
        binary_seg_mask = (seg_mask > 0).float()

        total_iou = 0.0
        for bbox in pred_bboxes:
            # Convert bbox to mask
            bbox_mask = bbox_to_mask(bbox, height, width)

            # Calculate intersection and union
            intersection = torch.sum(bbox_mask * binary_seg_mask)
            union = torch.sum(torch.clamp(bbox_mask + binary_seg_mask, 0, 1))

            # Calculate IoU
            iou = intersection / union if union > 0 else 0.0
            total_iou += iou

        # Return mean IoU
        return total_iou / len(pred_bboxes)

    elif gt_bbox is not None:
        # Calculate IoU directly between bounding boxes
        total_iou = 0.0
        for pred_bbox in pred_bboxes:
            # Calculate intersection
            x1 = max(pred_bbox[0], gt_bbox[0])
            y1 = max(pred_bbox[1], gt_bbox[1])
            x2 = min(pred_bbox[2], gt_bbox[2])
            y2 = min(pred_bbox[3], gt_bbox[3])

            # Check if boxes overlap
            if x1 >= x2 or y1 >= y2:
                iou = 0.0
            else:
                # Calculate areas
                intersection = (x2 - x1) * (y2 - y1)
                pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
                gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
                union = pred_area + gt_area - intersection

                # Calculate IoU
                iou = intersection / union if union > 0 else 0.0

            total_iou += iou

        # Return mean IoU
        return total_iou / len(pred_bboxes)

    else:
        # Neither segmentation mask nor ground truth bbox provided
        return 0.0


def medical_compute_score(predict_str: str, ground_truth: str, segmentation_mask=None, bbox=None) -> tuple:
    """
    Compute medical scoring including standard score and bounding box IoU.

    Args:
        predict_str: The model's prediction string
        ground_truth: The ground truth string
        segmentation_mask: Ground truth segmentation mask tensor
        bbox: Ground truth bounding box

    Returns:
        Tuple of (standard_score, bbox_score)
    """
    # Calculate standard score
    answer = extract_boxed_content(predict_str)
    if answer == "None":
        standard_score = 0.0  # no answer
    else:
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
        standard_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate bounding box score
    bbox_score = 0.0
    # Extract predicted bounding boxes from the response
    json_data = extract_json_from_response(predict_str)
    if json_data:
        # Extract bounding boxes from the JSON
        pred_bboxes = []
        for item in json_data:
            if isinstance(item, dict) and "bbox_2d" in item:
                pred_bboxes.append(item["bbox_2d"])

        # Calculate IoU between predicted boxes and ground truth
        bbox_score = calculate_bbox_iou(pred_bboxes, segmentation_mask, bbox)

    return standard_score, bbox_score