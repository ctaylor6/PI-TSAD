"""Evaluation metrics for event-level anomaly detection."""

from __future__ import annotations


def evaluate_event_detection(
    ground_truth: list[tuple[float, float]],
    predictions: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """Compute one-to-one event precision, recall, and F1 from intervals."""
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt or pred_idx in matched_pred:
                continue
            if intervals_overlap(pred, gt):
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                break

    tp = len(matched_gt)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])
