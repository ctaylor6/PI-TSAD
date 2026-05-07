"""Thresholding and interval utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def robust_threshold(
    scores: np.ndarray,
    *,
    alpha: float = 0.06,
    eps: float = 1e-6,
    merge_gap: int = 0,
    mode: str = "empirical",
    windowed: bool = False,
    window_size: int = 50_000,
) -> tuple[np.ndarray | float, float, float, float, str, np.ndarray]:
    """Compute a robust threshold for anomaly probabilities."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if not windowed:
        median = float(np.median(scores))
        mad = float(1.4826 * np.median(np.abs(scores - median)))
        scale = mad if mad > eps else eps
        z_scores = (scores - median) / (scale + eps)
        if mode == "parametric":
            z_threshold = float(norm.ppf(1 - alpha))
        elif mode == "empirical":
            z_threshold = float(np.quantile(z_scores, 1 - alpha))
        else:
            raise ValueError("mode must be 'parametric' or 'empirical'.")
        threshold = float(median + z_threshold * scale)
        thresholds = np.full_like(scores, threshold, dtype=float)
        method = "MAD"
    else:
        series = pd.Series(scores)
        baseline = series.rolling(window=window_size, center=True, min_periods=1).median().to_numpy()
        residual = scores - baseline
        if mode == "parametric":
            scale_arr = (
                pd.Series(residual)
                .rolling(window=window_size, center=True, min_periods=1)
                .std(ddof=0)
                .to_numpy()
            )
            scale_arr[scale_arr < eps] = eps
            z_threshold = float(norm.ppf(1 - alpha))
            thresholds = baseline + z_threshold * scale_arr
            threshold = thresholds
            median = float(np.median(baseline))
            scale = float(np.median(scale_arr))
            method = "ROLLING_STD"
        elif mode == "empirical":
            thresholds = series.rolling(
                window=window_size, center=True, min_periods=1
            ).quantile(1 - alpha).to_numpy()
            threshold = thresholds
            z_threshold = float("nan")
            median = float(np.median(baseline))
            scale = float(np.median(np.abs(residual)))
            method = "ROLLING_QUANTILE"
        else:
            raise ValueError("mode must be 'parametric' or 'empirical'.")

    raw_indices = np.where(scores > thresholds)[0]
    anomaly_indices = merge_anomaly_indices(raw_indices, scores, merge_gap=merge_gap)
    return threshold, z_threshold, median, scale, method, anomaly_indices


def merge_anomaly_indices(indices: np.ndarray, scores: np.ndarray, *, merge_gap: int = 0) -> np.ndarray:
    """Merge nearby anomaly indices and keep the highest-score representative."""
    indices = np.asarray(indices, dtype=int)
    if merge_gap <= 0 or len(indices) == 0:
        return indices

    indices = np.sort(indices)
    merged: list[int] = []
    cluster = [int(indices[0])]
    cluster_start = int(indices[0])
    max_span = 2 * int(merge_gap)
    for idx in indices[1:]:
        idx = int(idx)
        if idx - cluster_start > max_span:
            merged.append(int(cluster[np.argmax(scores[cluster])]))
            cluster = [idx]
            cluster_start = idx
        else:
            cluster.append(idx)
    merged.append(int(cluster[np.argmax(scores[cluster])]))
    return np.asarray(merged, dtype=int)


def event_intervals(event_times: list[float], *, window_size: float) -> list[tuple[float, float]]:
    half = window_size / 2
    return [(event_time - half, event_time + half) for event_time in event_times]


def predicted_intervals(
    centers: np.ndarray,
    pred_labels: np.ndarray,
    *,
    window_size: float,
) -> list[tuple[float, float]]:
    """Convert binary window predictions into contiguous time intervals."""
    intervals: list[tuple[float, float]] = []
    start = None
    half = window_size / 2
    for i, label in enumerate(pred_labels):
        if label == 1 and start is None:
            start = float(centers[i] - half)
        elif label == 0 and start is not None:
            intervals.append((start, float(centers[i - 1] + half)))
            start = None
    if start is not None:
        intervals.append((start, float(centers[-1] + half)))
    return intervals
