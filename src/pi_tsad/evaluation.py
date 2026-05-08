"""Cross-validation workflows for APS examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pi_tsad.constants import APS_COLLAPSE_TIMES, APS_EXPERIMENT_KEYS
from pi_tsad.data import aps_csv_files, load_ted_signal
from pi_tsad.features import label_windows_by_times
from pi_tsad.metrics import evaluate_event_detection
from pi_tsad.model import PITSADConfig, PITSADModel, PredictionResult
from pi_tsad.thresholding import event_intervals, predicted_intervals


@dataclass
class FoldResult:
    test_key: str
    precision: float
    recall: float
    f1: float
    threshold: float
    n_predicted_events: int
    n_ground_truth_events: int
    n_predicted_windows: int


@dataclass
class FoldArtifacts:
    fold: FoldResult
    model: PITSADModel
    time: np.ndarray
    signal: np.ndarray
    prediction: PredictionResult
    ground_truth_intervals: list[tuple[float, float]]
    predicted_intervals: list[tuple[float, float]]


def train_leave_one_out_fold(
    test_key: str,
    *,
    data_dir: str | Path | None = None,
    config: PITSADConfig | None = None,
) -> FoldArtifacts:
    config = config or PITSADConfig()
    files = aps_csv_files(data_dir)
    if test_key not in files:
        raise ValueError(f"Unknown APS test key {test_key!r}. Expected one of {APS_EXPERIMENT_KEYS}.")

    model = PITSADModel(config)
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    test_time = test_signal = None

    for key, path in files.items():
        time, signal = load_ted_signal(path)
        features, _, centers, _ = model.make_features(time, signal)
        sampling_interval = float(np.mean(np.diff(time)))
        window_size = (config.radius_multiplier * config.window_radius + 1) * sampling_interval
        labels = label_windows_by_times(
            centers,
            APS_COLLAPSE_TIMES[key],
            window_size=window_size,
        )
        if key == test_key:
            test_time, test_signal = time, signal
        else:
            feature_blocks.append(features)
            label_blocks.append(labels)

    model.fit(feature_blocks, label_blocks)
    assert test_time is not None and test_signal is not None
    prediction = model.predict_signal(test_time, test_signal)
    sampling_interval = float(np.mean(np.diff(test_time)))
    window_size = (config.radius_multiplier * config.window_radius + 1) * sampling_interval
    gt_intervals = event_intervals(APS_COLLAPSE_TIMES[test_key], window_size=window_size)
    pred_intervals = predicted_intervals(prediction.centers, prediction.labels, window_size=window_size)
    precision, recall, f1 = evaluate_event_detection(gt_intervals, pred_intervals)

    threshold = float(np.asarray(prediction.threshold).mean())
    fold = FoldResult(
        test_key=test_key,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=threshold,
        n_predicted_events=len(pred_intervals),
        n_ground_truth_events=len(gt_intervals),
        n_predicted_windows=int(prediction.labels.sum()),
    )
    return FoldArtifacts(
        fold=fold,
        model=model,
        time=test_time,
        signal=test_signal,
        prediction=prediction,
        ground_truth_intervals=gt_intervals,
        predicted_intervals=pred_intervals,
    )


def cross_validate_aps(
    *,
    data_dir: str | Path | None = None,
    config: PITSADConfig | None = None,
) -> tuple[pd.DataFrame, list[FoldArtifacts]]:
    """Run four-fold leave-one-dataset-out validation on APS samples 1, 2, 3, and 4."""
    config = config or PITSADConfig()
    artifacts = [
        train_leave_one_out_fold(key, data_dir=data_dir, config=config)
        for key in APS_EXPERIMENT_KEYS
    ]
    rows = [asdict(artifact.fold) for artifact in artifacts]
    summary = pd.DataFrame(rows)
    mean_row = {
        "test_key": "mean",
        "precision": summary["precision"].mean(),
        "recall": summary["recall"].mean(),
        "f1": summary["f1"].mean(),
        "threshold": summary["threshold"].mean(),
        "n_predicted_events": summary["n_predicted_events"].mean(),
        "n_ground_truth_events": summary["n_ground_truth_events"].mean(),
        "n_predicted_windows": summary["n_predicted_windows"].mean(),
    }
    return pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True), artifacts


def train_aps_model(
    *,
    data_dir: str | Path | None = None,
    config: PITSADConfig | None = None,
) -> PITSADModel:
    """Train one PI-TSAD model on all four bundled APS example datasets."""
    config = config or PITSADConfig()
    model = PITSADModel(config)
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []

    for key, path in aps_csv_files(data_dir).items():
        time, signal = load_ted_signal(path)
        features, _, centers, _ = model.make_features(time, signal)
        sampling_interval = float(np.mean(np.diff(time)))
        window_size = (config.radius_multiplier * config.window_radius + 1) * sampling_interval
        labels = label_windows_by_times(
            centers,
            APS_COLLAPSE_TIMES[key],
            window_size=window_size,
        )
        feature_blocks.append(features)
        label_blocks.append(labels)

    return model.fit(feature_blocks, label_blocks)
