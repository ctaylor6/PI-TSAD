"""Core PI-TSAD model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from pi_tsad.features import bandpass_filter, extract_features_with_times
from pi_tsad.thresholding import robust_threshold


@dataclass
class PITSADConfig:
    """Configuration for the default PI-TSAD random forest workflow."""

    window_radius: int = 15
    radius_multiplier: int = 2
    filter_signal: bool = True
    sample_rate: float = 200_000
    n_estimators: int = 50
    random_state: int = 0
    class_weight: str = "balanced"
    alpha: float = 0.06
    merge_gap: int = 0


@dataclass
class PredictionResult:
    centers: np.ndarray
    features: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    threshold: float | np.ndarray
    anomaly_indices: np.ndarray
    processed_signal: np.ndarray


class PITSADModel:
    """Train and apply the PI-TSAD classifier to TED signals."""

    def __init__(self, config: PITSADConfig | None = None) -> None:
        self.config = config or PITSADConfig()
        self.classifier = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
        )
        self.feature_names: list[str] | None = None

    def make_features(self, time: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
        processed_signal = (
            bandpass_filter(signal, sample_rate=self.config.sample_rate)
            if self.config.filter_signal
            else np.asarray(signal, dtype=float)
        )
        features, feature_names, centers = extract_features_with_times(
            time,
            processed_signal,
            window_radius=self.config.window_radius,
            sample_rate=self.config.sample_rate,
        )
        self.feature_names = feature_names
        return features, feature_names, centers, processed_signal

    def fit(self, feature_blocks: list[np.ndarray], label_blocks: list[np.ndarray]) -> "PITSADModel":
        features = np.vstack(feature_blocks)
        labels = np.concatenate(label_blocks)
        self.classifier.fit(features, labels)
        return self

    def predict_signal(self, time: np.ndarray, signal: np.ndarray) -> PredictionResult:
        features, _, centers, processed_signal = self.make_features(time, signal)
        probabilities = self.classifier.predict_proba(features)[:, 1]
        threshold, _, _, _, _, anomaly_indices = robust_threshold(
            probabilities,
            alpha=self.config.alpha,
            merge_gap=self.config.merge_gap,
            mode="empirical",
        )
        labels = (probabilities > threshold).astype(int)
        return PredictionResult(
            centers=centers,
            features=features,
            probabilities=probabilities,
            labels=labels,
            threshold=threshold,
            anomaly_indices=anomaly_indices,
            processed_signal=processed_signal,
        )

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "config": self.config,
                "classifier": self.classifier,
                "feature_names": self.feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PITSADModel":
        state = joblib.load(path)
        model = cls(state["config"])
        model.classifier = state["classifier"]
        model.feature_names = state.get("feature_names")
        return model
