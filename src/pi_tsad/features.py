"""Signal processing and window feature extraction."""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt


def bandpass_filter(
    signal: np.ndarray,
    *,
    sample_rate: float = 200_000,
    lowcut: float = 250,
    highcut: float = 30_000,
    order: int = 4,
) -> np.ndarray:
    """Apply the bandpass filter used by the APS notebook workflow."""
    signal = np.asarray(signal, dtype=float)
    b, a = butter(order, [lowcut / (sample_rate / 2), highcut / (sample_rate / 2)], btype="band")
    return filtfilt(b, a, signal)


def extract_features_with_times(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    window_radius: int = 15,
    sample_rate: float = 200_000,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Extract non-overlapping TED window features and center times."""
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    window_length = 2 * window_radius + 1
    if len(signal) < window_length:
        raise ValueError("Signal is shorter than one feature window.")

    rows: list[list[float]] = []
    centers: list[float] = []
    feature_names = [
        "std",
        "auc",
        "peak_to_peak",
        "mean_post",
        "energy_6p2_12p5khz",
        "energy_0_6p2khz",
    ]

    high_band = (6_200, 12_500)
    low_band = (0, 6_200)

    for start in range(0, len(signal) - window_length + 1, window_length):
        end = start + window_length
        w = signal[start:end]
        t_w = time[start:end]
        center_idx = start + window_radius
        centers.append(time[center_idx])

        w_post = w[window_radius + 1 :]
        freqs = rfftfreq(len(w), d=1 / sample_rate)
        fft_power = np.abs(rfft(w)) ** 2
        high_energy = np.sum(fft_power[(freqs >= high_band[0]) & (freqs <= high_band[1])])
        low_energy = np.sum(fft_power[(freqs >= low_band[0]) & (freqs <= low_band[1])])

        rows.append(
            [
                float(np.std(w)),
                float(np.trapezoid(w, t_w)),
                float(np.ptp(w)),
                float(np.mean(w_post)),
                float(high_energy),
                float(low_energy),
            ]
        )

    return np.asarray(rows, dtype=float), feature_names, np.asarray(centers, dtype=float)


def label_windows_by_times(
    centers: np.ndarray,
    event_times: list[float],
    *,
    window_size: float,
) -> np.ndarray:
    """Label each window center as anomalous when it falls within an event window."""
    half = window_size / 2
    labels = np.zeros(len(centers), dtype=int)
    for i, center in enumerate(centers):
        labels[i] = int(any(abs(center - event_time) <= half for event_time in event_times))
    return labels
