"""Plotting helpers for PI-TSAD results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_detection(
    *,
    time: np.ndarray,
    signal: np.ndarray,
    centers: np.ndarray,
    probabilities: np.ndarray,
    ground_truth_intervals: list[tuple[float, float]] | None = None,
    predicted_intervals: list[tuple[float, float]] | None = None,
    threshold: float | None = None,
    title: str = "PI-TSAD Detection",
    output_path: str | Path | None = None,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(time, signal, color="#1f4e79", linewidth=1.5, label="TED")
    for i, interval in enumerate(ground_truth_intervals or []):
        axes[0].axvspan(*interval, color="#c0392b", alpha=0.16, label="Ground truth" if i == 0 else None)
    for i, interval in enumerate(predicted_intervals or []):
        axes[0].axvspan(*interval, color="#27ae60", alpha=0.22, label="Prediction" if i == 0 else None)
    axes[0].set_title(title)
    axes[0].set_ylabel("TED")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(centers, probabilities, color="#5e3c99", linewidth=1.4, label="Anomaly probability")
    if threshold is not None:
        axes[1].axhline(threshold, color="#c0392b", linestyle="--", linewidth=1.0, label="Threshold")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_cross_validation_probabilities(
    artifacts,
    *,
    output_path: str | Path | None = None,
    title: str = "PI-TSAD APS Cross-Validation Probabilities",
) -> None:
    """Plot all APS fold signals and anomaly probabilities in one figure."""
    n_folds = len(artifacts)
    if n_folds == 0:
        raise ValueError("At least one fold artifact is required.")

    fig, axes = plt.subplots(n_folds, 2, figsize=(13, 2.7 * n_folds), sharex=False)
    if n_folds == 1:
        axes = np.asarray([axes])

    for row, artifact in enumerate(artifacts):
        signal_ax, prob_ax = axes[row]
        fold = artifact.fold
        prediction = artifact.prediction

        signal_ax.plot(artifact.time, prediction.processed_signal, color="#1f4e79", linewidth=1.2)
        for i, interval in enumerate(artifact.ground_truth_intervals):
            signal_ax.axvspan(
                *interval,
                color="#c0392b",
                alpha=0.16,
                label="Ground truth" if i == 0 else None,
            )
        for i, interval in enumerate(artifact.predicted_intervals):
            signal_ax.axvspan(
                *interval,
                color="#27ae60",
                alpha=0.22,
                label="Prediction" if i == 0 else None,
            )
        signal_ax.set_title(f"APS {fold.test_key} TED")
        signal_ax.set_ylabel("TED")
        signal_ax.grid(True, alpha=0.3)
        if row == 0:
            signal_ax.legend(loc="best")

        threshold = float(np.asarray(prediction.threshold).mean())
        prob_ax.plot(
            prediction.centers,
            prediction.probabilities,
            color="#5e3c99",
            linewidth=1.2,
        )
        prob_ax.axhline(threshold, color="#c0392b", linestyle="--", linewidth=1.0)
        prob_ax.set_title(f"Probability, F1={fold.f1:.3f}, threshold={threshold:.3f}")
        prob_ax.set_ylabel("P(anomaly)")
        prob_ax.set_ylim(bottom=0)
        prob_ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_probability_histogram(
    probabilities: np.ndarray,
    *,
    cutoff: float,
    title: str = "PI-TSAD Probability Histogram",
    output_path: str | Path | None = None,
    bins: int = 60,
) -> None:
    """Plot nominal/anomalous probability histograms with a cutoff marker."""
    probabilities = np.asarray(probabilities, dtype=float)
    nominal = probabilities[probabilities <= cutoff]
    anomalous = probabilities[probabilities > cutoff]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    hist_range = (0.0, max(1.0, float(np.nanmax(probabilities)) if probabilities.size else 1.0))
    ax.hist(
        nominal,
        bins=bins,
        range=hist_range,
        color="#4C78A8",
        alpha=0.72,
        label=f"Nominal (n={len(nominal):,})",
    )
    ax.hist(
        anomalous,
        bins=bins,
        range=hist_range,
        color="#C44E52",
        alpha=0.72,
        label=f"Anomalous (n={len(anomalous):,})",
    )
    ax.axvline(cutoff, color="#111111", linestyle="--", linewidth=1.5, label=f"Cutoff = {cutoff:.4g}")
    ax.set_title(title)
    ax.set_xlabel("Predicted anomaly probability")
    ax.set_ylabel("Window count")
    ax.set_xlim(hist_range)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
