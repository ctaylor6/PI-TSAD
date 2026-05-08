"""Plotting helpers for PI-TSAD results."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import gaussian_kde


class ProbabilityDistribution(NamedTuple):
    """Probability distribution data for one part/file."""

    label: str
    probabilities: np.ndarray
    cutoff: float


PROBABILITY_DENSITY_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "axes.labelsize": 14,
    "axes.titlesize": 12,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
    "axes.edgecolor": "black",
    "axes.labelpad": 5,
    "grid.alpha": 0.35,
    "figure.dpi": 600,
    "savefig.dpi": 600,
}


def _clean_decimal(x: float, _pos: int) -> str:
    if float(x).is_integer():
        return f"{int(x)}"
    return f"{x:.1f}"


def _probability_density(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities[np.isfinite(probabilities)]
    x = np.linspace(0, 1, 500)
    if probabilities.size < 2 or np.allclose(probabilities, probabilities[0]):
        density = np.zeros_like(x)
        if probabilities.size:
            density[np.argmin(np.abs(x - probabilities[0]))] = 1.0
        return x, density
    kde = gaussian_kde(np.clip(probabilities, 0, 1))
    return x, kde(x)


def _style_probability_density_axis(
    ax,
    *,
    index: int,
    label: str,
    cutoff: float,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    cutoff = float(np.clip(cutoff, 0, 1))
    ax.axvspan(0, cutoff, color="green", alpha=0.12, zorder=0)
    ax.axvspan(cutoff, 1, color="red", alpha=0.12, zorder=0)
    ax.axvline(cutoff, color="red", ls="--", lw=1.1, label="Threshold")

    y_text = 0.85 * ax.get_ylim()[1]
    nominal_x = cutoff / 2 if cutoff > 0 else 0.08
    anomalous_x = cutoff + (1 - cutoff) / 2 if cutoff < 1 else 0.92
    ax.text(
        nominal_x,
        y_text,
        "Nominal\nRegion",
        color="green",
        fontsize=16,
        ha="center",
        va="center",
        fontstyle="italic",
    )
    ax.text(
        anomalous_x,
        y_text,
        "Anomalous\nRegion",
        color="red",
        fontsize=16,
        ha="center",
        va="center",
        fontstyle="italic",
    )

    ax.set_title(f"({chr(97 + index)}) {label}", loc="left", pad=5, fontsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, ls="--", lw=0.4, alpha=0.55)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_clean_decimal))
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="x", which="minor", length=3, width=0.8)

    if not show_ylabel:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    if not show_xlabel:
        ax.set_xlabel("")
        ax.set_xticklabels([])


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
    """Plot a publication-style probability density with the PI-TSAD cutoff."""
    del bins
    distribution = ProbabilityDistribution("Part", probabilities, cutoff)
    plot_probability_distribution_grid(
        [distribution],
        output_path=output_path,
        title=title,
        columns=1,
        shared_labels=False,
    )


def plot_probability_distribution_grid(
    distributions: list[ProbabilityDistribution],
    *,
    output_path: str | Path | None = None,
    title: str = "Distribution of Predicted Anomaly Probabilities Across Parts",
    columns: int = 4,
    shared_labels: bool = True,
) -> None:
    """Plot KDE probability distributions for one or more full-scale parts."""
    if not distributions:
        raise ValueError("At least one probability distribution is required.")

    n_parts = len(distributions)
    columns = max(1, min(columns, n_parts))
    rows = int(np.ceil(n_parts / columns))
    figsize = (13, 3 * rows) if columns > 1 else (6.5, 4.8)

    with plt.rc_context(PROBABILITY_DENSITY_STYLE):
        fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
        flat_axes = axes.flatten()

        for i, distribution in enumerate(distributions):
            ax = flat_axes[i]
            x, density = _probability_density(distribution.probabilities)
            ax.fill_between(x, density, color="slateblue", alpha=0.35, label="KDE")
            ax.plot(x, density, color="slateblue", lw=1.3)
            show_xlabel = rows == 1 or i >= columns * (rows - 1)
            show_ylabel = i % columns == 0
            _style_probability_density_axis(
                ax,
                index=i,
                label=distribution.label,
                cutoff=distribution.cutoff,
                show_xlabel=show_xlabel if shared_labels else True,
                show_ylabel=show_ylabel if shared_labels else True,
            )

        for ax in flat_axes[n_parts:]:
            ax.axis("off")

        if shared_labels:
            fig.text(0.53, 0.01, "Predicted Probability", ha="center", fontsize=20)
            fig.text(
                0.025,
                0.5,
                "Probability Density",
                va="center",
                rotation="vertical",
                fontsize=20,
            )
            fig.suptitle(title, fontsize=24, y=1.025)
            fig.subplots_adjust(
                left=0.08,
                right=0.97,
                top=0.92,
                bottom=0.1,
                wspace=0.10,
                hspace=0.15,
            )
        else:
            flat_axes[0].set_xlabel("Predicted Probability")
            flat_axes[0].set_ylabel("Probability Density")
            fig.suptitle(title, fontsize=18, y=0.98)
            fig.tight_layout()

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
