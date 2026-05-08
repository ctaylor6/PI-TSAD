"""Create an animated PI-TSAD detection GIF for APS sample 3.

The animation trains PI-TSAD on APS samples 1, 2, and 4, tests on sample 3,
then sweeps through time showing the filtered TED signal and the anomaly
probability as the detector moves across the track.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi_tsad.evaluation import train_leave_one_out_fold
from pi_tsad.model import PITSADConfig


def create_aps_sample_3_animation(
    output_path: str | Path = "docs/assets/aps_sample_3_detection_process.gif",
    *,
    fps: int = 12,
    n_frames: int = 120,
) -> Path:
    """Build and save the APS sample 3 leave-one-out animation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = PITSADConfig(window_radius=15, radius_multiplier=2, alpha=0.06)
    artifact = train_leave_one_out_fold("3", config=config)
    prediction = artifact.prediction

    time_ms = artifact.time * 1e3
    centers_ms = prediction.centers * 1e3
    gt_intervals_ms = [(start * 1e3, end * 1e3) for start, end in artifact.ground_truth_intervals]
    pred_intervals_ms = [(start * 1e3, end * 1e3) for start, end in artifact.predicted_intervals]
    threshold = float(np.asarray(prediction.threshold).mean())

    frame_indices = np.linspace(1, len(time_ms) - 1, n_frames).astype(int)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )

    fig, (signal_ax, prob_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("white")

    signal_ax.plot(time_ms, prediction.processed_signal, color="#d7dee6", linewidth=1.0)
    signal_line, = signal_ax.plot([], [], color="#174a7c", linewidth=2.0, label="Filtered TED")
    cursor_signal = signal_ax.axvline(time_ms[0], color="#333333", linewidth=1.0)

    for i, (start, end) in enumerate(gt_intervals_ms):
        signal_ax.axvspan(
            start,
            end,
            color="#d62728",
            alpha=0.10,
            label="X-ray verified collapse" if i == 0 else None,
        )

    signal_ax.set_title("PI-TSAD leave-one-out detection on APS sample 3")
    signal_ax.set_ylabel("Filtered TED")
    signal_ax.grid(True, alpha=0.25)
    signal_ax.legend(loc="upper right")

    prob_ax.plot(centers_ms, prediction.probabilities, color="#ded3ef", linewidth=1.0)
    prob_line, = prob_ax.plot([], [], color="#5e3c99", linewidth=2.0, label="Anomaly probability")
    for i, (start, end) in enumerate(pred_intervals_ms):
        prob_ax.axvspan(
            start,
            end,
            color="#2ca02c",
            alpha=0.12,
            label="PI-TSAD detection" if i == 0 else None,
        )
    threshold_line = prob_ax.axhline(
        threshold,
        color="#c44e52",
        linestyle="--",
        linewidth=1.2,
        label=f"Empirical threshold = {threshold:.3f}",
    )
    cursor_prob = prob_ax.axvline(time_ms[0], color="#333333", linewidth=1.0)

    prob_ax.set_xlabel("Time (ms)")
    prob_ax.set_ylabel("Probability")
    prob_ax.set_ylim(0, max(1.0, float(np.max(prediction.probabilities)) * 1.15))
    prob_ax.grid(True, alpha=0.25)
    prob_ax.legend(loc="upper right")

    signal_ax.set_xlim(time_ms[0], time_ms[-1])
    y_margin = 0.08 * (float(np.max(prediction.processed_signal)) - float(np.min(prediction.processed_signal)))
    signal_ax.set_ylim(
        float(np.min(prediction.processed_signal)) - y_margin,
        float(np.max(prediction.processed_signal)) + y_margin,
    )

    def update(frame_number: int):
        idx = frame_indices[frame_number]
        current_time = time_ms[idx]
        signal_line.set_data(time_ms[: idx + 1], prediction.processed_signal[: idx + 1])
        cursor_signal.set_xdata([current_time, current_time])
        cursor_prob.set_xdata([current_time, current_time])

        prob_idx = int(np.searchsorted(centers_ms, current_time, side="right"))
        prob_line.set_data(centers_ms[:prob_idx], prediction.probabilities[:prob_idx])

        return signal_line, prob_line, cursor_signal, cursor_prob, threshold_line

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / fps,
        blit=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return output_path


def main() -> None:
    output_path = create_aps_sample_3_animation()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
