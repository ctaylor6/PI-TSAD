"""Command-line interface for PI-TSAD."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pi_tsad.data import load_part_csv
from pi_tsad.evaluation import cross_validate_aps, train_aps_model
from pi_tsad.model import PITSADConfig, PITSADModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PI-TSAD TED signal anomaly detection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cv = subparsers.add_parser("cross-validate-aps", help="Run 4-fold APS cross-validation.")
    add_common_model_args(cv)
    cv.add_argument("--data-dir", type=Path, default=None, help="Directory containing APS 1/2/3/4 CSVs.")
    cv.add_argument("--save-csv", type=Path, default=None, help="Optional path for metrics CSV output.")
    cv.add_argument("--plot-dir", type=Path, default=None, help="Optional directory for per-fold PNG plots.")
    cv.add_argument(
        "--summary-plot",
        type=Path,
        default=None,
        help="Optional path for one combined APS probability summary PNG.",
    )

    train = subparsers.add_parser("train-aps", help="Train on all four APS example datasets.")
    add_common_model_args(train)
    train.add_argument("--data-dir", type=Path, default=None)
    train.add_argument("--output", type=Path, default=Path("models/pi_tsad_aps.joblib"))

    detect = subparsers.add_parser("detect-part", help="Apply a trained model to a user-provided full part CSV.")
    detect.add_argument("model", type=Path)
    detect.add_argument("csv_file", type=Path)
    detect.add_argument("--layer", type=int, default=None, help="Optional layer to analyze.")
    detect.add_argument("--max-rows", type=int, default=None, help="Optional row limit for a quick run.")
    detect.add_argument("--output", type=Path, default=Path("outputs/part_anomalies.csv"))

    return parser


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-radius", type=int, default=15)
    parser.add_argument("--radius-multiplier", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.06)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--no-filter", action="store_true", help="Disable TED bandpass filtering.")


def config_from_args(args: argparse.Namespace) -> PITSADConfig:
    return PITSADConfig(
        window_radius=args.window_radius,
        radius_multiplier=args.radius_multiplier,
        alpha=args.alpha,
        n_estimators=args.n_estimators,
        filter_signal=not args.no_filter,
    )


def run_cross_validate(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    summary, artifacts = cross_validate_aps(data_dir=args.data_dir, config=config)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.save_csv, index=False)
    if args.plot_dir:
        from pi_tsad.visualization import plot_detection

        args.plot_dir.mkdir(parents=True, exist_ok=True)
        for artifact in artifacts:
            plot_detection(
                time=artifact.time,
                signal=artifact.prediction.processed_signal,
                centers=artifact.prediction.centers,
                probabilities=artifact.prediction.probabilities,
                ground_truth_intervals=artifact.ground_truth_intervals,
                predicted_intervals=artifact.predicted_intervals,
                threshold=float(artifact.prediction.threshold),
                title=f"APS {artifact.fold.test_key} leave-one-out",
                output_path=args.plot_dir / f"aps_{artifact.fold.test_key}.png",
            )
    if args.summary_plot:
        from pi_tsad.visualization import plot_cross_validation_probabilities

        args.summary_plot.parent.mkdir(parents=True, exist_ok=True)
        plot_cross_validation_probabilities(artifacts, output_path=args.summary_plot)
    return 0


def run_train_aps(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    model = train_aps_model(data_dir=args.data_dir, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved APS-trained PI-TSAD model to {args.output}")
    return 0


def run_detect_part(args: argparse.Namespace) -> int:
    model = PITSADModel.load(args.model)
    df = load_part_csv(args.csv_file)
    if args.layer is not None:
        df = df.query("layer == @args.layer").copy()
    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].copy()
    if df.empty:
        raise ValueError("No rows available after layer/max-rows filtering.")

    time = df.index.to_numpy(dtype=float)
    result = model.predict_signal(time, df["TED"].to_numpy(dtype=float))
    output = pd.DataFrame(
        {
            "center_index": result.centers,
            "anomaly_probability": result.probabilities,
            "anomaly_label": result.labels,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output):,} window predictions to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "cross-validate-aps":
        return run_cross_validate(args)
    if args.command == "train-aps":
        return run_train_aps(args)
    if args.command == "detect-part":
        return run_detect_part(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
