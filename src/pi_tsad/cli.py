"""Command-line interface for PI-TSAD."""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from pi_tsad.data import load_part_csv
from pi_tsad.evaluation import cross_validate_aps, train_aps_model
from pi_tsad.model import PITSADConfig, PITSADModel
from pi_tsad.thresholding import robust_threshold


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
    detect.add_argument("--cutoff", type=float, default=None, help="Optional fixed probability cutoff.")
    detect.add_argument("--histogram", type=Path, default=None, help="Optional histogram PNG output.")

    batch = subparsers.add_parser(
        "detect-batch",
        help="Apply a trained model to one or more full part CSVs and save reusable probabilities.",
    )
    batch.add_argument("model", type=Path)
    batch.add_argument("csv_files", nargs="+", help="CSV paths or quoted glob patterns, e.g. 'part-scale/*.csv'.")
    batch.add_argument("--output-dir", type=Path, default=Path("outputs/part-scale"))
    batch.add_argument(
        "--hist-dir",
        type=Path,
        default=None,
        help="Optional legacy root for histograms. By default, histograms are saved inside each part folder.",
    )
    batch.add_argument("--layer", type=int, default=None, help="Optional layer to analyze for every CSV.")
    batch.add_argument("--max-rows", type=int, default=None, help="Optional row limit for a quick run.")
    batch.add_argument("--cutoff", type=float, default=None, help="Optional fixed probability cutoff.")
    batch.add_argument("--alpha", type=float, default=0.06, help="Empirical cutoff alpha when --cutoff is omitted.")
    batch.add_argument(
        "--summary-plot",
        type=Path,
        default=None,
        help="Optional combined probability-density plot for all processed CSVs.",
    )

    threshold = subparsers.add_parser(
        "threshold-probabilities",
        help="Re-threshold saved probability CSVs and remake histograms without rerunning model inference.",
    )
    threshold.add_argument("probability_csvs", nargs="+", help="Probability CSV paths or quoted glob patterns.")
    threshold.add_argument("--cutoff", type=float, default=None, help="Fixed probability cutoff to apply.")
    threshold.add_argument("--alpha", type=float, default=0.06, help="Empirical cutoff alpha when --cutoff is omitted.")
    threshold.add_argument("--output-dir", type=Path, default=Path("outputs/rethresholded"))
    threshold.add_argument(
        "--hist-dir",
        type=Path,
        default=None,
        help="Optional legacy root for histograms. By default, histograms are saved inside each part folder.",
    )
    threshold.add_argument(
        "--summary-plot",
        type=Path,
        default=None,
        help="Optional combined probability-density plot after applying the selected cutoff.",
    )

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
    result = model.predict_signal(time, df["TED"].to_numpy(dtype=float), scale_features=False)
    cutoff = args.cutoff if args.cutoff is not None else float(np.asarray(result.threshold).mean())
    labels = (result.probabilities > cutoff).astype(int)
    output = pd.DataFrame(
        {
            "center_index": result.centers,
            "anomaly_probability": result.probabilities,
            "anomaly_label": labels,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    if args.histogram:
        from pi_tsad.visualization import plot_probability_histogram

        plot_probability_histogram(
            result.probabilities,
            cutoff=cutoff,
            title=f"PI-TSAD probabilities: {args.csv_file.stem}",
            output_path=args.histogram,
        )
    print(f"Saved {len(output):,} window predictions to {args.output}")
    print(f"Cutoff: {cutoff:.6g}; anomalous windows: {int(labels.sum()):,}")
    return 0


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)
    return unique_paths


def cutoff_from_probabilities(probabilities: np.ndarray, *, cutoff: float | None, alpha: float) -> float:
    if cutoff is not None:
        return float(cutoff)
    threshold, *_ = robust_threshold(probabilities, alpha=alpha, mode="empirical")
    return float(np.asarray(threshold).mean())


def part_label(index: int, path: Path) -> str:
    numeric_suffix = path.stem.rsplit("_", maxsplit=1)[-1]
    if numeric_suffix.isdigit():
        return f"Part {int(numeric_suffix)}"
    return f"Part {index + 1}"


def part_name_from_probability_csv(path: Path) -> str:
    if path.name == "probabilities.csv":
        return path.parent.name
    return path.stem.removesuffix("_probabilities")


def run_detect_batch(args: argparse.Namespace) -> int:
    model = PITSADModel.load(args.model)
    csv_files = expand_paths(args.csv_files)
    missing = [path for path in csv_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV file(s): {', '.join(str(path) for path in missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.hist_dir:
        args.hist_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    distributions = []
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        part_dir = args.output_dir / csv_file.stem
        part_dir.mkdir(parents=True, exist_ok=True)
        df = load_part_csv(csv_file)
        if args.layer is not None:
            df = df.query("layer == @args.layer").copy()
        if args.max_rows is not None:
            df = df.iloc[: args.max_rows].copy()
        if df.empty:
            raise ValueError(f"No rows available after filtering {csv_file}.")

        time = df.index.to_numpy(dtype=float)
        result = model.predict_signal(time, df["TED"].to_numpy(dtype=float), scale_features=False)
        cutoff = cutoff_from_probabilities(result.probabilities, cutoff=args.cutoff, alpha=args.alpha)
        labels = (result.probabilities > cutoff).astype(int)

        output = pd.DataFrame(
            {
                "source_csv": str(csv_file),
                "center_index": result.centers,
                "anomaly_probability": result.probabilities,
                "anomaly_label": labels,
                "cutoff": cutoff,
            }
        )
        output_path = part_dir / "probabilities.csv"
        output.to_csv(output_path, index=False)

        hist_path = None
        if args.hist_dir:
            from pi_tsad.visualization import plot_probability_histogram

            hist_dir = args.hist_dir / csv_file.stem
            hist_path = hist_dir / "probability_distribution.png"
            plot_probability_histogram(
                result.probabilities,
                cutoff=cutoff,
                title=f"PI-TSAD probabilities: {csv_file.stem}",
                output_path=hist_path,
            )
        else:
            from pi_tsad.visualization import plot_probability_histogram

            hist_path = part_dir / "probability_distribution.png"
            plot_probability_histogram(
                result.probabilities,
                cutoff=cutoff,
                title=f"PI-TSAD probabilities: {csv_file.stem}",
                output_path=hist_path,
            )

        rows.append(
            {
                "source_csv": str(csv_file),
                "probability_csv": str(output_path),
                "histogram": str(hist_path) if hist_path else "",
                "cutoff": cutoff,
                "n_windows": len(output),
                "n_anomalous": int(labels.sum()),
            }
        )
        if args.summary_plot:
            from pi_tsad.visualization import ProbabilityDistribution

            distributions.append(
                ProbabilityDistribution(part_label(len(distributions), csv_file), result.probabilities, cutoff)
            )
        print(f"  saved {output_path} | cutoff={cutoff:.6g} | anomalous={int(labels.sum()):,}")

    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    if args.summary_plot:
        from pi_tsad.visualization import plot_probability_distribution_grid

        plot_probability_distribution_grid(distributions, output_path=args.summary_plot)
        print(f"Saved combined probability-density plot to {args.summary_plot}")
    print(f"Saved batch summary to {summary_path}")
    return 0


def run_threshold_probabilities(args: argparse.Namespace) -> int:
    probability_csvs = expand_paths(args.probability_csvs)
    missing = [path for path in probability_csvs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing probability CSV file(s): {', '.join(str(path) for path in missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.hist_dir:
        args.hist_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    distributions = []
    for i, probability_csv in enumerate(probability_csvs):
        part_name = part_name_from_probability_csv(probability_csv)
        part_dir = args.output_dir / part_name
        part_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(probability_csv)
        if "anomaly_probability" not in df.columns:
            raise ValueError(f"{probability_csv} must contain an 'anomaly_probability' column.")
        probabilities = pd.to_numeric(df["anomaly_probability"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(probabilities)
        if not valid.all():
            df = df.loc[valid].copy()
            probabilities = probabilities[valid]
        cutoff = cutoff_from_probabilities(probabilities, cutoff=args.cutoff, alpha=args.alpha)
        labels = (probabilities > cutoff).astype(int)
        df["anomaly_label"] = labels
        df["cutoff"] = cutoff

        output_path = part_dir / f"thresholded_cutoff_{cutoff:.4g}.csv"
        df.to_csv(output_path, index=False)

        from pi_tsad.visualization import plot_probability_histogram

        hist_dir = args.hist_dir / part_name if args.hist_dir else part_dir
        hist_path = hist_dir / f"probability_distribution_cutoff_{cutoff:.4g}.png"
        plot_probability_histogram(
            probabilities,
            cutoff=cutoff,
            title=f"PI-TSAD probabilities: {probability_csv.stem}",
            output_path=hist_path,
        )
        rows.append(
            {
                "probability_csv": str(probability_csv),
                "thresholded_csv": str(output_path),
                "histogram": str(hist_path),
                "cutoff": cutoff,
                "n_windows": len(df),
                "n_anomalous": int(labels.sum()),
            }
        )
        if args.summary_plot:
            from pi_tsad.visualization import ProbabilityDistribution

            distributions.append(ProbabilityDistribution(part_label(i, Path(part_name)), probabilities, cutoff))
        print(f"Saved {output_path} and {hist_path}")

    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    if args.summary_plot:
        from pi_tsad.visualization import plot_probability_distribution_grid

        plot_probability_distribution_grid(distributions, output_path=args.summary_plot)
        print(f"Saved combined probability-density plot to {args.summary_plot}")
    print(f"Saved threshold summary to {summary_path}")
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
    if args.command == "detect-batch":
        return run_detect_batch(args)
    if args.command == "threshold-probabilities":
        return run_threshold_probabilities(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
