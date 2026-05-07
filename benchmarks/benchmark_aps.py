"""Small benchmark runner for the public APS PI-TSAD workflow.

This is intentionally lightweight: it benchmarks the reproducible single-track
APS path, not a full-part build that can take a long time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from pi_tsad.evaluation import cross_validate_aps, train_aps_model
from pi_tsad.model import PITSADConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark PI-TSAD on bundled APS data.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_aps.json"))
    parser.add_argument("--window-radius", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.06)
    parser.add_argument("--n-estimators", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    config = PITSADConfig(
        window_radius=args.window_radius,
        alpha=args.alpha,
        n_estimators=args.n_estimators,
    )

    cv_times = []
    train_times = []
    last_summary = None

    for _ in range(args.repeats):
        start = perf_counter()
        summary, _ = cross_validate_aps(data_dir=args.data_dir, config=config)
        cv_times.append(perf_counter() - start)
        last_summary = summary

        start = perf_counter()
        train_aps_model(data_dir=args.data_dir, config=config)
        train_times.append(perf_counter() - start)

    assert last_summary is not None
    mean_row = last_summary[last_summary["test_key"] == "mean"].iloc[0].to_dict()
    result = {
        "config": {
            "window_radius": config.window_radius,
            "alpha": config.alpha,
            "n_estimators": config.n_estimators,
            "threshold": "empirical robust-z quantile",
        },
        "repeats": args.repeats,
        "cross_validation_seconds": {
            "min": min(cv_times),
            "mean": sum(cv_times) / len(cv_times),
            "max": max(cv_times),
            "runs": cv_times,
        },
        "train_all_aps_seconds": {
            "min": min(train_times),
            "mean": sum(train_times) / len(train_times),
            "max": max(train_times),
            "runs": train_times,
        },
        "mean_metrics": {
            "precision": float(mean_row["precision"]),
            "recall": float(mean_row["recall"]),
            "f1": float(mean_row["f1"]),
            "threshold": float(mean_row["threshold"]),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
