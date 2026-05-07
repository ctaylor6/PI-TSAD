"""Data loading helpers for TED signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pi_tsad.constants import APS_EXPERIMENT_KEYS, APS_START_TIMES


def repo_root() -> Path:
    """Return the project root when running from a source checkout."""
    return Path(__file__).resolve().parents[2]


def default_aps_data_dir() -> Path:
    return repo_root() / "data" / "aps"


def aps_csv_files(data_dir: str | Path | None = None) -> dict[str, Path]:
    """Return paths for the four APS demonstration CSV files."""
    root = Path(data_dir) if data_dir is not None else default_aps_data_dir()
    return {key: root / f"{key}.csv" for key in APS_EXPERIMENT_KEYS}


def experiment_key_from_path(path: str | Path) -> str:
    return Path(path).stem.zfill(3)


def load_ted_signal(
    csv_file: str | Path,
    *,
    time_column: str = "time",
    signal_column: str = "TED",
    start_time: float | None = None,
    end_time: float | None = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a compact TED signal CSV and return time and signal arrays."""
    path = Path(csv_file)
    df = pd.read_csv(path)
    if time_column not in df.columns or signal_column not in df.columns:
        raise ValueError(
            f"{path} must contain columns {time_column!r} and {signal_column!r}."
        )

    key = experiment_key_from_path(path)
    if start_time is None:
        start_time = APS_START_TIMES.get(key, 0.0017)

    time = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=float)
    signal = pd.to_numeric(df[signal_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time) & np.isfinite(signal)
    if start_time is not None:
        valid &= time >= start_time
    if end_time is not None:
        valid &= time <= end_time
    return time[valid], signal[valid]


def load_part_csv(
    csv_file: str | Path,
    *,
    skiprows: int = 1,
    columns: tuple[str, ...] = ("x", "y", "TEP", "TED", "layer"),
    encoding: str = "ISO-8859-1",
) -> pd.DataFrame:
    """Load a full-scale part CSV supplied by a user.

    Full part files can be very large. This helper centralizes the expected
    column cleanup while keeping the long-running analysis out of import time.
    """
    df = pd.read_csv(csv_file, skiprows=skiprows, names=list(columns), encoding=encoding)
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna().reset_index(drop=True)
