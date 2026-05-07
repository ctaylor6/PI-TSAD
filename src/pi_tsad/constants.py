"""Default APS example dataset metadata."""

from __future__ import annotations

APS_EXPERIMENT_KEYS = ("050", "053", "074", "077")

APS_COLLAPSE_TIMES: dict[str, list[float]] = {
    "050": [0.00368, 0.00448, 0.00532, 0.00582],
    "053": [0.00208, 0.00504, 0.005765, 0.00978, 0.00732],
    "074": [0.00938, 0.004305, 0.00456, 0.00562],
    "077": [0.002653, 0.00331, 0.00493],
}

APS_START_TIMES: dict[str, float] = {
    "050": 0.0017,
    "053": 0.0015,
    "074": 0.0020,
    "077": 0.0020,
}
