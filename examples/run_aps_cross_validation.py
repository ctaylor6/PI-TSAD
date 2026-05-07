"""Run PI-TSAD 4-fold cross-validation on the bundled APS example CSVs."""

from pi_tsad.evaluation import cross_validate_aps
from pi_tsad.model import PITSADConfig


def main() -> None:
    config = PITSADConfig(window_radius=15, radius_multiplier=2, alpha=0.06)
    summary, _ = cross_validate_aps(config=config)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
