"""Example skeleton for applying PI-TSAD to a large user-supplied part file."""

from pathlib import Path

from pi_tsad.cli import main


MODEL_PATH = Path("models/pi_tsad_aps.joblib")
PART_CSV = Path("path/to/your/full_part.csv")


if __name__ == "__main__":
    main(["train-aps", "--output", str(MODEL_PATH)])
    main(
        [
            "detect-part",
            str(MODEL_PATH),
            str(PART_CSV),
            "--layer",
            "150",
            "--max-rows",
            "10000",
            "--output",
            "outputs/layer_150_predictions.csv",
        ]
    )
