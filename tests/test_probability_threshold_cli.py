import numpy as np
import pandas as pd

from pi_tsad.cli import main


def test_threshold_probabilities_reuses_saved_scores(tmp_path):
    probabilities = tmp_path / "part_probabilities.csv"
    pd.DataFrame(
        {
            "center_index": [0, 1, 2, 3],
            "anomaly_probability": [0.05, 0.2, 0.7, 0.9],
        }
    ).to_csv(probabilities, index=False)

    output_dir = tmp_path / "thresholded"
    hist_dir = tmp_path / "histograms"
    summary_plot = tmp_path / "summary.png"
    exit_code = main(
        [
            "threshold-probabilities",
            str(probabilities),
            "--cutoff",
            "0.5",
            "--output-dir",
            str(output_dir),
            "--hist-dir",
            str(hist_dir),
            "--summary-plot",
            str(summary_plot),
        ]
    )

    assert exit_code == 0
    thresholded = list(output_dir.glob("part/thresholded_cutoff_0.5.csv"))
    histograms = list(hist_dir.glob("part/probability_distribution_cutoff_0.5.png"))
    assert len(thresholded) == 1
    assert len(histograms) == 1
    assert summary_plot.exists()
    df = pd.read_csv(thresholded[0])
    assert df["anomaly_label"].tolist() == [0, 0, 1, 1]


def test_detect_part_skips_feature_scaling_for_full_scale_parts(monkeypatch, tmp_path):
    csv_file = tmp_path / "part.csv"
    csv_file.write_text("metadata\n0,0,0,0.1,1\n", encoding="utf-8")
    model_file = tmp_path / "model.joblib"
    model_file.write_text("placeholder", encoding="utf-8")

    class DummyResult:
        centers = np.array([0])
        probabilities = np.array([0.2])
        threshold = 0.1

    class DummyModel:
        @classmethod
        def load(cls, path):
            assert path == model_file
            return cls()

        def predict_signal(self, time, signal, *, scale_features=True):
            assert scale_features is False
            return DummyResult()

    monkeypatch.setattr("pi_tsad.cli.PITSADModel", DummyModel)

    exit_code = main(
        [
            "detect-part",
            str(model_file),
            str(csv_file),
            "--output",
            str(tmp_path / "predictions.csv"),
        ]
    )

    assert exit_code == 0
