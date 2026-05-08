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
