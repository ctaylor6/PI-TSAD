import numpy as np

from pi_tsad.thresholding import predicted_intervals, robust_threshold


def test_robust_threshold_flags_high_scores():
    scores = np.array([0.01, 0.02, 0.02, 0.03, 0.9])
    threshold, *_rest, indices = robust_threshold(scores, alpha=0.2, mode="empirical")
    assert threshold < 0.9
    assert indices.tolist() == [4]


def test_predicted_intervals_groups_contiguous_labels():
    centers = np.array([0.0, 1.0, 2.0, 3.0])
    labels = np.array([0, 1, 1, 0])
    assert predicted_intervals(centers, labels, window_size=0.5) == [(0.75, 2.25)]
