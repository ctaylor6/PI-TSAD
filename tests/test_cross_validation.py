from pi_tsad.evaluation import cross_validate_aps
from pi_tsad.model import PITSADConfig


def test_cross_validate_aps_returns_four_folds_and_mean():
    config = PITSADConfig(n_estimators=10, window_radius=15, alpha=0.06)
    summary, artifacts = cross_validate_aps(config=config)
    assert list(summary["test_key"]) == ["050", "053", "074", "077", "mean"]
    assert len(artifacts) == 4
    assert summary["f1"].notna().all()
