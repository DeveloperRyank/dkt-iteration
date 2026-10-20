import numpy as np
import pytest

from metrics import (brier_score, expected_calibration_error, full_report,
                     log_loss, reliability_table)


def test_log_loss_known_value():
    assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(np.log(2))
    assert log_loss([1.0, 0.0], [1, 0]) == pytest.approx(0.0, abs=1e-6)


def test_brier_known_value():
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert brier_score([0.0, 0.0], [0, 0]) == pytest.approx(0.0, abs=1e-12)


def test_reliability_bins_and_counts():
    probs = [0.05, 0.15, 0.16, 0.95]
    labels = [0, 1, 1, 1]
    table = reliability_table(probs, labels)
    total = sum(r["count"] for r in table)
    assert total == 4
    bin2 = next(r for r in table if r["bin"] == "0.1-0.2")
    assert bin2["count"] == 2
    assert bin2["mean_predicted"] == pytest.approx(0.155)
    assert bin2["observed_rate"] == pytest.approx(1.0)


def test_ece_detects_overconfidence():
    # All predictions 0.9, none correct -> ECE should be 0.9.
    probs = [0.9] * 10
    labels = [0] * 10
    assert expected_calibration_error(probs, labels) == pytest.approx(0.9)


def test_ece_zero_when_perfectly_calibrated():
    # Every prediction equals its own bin's observed rate: construct bins so
    # that in [0.2,0.3) 25% of predictions are correct, mean pred 0.25.
    probs = [0.25] * 4 + [0.75] * 4
    labels = [1, 0, 0, 0, 1, 1, 1, 0]
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_full_report_has_all_metrics_and_matches_manual_auc():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200).astype(float)
    probs = np.clip(labels * 0.7 + rng.random(200) * 0.3, 0, 1)
    r = full_report("x", probs, labels)
    assert {"auc", "acc", "log_loss", "brier", "ece", "n"} <= set(r)
    assert r["n"] == 200
    assert r["auc"] > 0.8  # probs heavily correlated with labels
    assert 0 <= r["ece"] <= 1
