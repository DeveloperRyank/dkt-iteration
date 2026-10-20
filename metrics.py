"""Prediction-quality metrics beyond AUC/accuracy.

AUC measures ranking, accuracy measures thresholded calls, log loss and the
Brier score measure how close predicted probabilities are to outcomes, and
reliability/ECE measure whether "70% predictions" really happen 70% of the
time. None of them establish that a model understands mastery.
"""

import numpy as np

EPS = 1e-7


def _clean(probs, labels):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    assert probs.shape == labels.shape, "probs and labels must align"
    return np.clip(probs, EPS, 1.0 - EPS), labels


def log_loss(probs, labels):
    probs, labels = _clean(probs, labels)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def brier_score(probs, labels):
    probs, labels = _clean(probs, labels)
    return float(np.mean((probs - labels) ** 2))


def reliability_table(probs, labels, n_bins=10):
    """Mean predicted vs observed rate per probability bin, with counts.

    Bins are [0, 0.1), [0.1, 0.2), ... [0.9, 1.0]. Empty bins are omitted.
    """
    probs, labels = _clean(probs, labels)
    rows = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        in_bin = (probs >= lo) & (probs < hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "count": n,
            "mean_predicted": float(probs[in_bin].mean()),
            "observed_rate": float(labels[in_bin].mean()),
        })
    return rows


def expected_calibration_error(probs, labels, n_bins=10):
    probs, labels = _clean(probs, labels)
    table = reliability_table(probs, labels, n_bins)
    total = sum(r["count"] for r in table)
    return float(sum(r["count"] / total * abs(r["mean_predicted"] - r["observed_rate"])
                     for r in table))


def full_report(name, probs, labels, extra=None):
    """All metrics for one predictor, plus any caller-supplied fields."""
    probs, labels = _clean(probs, labels)
    n_unique = len(np.unique(labels))
    from sklearn.metrics import accuracy_score, roc_auc_score
    out = {
        "name": name,
        "n": int(len(labels)),
        "auc": float(roc_auc_score(labels, probs)) if n_unique > 1 else float("nan"),
        "acc": float(accuracy_score(labels, (np.asarray(probs) >= 0.5).astype(int))),
        "log_loss": log_loss(probs, labels),
        "brier": brier_score(probs, labels),
        "ece": expected_calibration_error(probs, labels),
    }
    if extra:
        out.update(extra)
    return out
