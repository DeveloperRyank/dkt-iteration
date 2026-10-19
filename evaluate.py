import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def report(name, probs, labels):
    """ROC-AUC and accuracy at threshold 0.5 for one predictor."""
    if len(np.unique(labels)) < 2:
        return {"name": name, "auc": float("nan"), "acc": float("nan"), "n": len(labels)}
    return {
        "name": name,
        "auc": roc_auc_score(labels, probs),
        "acc": accuracy_score(labels, (probs >= 0.5).astype(int)),
        "n": len(labels),
    }


def format_report(r):
    return f"{r['name']:<28} n={r['n']:>7}  AUC={r['auc']:.4f}  Acc={r['acc']:.4f}"
