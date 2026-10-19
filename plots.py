"""Matplotlib figures saved to outputs/ (non-interactive backend)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def save_plots(loss_history, eval_named_pairs, prefix="run"):
    os.makedirs("outputs", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean train loss (masked BCE)")
    ax.set_title(f"Training loss ({prefix})")
    loss_path = f"outputs/loss_{prefix}.png"
    fig.tight_layout()
    fig.savefig(loss_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, probs, labels in eval_named_pairs:
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random (AUC=0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Test ROC curves ({prefix})")
    ax.legend(loc="lower right", fontsize=8)
    roc_path = f"outputs/roc_{prefix}.png"
    fig.tight_layout()
    fig.savefig(roc_path)
    plt.close(fig)

    print(f"saved: {loss_path}, {roc_path}")
