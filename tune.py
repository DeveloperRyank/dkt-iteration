"""Random-search hyperparameter tuning for DKT on ASSISTments.

Protocol (leakage-safe):
- The Stage 4 student-level test split (seed 42, 20%) is set aside and never
  touched during tuning.
- A validation split (seed 43, 12.5% of the remaining train students) is used
  for ALL model selection.
- After the search, the best config is retrained on train+val and evaluated on
  test exactly once.
"""

import argparse
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from config import Config
from data.assistments import load_assistments
from data.dataset import make_loader
from data.synthetic import split_by_student
from model.dkt import DKT, collect_predictions, masked_loss
from train import train_model


def val_auc(model, seqs, batch_size):
    probs, labels, _, seen = collect_predictions(model, make_loader(seqs, batch_size))
    probs, labels = probs[seen], labels[seen]
    return roc_auc_score(labels, probs)


def run_trial(name, num_skills, train_seqs, val_seqs, cfg, epochs):
    cfg.epochs = epochs
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = DKT(num_skills, cfg.hidden_size, cfg.dropout)
    loader = make_loader(train_seqs, cfg.batch_size, shuffle=True)
    t0 = time.time()
    train_model(model, loader, cfg)
    auc = val_auc(model, val_seqs, cfg.batch_size)
    print(f"trial {name}: hidden={cfg.hidden_size} lr={cfg.lr} "
          f"dropout={cfg.dropout} -> val AUC={auc:.4f} ({time.time() - t0:.0f}s)",
          flush=True)
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--final-epochs", type=int, default=20)
    args = parser.parse_args()

    cfg = Config()
    data, _ = load_assistments(verbose=False)
    num_skills = data["num_skills"]
    train_full, test_seqs = split_by_student(data["sequences"], seed=cfg.seed)
    train_seqs, val_seqs = split_by_student(train_full, test_frac=0.125, seed=43)
    print(f"students: train={len(train_seqs)} val={len(val_seqs)} "
          f"test={len(test_seqs)} (test untouched during search)", flush=True)

    space = [
        {"hidden_size": h, "dropout": d, "lr": lr}
        for h in (32, 64, 128, 200)
        for d in (0.2, 0.5)
        for lr in (3e-4, 1e-3, 3e-3)
    ]
    rng = random.Random(cfg.seed)
    trials = rng.sample(space, min(args.trials, len(space)))

    results = []
    for i, t in enumerate(trials):
        trial_cfg = Config(**{k: v for k, v in t.items()})
        auc = run_trial(f"{i + 1}/{len(trials)}", num_skills, train_seqs, val_seqs,
                        trial_cfg, args.epochs)
        results.append({"config": t, "val_auc": auc})

    results.sort(key=lambda r: -r["val_auc"])
    print("\nranking by validation AUC:")
    for r in results:
        c = r["config"]
        print(f"  val AUC={r['val_auc']:.4f}  hidden={c['hidden_size']} "
              f"lr={c['lr']} dropout={c['dropout']}")

    best = results[0]
    print(f"\nretraining best config on train+val for {args.final_epochs} epochs...",
          flush=True)
    best_cfg = Config(**{k: v for k, v in best["config"].items()})
    best_cfg.epochs = args.final_epochs
    combined = train_seqs + val_seqs
    torch.manual_seed(best_cfg.seed)
    np.random.seed(best_cfg.seed)
    model = DKT(num_skills, best_cfg.hidden_size, best_cfg.dropout)
    train_model(model, make_loader(combined, best_cfg.batch_size, shuffle=True),
                best_cfg)

    probs, labels, _, seen = collect_predictions(
        model, make_loader(test_seqs, best_cfg.batch_size))
    test_auc = roc_auc_score(labels[seen], probs[seen])
    acc = float(((probs[seen] >= 0.5).astype(int) == labels[seen]).mean())
    print(f"\nFINAL (single test evaluation):")
    print(f"  test AUC={test_auc:.4f}  Acc={acc:.4f}  n={int(seen.sum())}")
    with open("outputs/best_config.json", "w", encoding="utf-8") as f:
        json.dump({"config": best["config"], "val_auc": best["val_auc"],
                   "test_auc": test_auc, "test_acc": acc,
                   "final_epochs": args.final_epochs}, f, indent=2)
    print("saved outputs/best_config.json")


if __name__ == "__main__":
    main()
