"""Compare DKT / BKT / per-skill rate baseline / random on identical pairs.

All predictors are evaluated on exactly the same eligible test pairs (each
skill's first observation excluded), so the numbers are directly comparable.
Writes outputs/comparison.json and outputs/bkt_params.json.

Usage: python compare.py [--checkpoint outputs/model_assistments.pt]
"""

import argparse
import json
import time

import numpy as np
import torch

from baseline import fit_per_skill_rate, predict_per_skill_rate
from bkt import BKTModel
from data.dataset import make_loader, training_pairs
from data.synthetic import split_by_student
from metrics import full_report, reliability_table
from model.dkt import DKT, collect_predictions

SETTING_KEYS = ("hidden_size", "dropout", "epochs", "lr", "batch_size", "seed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/model_assistments.pt")
    ap.add_argument("--out", default="outputs/comparison.json")
    args = ap.parse_args()

    from data.assistments import load_assistments
    data, _ = load_assistments(verbose=False)
    train_seqs, test_seqs = split_by_student(data["sequences"], seed=42)
    print(f"{len(train_seqs)} train / {len(test_seqs)} test students, "
          f"{data['num_skills']} skills")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DKT(ckpt["num_skills"], ckpt["hidden_size"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    t0 = time.perf_counter()
    probs, labels, skills, seen = collect_predictions(model,
                                                      make_loader(test_seqs, 64))
    dkt_eval_s = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    bkt = BKTModel(data["num_skills"]).fit(train_seqs)
    bkt_fit_s = round(time.perf_counter() - t0, 1)
    bprobs, blabels, bskills, bseen = bkt.predict_pairs(test_seqs)
    # Same pair convention as collect_predictions: verify alignment.
    assert len(bprobs) == len(probs)
    assert (bskills == skills).all() and np.array_equal(bseen, seen)
    assert np.allclose(blabels, labels)
    print(f"BKT fitted per skill in {bkt_fit_s}s")

    tr_skills, tr_labels = training_pairs(train_seqs)
    rates, g = fit_per_skill_rate(tr_skills.numpy(), tr_labels.numpy(),
                                  data["num_skills"])

    keep = seen
    y = labels[keep]
    predictors = {
        "DKT": probs[keep],
        "BKT (EM, per skill)": bprobs[keep],
        "per-skill rate baseline": predict_per_skill_rate(rates, g, skills[keep]),
        "random guess": np.random.default_rng(0).random(int(keep.sum())),
    }
    extra = {
        "DKT": {"settings": {k: ckpt.get(k) for k in SETTING_KEYS},
                "train_seconds": ckpt.get("train_seconds"),
                "eval_seconds": dkt_eval_s},
        "BKT (EM, per skill)": {"settings": {"em_iters_max": 100, "em_tol": 1e-6,
                                             "structure": "one HMM per skill"},
                                "train_seconds": bkt_fit_s},
    }

    rows, reliability = [], {}
    for name, p in predictors.items():
        rows.append(full_report(name, p, y, extra=extra.get(name)))
        reliability[name] = reliability_table(p, y)

    for r in rows:
        print(f"{r['name']:<26} n={r['n']:>6}  AUC={r['auc']:.4f}  "
              f"Acc={r['acc']:.4f}  LL={r['log_loss']:.4f}  "
              f"Brier={r['brier']:.4f}  ECE={r['ece']:.4f}")

    out = {
        "protocol": "same eligible test pairs for all models "
                    "(first observation of each skill excluded); "
                    "student-level split seed 42; ASSISTments 2009-2010",
        "n_pairs": int(keep.sum()),
        "models": rows,
        "reliability": reliability,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")
    with open("outputs/bkt_params.json", "w") as f:
        json.dump({str(k): v for k, v in bkt.params.items()}, f, indent=2)
    print("wrote outputs/bkt_params.json")


if __name__ == "__main__":
    main()
