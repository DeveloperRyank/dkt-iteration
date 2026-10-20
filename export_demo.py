"""Export a gzipped bundle for the Streamlit dashboard.

The web app must not carry the 63 MB raw CSV, so we precompute everything it
needs: the held-out test-student sequences (same split seed as training),
per-skill baseline rates (train students only), skill names, AND the flat
evaluation pairs (DKT + BKT probabilities per pair) so the comparison page
can react to slider moves instantly without refitting models.

Usage: python export_demo.py
"""

import gzip
import json

import numpy as np
import torch

from baseline import fit_per_skill_rate
from bkt import BKTModel
from data.dataset import make_loader, training_pairs
from data.synthetic import split_by_student
from model.dkt import DKT, collect_predictions


def load_processed():
    from data.assistments import _cache_path_for
    with gzip.open(_cache_path_for(1000, 5), "rt", encoding="utf-8") as f:
        return json.load(f)


def main():
    payload = load_processed()
    sequences = payload["sequences"]
    num_skills = payload["num_skills"]
    skill_names = sorted(payload["skill_index"], key=payload["skill_index"].get)

    train_seqs, test_seqs = split_by_student(sequences, seed=42)
    tr_skills, tr_labels = training_pairs(train_seqs)
    rates, _ = fit_per_skill_rate(tr_skills.numpy(), tr_labels.numpy(), num_skills)

    ckpt = torch.load("outputs/model_assistments.pt", map_location="cpu",
                      weights_only=True)
    model = DKT(ckpt["num_skills"], ckpt["hidden_size"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    dkt_p, labels, skills, seen = collect_predictions(model,
                                                      make_loader(test_seqs, 64))
    bkt = BKTModel(num_skills).fit(train_seqs)
    bkt_p, b_labels, b_skills, b_seen = bkt.predict_pairs(test_seqs)
    assert len(bkt_p) == len(dkt_p) and np.array_equal(b_seen, seen) \
        and (b_skills == skills).all() and np.allclose(b_labels, labels)

    bundle = {
        "num_skills": num_skills,
        "skill_names": skill_names,
        "baseline_rates": [float(r) for r in rates],
        "model_meta": {k: v for k, v in ckpt.items() if k != "state_dict"},
        "test_students": [
            {"label": f"student {i + 1} ({len(s['skills'])} interactions)",
             **s}
            for i, s in enumerate(test_seqs)
        ],
        "eval_pairs": {
            "probs_dkt": [round(float(p), 4) for p in dkt_p],
            "probs_bkt": [round(float(p), 4) for p in bkt_p],
            "labels": [int(x) for x in labels],
            "skills": [int(x) for x in skills],
            "seen": [bool(x) for x in seen],
        },
    }
    with gzip.open("outputs/demo_bundle.json.gz", "wt", encoding="utf-8") as f:
        json.dump(bundle, f)
    n_pairs = sum(len(s["correct"]) for s in test_seqs)
    print(f"bundle written: {len(test_seqs)} test students, {n_pairs} interactions, "
          f"{num_skills} skills, {len(dkt_p)} eval pairs")


if __name__ == "__main__":
    main()
