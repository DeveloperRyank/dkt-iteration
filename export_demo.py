"""Export a small gzipped bundle for the Streamlit demo app.

The web app must not carry the 63 MB raw CSV, so we precompute everything it
needs: the exact held-out test-student sequences (same split seed as training),
the per-skill baseline rates (fitted on train students only), and skill names.

Usage: python export_demo.py
"""

import gzip
import json

import numpy as np
import torch

from baseline import fit_per_skill_rate
from data.dataset import training_pairs
from data.synthetic import split_by_student


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
    }
    with gzip.open("outputs/demo_bundle.json.gz", "wt", encoding="utf-8") as f:
        json.dump(bundle, f)
    n_pairs = sum(len(s["correct"]) for s in test_seqs)
    print(f"bundle written: {len(test_seqs)} test students, {n_pairs} interactions, "
          f"{num_skills} skills")


if __name__ == "__main__":
    main()
