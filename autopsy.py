"""Prediction autopsy: inspect and stress-test one individual prediction.

For a chosen interaction we report the DKT prediction, the actual held-out
outcome, the student's raw history for that skill, and how the prediction
moves when each earlier interaction is dropped or flipped. These are
diagnostic ablation experiments — an interaction that changes the number is
not "the reason" the model predicted what it did.
"""

import numpy as np
import torch

from model.dkt import DKT


@torch.no_grad()
def predict_prob(model, skills, correct, target_idx):
    """P(correct at target_idx) from inputs of interactions 0..target_idx-1.

    Uses the model's own convention (logits[t] predicts interaction t+1), so
    no padding is involved: one live sequence of length target_idx + 1.
    """
    model.eval()
    skill_t = torch.tensor([skills], dtype=torch.long)
    corr_t = torch.tensor([correct], dtype=torch.float)
    valid_t = torch.ones_like(skill_t, dtype=torch.bool)
    logits = model(skill_t, corr_t, valid_t)
    t = target_idx - 1
    return float(torch.sigmoid(logits[0, t, skills[target_idx]]))


def autopsy(model, skills, correct, target_idx, flag_threshold=0.05):
    """Full diagnostic report for the prediction at `target_idx`.

    target_idx must be >= 1 (there must be some history).
    """
    assert 1 <= target_idx < len(skills)
    base = predict_prob(model, skills, correct, target_idx)

    prior = [j for j in range(target_idx) if skills[j] == skills[target_idx]]
    rows = []
    for j in range(target_idx):
        s_drop = skills[:j] + skills[j + 1:]
        c_drop = correct[:j] + correct[j + 1:]
        p_drop = predict_prob(model, s_drop, c_drop, target_idx - 1)
        c_flip = list(correct)
        c_flip[j] = 1 - c_flip[j]
        p_flip = predict_prob(model, skills, c_flip, target_idx)
        rows.append({
            "step": j,
            "skill": skills[j],
            "correct": correct[j],
            "prob_if_dropped": p_drop,
            "delta_drop": base - p_drop,
            "prob_if_flipped": p_flip,
            "delta_flip": base - p_flip,
            "flagged": abs(base - p_drop) > flag_threshold
                      or abs(base - p_flip) > flag_threshold,
        })

    return {
        "target_step": target_idx,
        "target_skill": skills[target_idx],
        "predicted_prob": base,
        "actual_outcome": correct[target_idx],
        "prior_same_skill_observations": len(prior),
        "prior_same_skill_rate": (float(np.mean([correct[j] for j in prior]))
                                  if prior else None),
        "interactions": rows,
        "max_drop_delta": max((abs(r["delta_drop"]) for r in rows), default=0.0),
        "max_flip_delta": max((abs(r["delta_flip"]) for r in rows), default=0.0),
        "note": "ablation deltas are diagnostic experiments, not causal "
                "explanations of the model's internal reasoning",
    }
