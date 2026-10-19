"""Tests for the masking semantics that make DKT correct:
padded positions must not contribute, predictions must be causal
(no looking at the answer being predicted), and the pair alignment
(logits[t] predicts interaction t+1) must hold exactly.
"""

import torch
import torch.nn as nn

from baseline import fit_per_skill_rate
from data.dataset import collate_fn, make_loader
from model.dkt import DKT, collect_predictions, masked_loss

NUM_SKILLS = 5


def _constant_logits(value, b, l):
    return torch.full((b, l, NUM_SKILLS), value)


def test_padding_does_not_change_loss():
    seqs = [{"skills": [0, 1, 2], "correct": [1, 0, 1]}]
    b1 = collate_fn(seqs)  # L = 3
    b2 = collate_fn(seqs)
    # append one padded column to batch 2: skill 0, correct 0, valid False
    b2["skill_ids"] = torch.cat([b2["skill_ids"], torch.zeros(1, 1, dtype=torch.long)], 1)
    b2["correct"] = torch.cat([b2["correct"], torch.zeros(1, 1)], 1)
    b2["valid"] = torch.cat([b2["valid"], torch.zeros(1, 1, dtype=torch.bool)], 1)
    logits1 = _constant_logits(0.7, 1, 3)
    logits2 = _constant_logits(0.7, 1, 4)
    l1 = masked_loss(logits1, b1["skill_ids"], b1["correct"], b1["valid"])
    l2 = masked_loss(logits2, b2["skill_ids"], b2["correct"], b2["valid"])
    assert torch.allclose(l1, l2)


def test_loss_matches_hand_computed_pairs():
    # sequence: (skill0,1)(skill1,0)(skill2,1)
    # pairs: t=0 predicts (skill1, 0); t=1 predicts (skill2, 1)
    seqs = [{"skills": [0, 1, 2], "correct": [1, 0, 1]}]
    b = collate_fn(seqs)
    k = 0.7
    logits = _constant_logits(k, 1, 3)
    loss = masked_loss(logits, b["skill_ids"], b["correct"], b["valid"])
    manual = (
        nn.functional.binary_cross_entropy_with_logits(torch.tensor(k), torch.tensor(0.0))
        + nn.functional.binary_cross_entropy_with_logits(torch.tensor(k), torch.tensor(1.0))
    ) / 2
    assert torch.allclose(loss, manual, atol=1e-6)


def test_predictions_are_causal():
    """Changing the input at step t must not alter predictions made at steps < t.

    If logits[t-1] changed, the model would be seeing the answer it is
    supposed to predict -- the classic knowledge-tracing leakage bug.
    """
    torch.manual_seed(0)
    model = DKT(NUM_SKILLS, hidden_size=8)
    model.eval()
    seqs = [{"skills": [0, 1, 2, 3], "correct": [1, 0, 1, 1]}]
    b = collate_fn(seqs)
    logits_a = model(b["skill_ids"], b["correct"], b["valid"])

    b_mod = {k: v.clone() for k, v in b.items()}
    b_mod["skill_ids"][0, 3] = 4
    b_mod["correct"][0, 3] = 0.0
    logits_b = model(b_mod["skill_ids"], b_mod["correct"], b_mod["valid"])

    assert torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-6)
    assert not torch.allclose(logits_a[:, 3], logits_b[:, 3], atol=1e-6)


def test_first_observation_seen_flag():
    seqs = [{"skills": [1, 1, 2, 1], "correct": [1, 0, 1, 1]}]
    loader = make_loader(seqs, batch_size=1)
    model = DKT(NUM_SKILLS, hidden_size=8)
    _, _, skills, seen = collect_predictions(model, loader)
    # pairs predict steps 1,2,3 -> skills [1, 2, 1]
    assert skills.tolist() == [1, 2, 1]
    # skill 1 appeared at step 0 (seen); skill 2 is at its first position;
    # skill 1 again seen at steps 0-2
    assert seen.tolist() == [True, False, True]


def test_baseline_handles_unseen_skill():
    import numpy as np

    skills = np.array([0, 0, 1, 1, 1])
    labels = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    rates, global_rate = fit_per_skill_rate(skills, labels, num_skills=4)
    assert rates[0] == 0.5
    assert abs(rates[1] - 2 / 3) < 1e-9
    assert rates[2] == global_rate  # skill 2 never seen -> global fallback
    assert abs(global_rate - 3 / 5) < 1e-9
