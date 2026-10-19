"""Shape and padding sanity tests."""

import torch

from data.dataset import collate_fn, make_loader
from data.synthetic import generate_sequences, split_by_student
from model.dkt import DKT, collect_predictions, masked_loss

NUM_SKILLS = 5


def _batch():
    seqs = [
        {"skills": [0, 1, 2, 3], "correct": [1, 0, 1, 1]},
        {"skills": [4, 0], "correct": [0, 1]},
    ]
    return collate_fn(seqs)


def test_collate_pads_to_longest():
    b = _batch()
    assert b["skill_ids"].shape == (2, 4)
    assert b["correct"].shape == (2, 4)
    assert b["valid"].dtype == torch.bool
    assert b["valid"][1].tolist() == [True, True, False, False]


def test_forward_shapes():
    b = _batch()
    model = DKT(NUM_SKILLS, hidden_size=8)
    logits = model(b["skill_ids"], b["correct"], b["valid"])
    assert logits.shape == (2, 4, NUM_SKILLS)
    assert torch.isfinite(logits).all()


def test_masked_loss_is_scalar_and_finite():
    b = _batch()
    model = DKT(NUM_SKILLS, hidden_size=8)
    logits = model(b["skill_ids"], b["correct"], b["valid"])
    loss = masked_loss(logits, b["skill_ids"], b["correct"], b["valid"])
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_loss_backprops():
    b = _batch()
    model = DKT(NUM_SKILLS, hidden_size=8)
    logits = model(b["skill_ids"], b["correct"], b["valid"])
    masked_loss(logits, b["skill_ids"], b["correct"], b["valid"]).backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_collect_predictions_pair_count():
    data = generate_sequences(num_students=6, num_skills=NUM_SKILLS, seed=1)
    loader = make_loader(data["sequences"], batch_size=3)
    model = DKT(NUM_SKILLS, hidden_size=8)
    probs, labels, skills, seen = collect_predictions(model, loader)
    expected = sum(len(s["skills"]) - 1 for s in data["sequences"])
    assert len(probs) == len(labels) == len(skills) == len(seen) == expected


def test_split_by_student_is_disjoint():
    data = generate_sequences(num_students=10, num_skills=NUM_SKILLS, seed=2)
    train, test = split_by_student(data["sequences"], seed=0)
    assert len(train) + len(test) == 10
    train_ids = {id(s) for s in train}
    assert all(id(s) not in train_ids for s in test)
