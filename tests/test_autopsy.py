import numpy as np
import pytest
import torch

from autopsy import autopsy, predict_prob
from model.dkt import DKT


def make_model(seed=0):
    torch.manual_seed(seed)
    return DKT(6, 8)


def test_autopsy_shapes_and_ranges():
    model = make_model()
    skills = [0, 1, 2, 0, 3]
    correct = [1, 0, 1, 1, 0]
    r = autopsy(model, skills, correct, 4)
    assert len(r["interactions"]) == 4  # all positions before the target
    assert 0.0 < r["predicted_prob"] < 1.0
    assert r["actual_outcome"] == correct[4]
    assert r["target_skill"] == 3
    assert r["prior_same_skill_observations"] == 0
    assert r["prior_same_skill_rate"] is None
    for row in r["interactions"]:
        assert 0.0 < row["prob_if_dropped"] < 1.0
        assert 0.0 < row["prob_if_flipped"] < 1.0


def test_prediction_ignores_target_outcome_itself():
    """Changing the audited interaction's own label must not change its
    prediction — predictions come from earlier inputs only."""
    model = make_model()
    skills = [0, 1, 2, 0, 1]
    p1 = predict_prob(model, skills, [1, 0, 1, 1, 0], 4)
    p2 = predict_prob(model, skills, [1, 0, 1, 1, 1], 4)
    assert p1 == pytest.approx(p2)


def test_flipping_a_prior_answer_moves_the_prediction():
    model = make_model(seed=2)
    skills = [1, 1, 1]
    p_after_win = predict_prob(model, skills, [1, 1, 0], 2)
    p_after_loss = predict_prob(model, skills, [0, 0, 0], 2)
    r = autopsy(model, skills, [1, 1, 0], 2)
    assert abs(r["interactions"][0]["delta_flip"]) > 0  # some movement exists
    assert isinstance(r["max_drop_delta"], float)


def test_autopsy_requires_history():
    model = make_model()
    with pytest.raises(AssertionError):
        autopsy(model, [0, 1], [1, 0], 0)
