import numpy as np

from bkt import BKTModel, fit_bkt
from data.synthetic import generate_sequences, split_by_student


def simulate_bkt(pi, pT, pG, pS, n_students, max_len, seed):
    """Generate data from the exact BKT process, for parameter recovery."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_students):
        learned = rng.random() < pi
        obs = []
        for _ in range(max_len):
            if not learned and rng.random() < pT:
                learned = True
            p_correct = (1 - pS) if learned else pG
            obs.append(1 if rng.random() < p_correct else 0)
        out.append(obs)
    return out


def test_parameter_recovery():
    true = dict(pi=0.3, pT=0.1, pG=0.2, pS=0.05)
    seqs = simulate_bkt(**true, n_students=400, max_len=40, seed=1)
    est = fit_bkt(seqs)
    for k, v in true.items():
        assert abs(est[k] - v) < 0.1, f"{k}: true {v}, est {est[k]:.3f}"


def test_predict_pairs_alignment_with_pair_convention():
    """One prediction per interaction after the first, per student."""
    seqs = generate_sequences(num_students=20, seed=3)["sequences"]
    model = BKTModel(12).fit(seqs[:10], n_iter=30)
    probs, labels, skills, seen = model.predict_pairs(seqs[10:])
    expected_n = sum(len(s["skills"]) - 1 for s in seqs[10:])
    assert len(probs) == expected_n
    assert (labels >= 0).all() and (labels <= 1).all()
    assert (probs > 0).all() and (probs < 1).all()
    # seen means the skill occurred earlier in the same student's history
    assert seen.dtype == bool


def test_beats_random_on_synthetic():
    seqs = generate_sequences(num_students=200, seed=5)["sequences"]
    train, test = split_by_student(seqs, seed=5)
    model = BKTModel(12).fit(train, n_iter=50)
    probs, labels, _, seen = model.predict_pairs(test)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels[seen], probs[seen])
    assert auc > 0.7, f"BKT should exploit mastery dynamics, got {auc:.3f}"
