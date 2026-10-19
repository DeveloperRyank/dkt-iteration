"""Tiny synthetic dataset with structure a DKT can learn.

Each student starts each skill at a mastery level near their overall ability
minus the skill's difficulty. Mastery then moves up after a correct answer and
down after an incorrect one, so a student's recent history on a skill predicts
their next answer on it. A static per-skill accuracy baseline cannot see this
recency effect; the LSTM can.
"""

import random

import numpy as np


def generate_sequences(
    num_students=400,
    num_skills=12,
    min_len=40,
    max_len=120,
    seed=0,
):
    rng = np.random.default_rng(seed)
    difficulty = rng.uniform(0.0, 2.0, size=num_skills)
    seqs = []
    for _ in range(num_students):
        ability = rng.normal(0.0, 1.0)
        mastery = ability - difficulty
        length = rng.integers(min_len, max_len + 1)
        skills, corrects = [], []
        for _ in range(length):
            s = int(rng.integers(num_skills))
            p = float(1.0 / (1.0 + np.exp(-mastery[s])))
            p = min(max(p, 0.05), 0.95)
            c = int(rng.random() < p)
            mastery[s] += 0.7 if c == 1 else -0.7
            skills.append(s)
            corrects.append(c)
        seqs.append({"skills": skills, "correct": corrects})
    return {"num_skills": num_skills, "sequences": seqs}


def split_by_student(seqs, test_frac=0.2, seed=42):
    """Leakage guard: a student's whole sequence goes to one side only."""
    idx = list(range(len(seqs)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    train = [seqs[i] for i in idx[:cut]]
    test = [seqs[i] for i in idx[cut:]]
    return train, test
