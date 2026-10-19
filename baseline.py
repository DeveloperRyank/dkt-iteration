"""Baselines: constant rate and per-skill empirical rate, both fitted on
training pairs only (fitted on the same (skill, correct) pairs the DKT sees).
"""

import numpy as np


def fit_per_skill_rate(skill_ids, labels, num_skills):
    counts = np.bincount(skill_ids, minlength=num_skills).astype(float)
    correct = np.bincount(skill_ids, weights=labels, minlength=num_skills)
    global_rate = labels.mean() if len(labels) else 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = correct / counts
    rates[counts == 0] = global_rate
    return rates, global_rate


def predict_per_skill_rate(rates, global_rate, skills):
    return rates[skills]
