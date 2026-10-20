"""Shared, cached data access for the dashboard pages.

Everything the pages show is computed from three real artifacts:
outputs/model_assistments.pt (checkpoint), outputs/demo_bundle.json.gz
(test students + skill names + baseline rates) and outputs/comparison.json
(from compare.py). Nothing is mocked or hard-coded.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from data.dataset import collate_fn
from model.dkt import DKT

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "outputs" / "demo_bundle.json.gz"
CHECKPOINT_PATH = ROOT / "outputs" / "model_assistments.pt"
COMPARISON_PATH = ROOT / "outputs" / "comparison.json"


@st.cache_resource
def load_model():
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=True)
    model = DKT(ckpt["num_skills"], ckpt["hidden_size"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@st.cache_data
def load_bundle():
    with gzip.open(BUNDLE_PATH, "rt", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_comparison():
    if not COMPARISON_PATH.exists():
        return None
    with open(COMPARISON_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_eval_pairs():
    """Flat held-out evaluation pairs (DKT + BKT probs, labels, skills, seen).

    Precomputed in the bundle so comparison controls respond instantly.
    Returns None for bundles exported before eval pairs existed.
    """
    bundle = load_bundle()
    ep = bundle.get("eval_pairs")
    if not ep:
        return None
    return {
        "probs_dkt": np.array(ep["probs_dkt"], dtype=float),
        "probs_bkt": np.array(ep["probs_bkt"], dtype=float),
        "labels": np.array(ep["labels"], dtype=float),
        "skills": np.array(ep["skills"]),
        "seen": np.array(ep["seen"], dtype=bool),
    }


@st.cache_data
def predict_sequence(student):
    """Causal per-step predictions for interactions 2..end (see demo docs)."""
    model, _ = load_model()
    batch = collate_fn([{"skills": student["skills"],
                         "correct": student["correct"]}])
    with torch.no_grad():
        logits = model(**batch)
    probs = torch.sigmoid(logits)[0, :-1].numpy()
    skills = np.array(student["skills"][1:])
    labels = np.array(student["correct"][1:])
    return probs[np.arange(len(skills)), skills], labels, skills


@st.cache_data
def skill_beliefs(student):
    """Model estimate for every skill after the student's FULL history.

    This is the probability the model would assign to a correct answer if a
    question on each skill came next -- a model estimate, not a measurement
    of the student's true knowledge.
    """
    return beliefs_upto(student["skills"], student["correct"],
                        len(student["skills"]))


@st.cache_data
def beliefs_upto(skills, correct, step):
    """Per-skill model beliefs after the first `step` interactions.

    step=0 means no history at all: the model honestly starts at 0.5 for
    every skill instead of guessing.
    """
    model, ckpt = load_model()
    if step <= 0:
        return np.full(ckpt["num_skills"], 0.5)
    batch = collate_fn([{"skills": list(skills[:step]),
                         "correct": list(correct[:step])}])
    with torch.no_grad():
        logits = model(**batch)
    return torch.sigmoid(logits)[0, -1].numpy()


def observed_by_skill(student, num_skills):
    """Raw observed accuracy and count per skill over the student's history."""
    counts = np.bincount(student["skills"], minlength=num_skills)
    correct = np.bincount(student["skills"], weights=student["correct"],
                          minlength=num_skills)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = correct / counts
    acc[counts == 0] = np.nan
    return counts, acc


MIN_EVIDENCE = 5  # below this many observations an estimate is "thin"

LOW_EVIDENCE_NOTE = (
    "Skills with fewer than 5 observations are marked as thin evidence: "
    "one or two questions cannot establish mastery of anything."
)
