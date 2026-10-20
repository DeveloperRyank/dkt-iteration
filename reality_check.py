"""Mastery reality check: does next-answer prediction reflect learning?

ASSISTments 2009-2010 contains no delayed retests and no labelled transfer
questions, so this module does two things instead of pretending otherwise:

1. A *sequence-position proxy*: accuracy/AUC for predictions early vs late in
   each student's test sequence. Later positions have longer histories and,
   in this dataset, generally appear later in the school year — suggestive,
   NOT proof of durable learning.
2. An *import path*: a documented CSV schema for delayed/transfer assessments
   collected later, with validation and an agreement report against the
   model's predictions. Never treat these reports as proof of mastery.
"""

import numpy as np
import pandas as pd
import torch

from data.dataset import make_loader
from metrics import full_report

REQUIRED_COLUMNS = ["student_index", "skill_name", "question_id", "is_transfer",
                    "correct"]

TEMPLATE_DOC = """Import schema (CSV):
  student_index  int    index of a test-set student (same order as demo bundle)
  skill_name     str    must match one of the dataset's skill names
  question_id    str    your identifier for the (delayed/transfer) question
  is_transfer    0|1    1 if the question tests the skill in a new context
  correct        0|1    the student's outcome on that question
"""


def collect_with_positions(model, sequences):
    """Like model.dkt.collect_predictions but also returns (pos, length)."""
    model.eval()
    loader = make_loader(sequences, 64, shuffle=False)
    out = {k: [] for k in ("probs", "labels", "skills", "seen", "pos", "len")}
    with torch.no_grad():
        for batch in loader:
            skill_ids, correct, valid = (batch["skill_ids"], batch["correct"],
                                         batch["valid"])
            logits = model(skill_ids, correct, valid)
            pair_mask = valid[:, :-1] & valid[:, 1:]
            selected = torch.gather(logits[:, :-1], 2,
                                    skill_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            probs = torch.sigmoid(selected)
            lengths = valid.sum(dim=1)
            b_idx, t_idx = pair_mask.nonzero(as_tuple=True)
            for b, t in zip(b_idx.tolist(), t_idx.tolist()):
                s = int(skill_ids[b, t + 1])
                out["probs"].append(float(probs[b, t]))
                out["labels"].append(float(correct[b, t + 1]))
                out["skills"].append(s)
                out["seen"].append(s in skill_ids[b, :t + 1].tolist())
                out["pos"].append(t + 1)
                out["len"].append(int(lengths[b]))
    return {k: np.array(v) for k, v in out.items()}


def position_band_report(m, n_bands=3):
    """Metrics per relative-position band over eligible pairs.

    n_bands=3 gives the early / middle / late thirds; other values split the
    sequence into that many equal relative-position bands.
    """
    keep = m["seen"]
    frac = m["pos"][keep] / m["len"][keep]
    rows = []
    for i in range(n_bands):
        lo, hi = i / n_bands, (i + 1) / n_bands
        sel = (frac >= lo) & (frac < hi if hi < 1.0 else frac <= hi)
        if sel.sum() == 0:
            continue
        name = (f"band {i + 1}/{n_bands} ({lo:.0%}-{hi:.0%} of their log)"
                if n_bands != 3 else
                ["early (first third)", "middle third",
                 "late (last third)"][i])
        rows.append(full_report(name, m["probs"][keep][sel],
                                m["labels"][keep][sel]))
    return rows


def make_template_csv(skill_names):
    """Empty import template using the real skill names of the dataset."""
    names = sorted(skill_names)
    return ("student_index,skill_name,question_id,is_transfer,correct\n"
            f"0,{names[0]},delayed-1,0,\n")


def validate_imported(df, skill_names, num_test_students):
    """Return list of human-readable problems; empty list means valid."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing column(s): {', '.join(missing)}"]
    if df["student_index"].isna().any():
        problems.append("student_index has empty values")
    elif not np.issubdtype(df["student_index"].dtype, np.number):
        problems.append("student_index must be an integer")
    elif ((df["student_index"] < 0) | (df["student_index"] >= num_test_students)).any():
        problems.append("student_index out of range for the test students")
    bad_skills = set(df["skill_name"].dropna()) - set(skill_names)
    if bad_skills:
        problems.append(f"unknown skill_name value(s): {sorted(bad_skills)[:5]}")
    for col, label in [("is_transfer", "is_transfer"), ("correct", "correct")]:
        vals = pd.to_numeric(df[col], errors="coerce")
        if not vals.isin([0, 1]).all():
            problems.append(f"{label} must be 0 or 1 (or empty for unfilled rows)")
    return problems


def agreement_report(df, model, test_seqs, skill_index):
    """For filled imported rows: observed accuracy vs model prediction.

    The model prediction uses the student's test-sequence history as-is; the
    imported question is treated as a new probe of that skill *after* the
    logged history. This compares two evidence sources; it does not retrain
    or update the model.
    """
    filled = df.dropna(subset=["correct"]).copy()
    filled["correct"] = filled["correct"].astype(int)
    filled["is_transfer"] = pd.to_numeric(filled["is_transfer"], errors="coerce").fillna(0).astype(int)
    rows = []
    for _, r in filled.iterrows():
        idx = int(r["student_index"])
        sid = skill_index.get(r["skill_name"])
        if sid is None:
            continue
        seq = test_seqs[idx]
        pred = _predict_next(model, seq["skills"], seq["correct"], sid)
        rows.append({"student_index": idx, "skill": r["skill_name"],
                     "transfer": r["is_transfer"], "predicted": pred,
                     "correct": int(r["correct"])})
    if not rows:
        return {"n": 0, "rows": [], "summary": []}
    df_r = pd.DataFrame(rows)
    summary = []
    for name, grp in [("all", df_r), ("in-context (transfer=0)", df_r[df_r.transfer == 0]),
                      ("transfer (transfer=1)", df_r[df_r.transfer == 1])]:
        if len(grp) == 0:
            continue
        summary.append({
            "group": name, "n": int(len(grp)),
            "mean_predicted": float(grp["predicted"].mean()),
            "observed_accuracy": float(grp["correct"].mean()),
            "gap": float(grp["predicted"].mean() - grp["correct"].mean()),
        })
    return {"n": int(len(df_r)), "rows": rows, "summary": summary,
            "warning": "small imported samples give noisy comparisons; "
                       "gap > 0 means the model was more optimistic than the assessment"}


@torch.no_grad()
def _predict_next(model, skills, correct, skill_id):
    skill_t = torch.tensor([skills], dtype=torch.long)
    corr_t = torch.tensor([correct], dtype=torch.float)
    valid_t = torch.ones_like(skill_t, dtype=torch.bool)
    logits = model(skill_t, corr_t, valid_t)
    t = len(skills) - 1
    return float(torch.sigmoid(logits[0, t, skill_id]))
