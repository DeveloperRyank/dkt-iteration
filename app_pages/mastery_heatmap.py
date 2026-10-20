"""Mastery heatmap: scrub through a student's history and watch the model's
per-skill beliefs change in real time."""

import numpy as np
import pandas as pd
import streamlit as st

from app_common import (MIN_EVIDENCE, beliefs_upto, load_bundle,
                        observed_by_skill)

st.header("Mastery Heatmap")
st.caption(
    "One student, every topic: what the MODEL estimates vs what the student "
    "actually did. Drag the slider to travel through the student's history — "
    "this is not a frozen report."
)

bundle = load_bundle()
students = bundle["test_students"]
skill_names = bundle["skill_names"]
S = bundle["num_skills"]

idx = st.selectbox(
    "Student (held-out, never seen during training)",
    range(len(students)),
    format_func=lambda i: students[i]["label"],
)
student = students[idx]
L = len(student["skills"])
counts, observed = observed_by_skill(student, S)

step = st.slider("Show the model's mind after how many questions?",
                 0, L, L)
st.caption("0 = before seeing anything: the model starts at 50% everywhere.")
belief = beliefs_upto(student["skills"], student["correct"], step)

attempts_so_far = np.zeros(S, dtype=int)
win_skills = np.array(student["skills"][:step])
win_correct = np.array(student["correct"][:step], dtype=float)
for s in win_skills:
    attempts_so_far[s] += 1

only_attempted = st.toggle("Only topics touched so far", value=True)
min_n = st.slider("Flag topics with fewer than this many attempts as thin "
                  "evidence", 1, 10, MIN_EVIDENCE)

rows = []
for s in range(S):
    n = int(attempts_so_far[s])
    if only_attempted and n == 0 and step > 0:
        continue
    rows.append({
        "topic": skill_names[s],
        "model estimate now": float(belief[s]),
        "observed accuracy (their part)": None if n == 0
            else float(win_correct[win_skills == s].mean()),
        "attempts so far": n,
        "evidence": "thin" if 0 < n < min_n else ("none" if n == 0 else "ok"),
    })
df = pd.DataFrame(rows).sort_values("model estimate now", ascending=False)
if df.empty:
    st.info("No topics match — slide forward or turn off the filter.")
    st.stop()

st.dataframe(
    df,
    hide_index=True,
    width="stretch",
    height=min(720, 38 * len(df)),
    column_config={
        "model estimate now": st.column_config.ProgressColumn(
            "model estimate now", min_value=0, max_value=1, format="%.2f",
            help="Predicted: probability a question on this topic, asked "
                 "RIGHT NOW, would be answered correctly."),
        "observed accuracy (their part)": st.column_config.ProgressColumn(
            "observed accuracy (their part)", min_value=0, max_value=1,
            format="%.2f",
            help="Measured: the student's own hit rate on this topic in the "
                 "shown part of their history."),
        "evidence": st.column_config.TextColumn(
            "evidence", help="'thin' = too few attempts to trust the numbers"),
    },
)
st.caption(
    "Model estimate and observed accuracy are different kinds of numbers — "
    "one is a prediction, one is a measurement. Gaps are interesting: after "
    "a streak of wrongs, watch the estimate drop before the raw average "
    "catches up."
)

st.subheader("Drill into one topic")
attempted = [s for s in range(S) if counts[s] > 0]
skill_sel = st.selectbox(
    "Topic", attempted,
    format_func=lambda s: f"{skill_names[s]} ({counts[s]} observations)",
)
hist = [(t + 1, c) for t, (sk, c) in enumerate(zip(student["skills"],
                                                   student["correct"]))
        if sk == skill_sel]
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Observed accuracy (whole history)",
              f"{np.mean([o for _, o in hist]):.0%}")
    st.metric("Model estimate (at the slider position)",
              f"{belief[skill_sel]:.2f}",
              help=f"Based on the first {step} question(s) only. Move the "
                   "slider to change it.")
    st.metric("Observations", len(hist))
    if len(hist) < MIN_EVIDENCE:
        st.warning(f"Fewer than {MIN_EVIDENCE} observations — treat both "
                   "numbers as unreliable.", icon=":material/warning:")
with col2:
    st.dataframe(pd.DataFrame(hist, columns=["question #", "outcome"]),
                 hide_index=True, width="stretch", height=260)
