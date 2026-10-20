"""Mastery heatmap: model estimate vs observed accuracy, per skill."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_common import (LOW_EVIDENCE_NOTE, MIN_EVIDENCE, load_bundle,
                        observed_by_skill, skill_beliefs)

st.header("Mastery Heatmap")
st.caption(
    "For one student: what the model estimates (probability a next question "
    "on each skill would be answered correctly) side by side with what the "
    "student actually did. These are different things — the model estimate "
    "is an output of a neural network, not a measurement of knowledge."
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
belief = skill_beliefs(student)
counts, observed = observed_by_skill(student, S)

show_mask = st.toggle("Only skills this student has actually attempted",
                      value=True)
rows = []
for s in range(S):
    if show_mask and counts[s] == 0:
        continue
    rows.append({
        "skill": skill_names[s],
        "model estimate": float(belief[s]),
        "observed accuracy": None if counts[s] == 0 else float(observed[s]),
        "observations": int(counts[s]),
        "evidence": "thin" if counts[s] < MIN_EVIDENCE else "ok",
    })
df = pd.DataFrame(rows).sort_values("model estimate", ascending=False)

table = df.copy()
table["observed accuracy"] = table["observed accuracy"].astype(float)
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "model estimate": st.column_config.ProgressColumn(
            "model estimate", min_value=0, max_value=1, format="%.2f"),
        "observed accuracy": st.column_config.ProgressColumn(
            "observed accuracy", min_value=0, max_value=1, format="%.2f"),
        "evidence": st.column_config.TextColumn(
            "evidence", help=f"'thin' = fewer than {MIN_EVIDENCE} observations"),
    },
)
if (df["evidence"] == "thin").any():
    st.warning(LOW_EVIDENCE_NOTE, icon=":material/warning:")

st.subheader("Heatmap view")
long = df.melt(id_vars=["skill", "observations", "evidence"],
               value_vars=["model estimate", "observed accuracy"],
               var_name="estimate", value_name="value").dropna(subset=["value"])
hm = (
    alt.Chart(long)
    .mark_rect()
    .encode(
        y=alt.Y("skill:N", sort=list(df["skill"]), title=None),
        x=alt.X("estimate:N", title=None, axis=alt.Axis(orient="top")),
        color=alt.Color("value:Q", scale=alt.Scale(scheme="tealblues",
                                                   domain=[0, 1]),
                        legend=alt.Legend(title="probability")),
        text=alt.Text("value:Q", format=".2f"),
        tooltip=["skill", "estimate", "value", "observations", "evidence"],
    )
    .properties(height=max(200, 18 * len(df)))
)
st.altair_chart(hm, width="stretch")
st.caption(
    "The two columns must not be conflated: 'model estimate' is what the "
    "LSTM predicts next; 'observed accuracy' is the student's raw record "
    "in their (held-out) history. Differences are informative — e.g. the "
    "model may stay optimistic after a slump that the raw average lags."
)

st.subheader("Drill into one skill")
attempted = [s for s in range(S) if counts[s] > 0]
skill_sel = st.selectbox(
    "Skill", attempted,
    format_func=lambda s: f"{skill_names[s]} ({counts[s]} observations)",
)
hist = [(t, c) for t, (sk, c) in enumerate(zip(student["skills"],
                                               student["correct"]))
        if sk == skill_sel]
hist_df = pd.DataFrame(hist, columns=["question # in sequence", "outcome"])
hist_df["outcome"] = hist_df["outcome"].map({1: "correct", 0: "wrong"})
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Observed accuracy",
              f"{np.mean([o for _, o in hist]):.0%}",
              help="Measured: fraction correct on this skill in this "
                   "student's logged history.")
    st.metric("Model estimate (after full history)", f"{belief[skill_sel]:.2f}",
              help="Predicted: the LSTM's probability if a question on this "
                   "skill came next.")
    st.metric("Observations", len(hist))
    if len(hist) < MIN_EVIDENCE:
        st.warning(f"Fewer than {MIN_EVIDENCE} observations — treat both "
                   "numbers as unreliable.", icon=":material/warning:")
with col2:
    st.dataframe(hist_df, hide_index=True, width="stretch")
