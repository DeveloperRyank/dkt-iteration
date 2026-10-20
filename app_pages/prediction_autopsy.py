"""Prediction autopsy: stress-test one model prediction with ablations."""

import altair as alt
import pandas as pd
import streamlit as st

from app_common import load_bundle, load_model
from autopsy import autopsy

st.header("Prediction Autopsy")
st.caption(
    "Pick one held-out interaction and look under the prediction: what the "
    "model said, what really happened, and how the number moves when each "
    "earlier answer is removed or flipped. These are diagnostic experiments "
    "— an interaction that changes the prediction is NOT proven to be the "
    "reason the model thought what it did."
)

bundle = load_bundle()
students = bundle["test_students"]
skill_names = bundle["skill_names"]

idx = st.selectbox(
    "Student (held-out)", range(len(students)),
    format_func=lambda i: students[i]["label"],
)
student = students[idx]
L = len(student["skills"])
target = st.slider("Which interaction to audit (2 = second question onward)",
                   1, L - 1, min(L - 1, 8))


@st.cache_data(show_spinner="Re-running the model once per ablation...")
def run_autopsy(student_idx, target_step):
    model, _ = load_model()
    s = students[student_idx]
    return autopsy(model, s["skills"], s["correct"], target_step)


r = run_autopsy(idx, target)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Topic", skill_names[r["target_skill"]])
c2.metric("Model predicted", f"{r['predicted_prob']:.2f}")
c3.metric("Actually", "correct" if r["actual_outcome"] else "wrong")
c4.metric("Prior observations of this skill",
          r["prior_same_skill_observations"])
if r["prior_same_skill_rate"] is not None:
    st.caption(
        f"Raw history for this skill before the audited question: "
        f"{r['prior_same_skill_rate']:.0%} correct over "
        f"{r['prior_same_skill_observations']} attempts."
    )
else:
    st.warning("This was the student's first question on this skill — the "
               "model has no per-skill history to lean on.",
               icon=":material/help_outline:")

wrong = (r["predicted_prob"] >= 0.5) != bool(r["actual_outcome"])
gap = abs(r["predicted_prob"] - r["actual_outcome"])
if wrong and gap > 0.3:
    st.error("Confidently wrong: the model's probability was far from the "
             "outcome. Inspect the ablations below.",
             icon=":material.report:")
elif wrong:
    st.warning("The prediction missed, though not by much.",
               icon=":material.warning:")

st.subheader("Sensitivity to earlier interactions")
st.caption(
    "'flip' = replace that correct/wrong with the opposite; 'drop' = remove "
    "the interaction entirely. Interactions changing the prediction by more "
    "than 0.05 are flagged. Bars show how much the predicted probability "
    "moved."
)
tbl = pd.DataFrame(r["interactions"])
tbl["skill_name"] = tbl["skill"].map(dict(enumerate(skill_names)))
view = tbl[["step", "skill_name", "correct", "delta_drop", "delta_flip",
            "flagged"]].copy()
st.dataframe(
    view, hide_index=True, width="stretch", height=280,
    column_config={
        "step": st.column_config.NumberColumn("question #", help="index in "
                                              "the student's sequence (0-based)"),
        "delta_drop": st.column_config.NumberColumn("Δ if dropped", format="%+.3f"),
        "delta_flip": st.column_config.NumberColumn("Δ if flipped", format="%+.3f"),
        "flagged": st.column_config.CheckboxColumn("sensitive"),
    },
)

top = tbl[tbl["flagged"]]
top = top.reindex(top["delta_flip"].abs().sort_values(ascending=False).index).head(12)
if len(top):
    chart = (
        alt.Chart(top)
        .mark_bar(color="#f59e0b")
        .encode(
            y=alt.Y("step:O", title="earlier question #"),
            x=alt.X("delta_flip:Q", title="prediction change if answer flipped"),
            tooltip=["step", "skill_name", "correct",
                     alt.Tooltip("delta_flip:Q", format="+.3f")],
        )
        .properties(height=240)
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Most sensitive flipped interactions. Amber = flagged at the "
               "0.05 threshold.")
else:
    st.info("No earlier interaction moves this prediction by more than 0.05 "
            "— the model's answer is robust to single edits of this "
            "student's history.")

st.caption(r["note"])
