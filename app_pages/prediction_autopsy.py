"""Prediction Autopsy — poke at one prediction instead of just reading it.

Two tools: a live editor where you flip or delete earlier answers and watch
the model's guess move instantly, and a full sweep that tests every earlier
answer for sensitivity.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app_common import load_bundle, load_model
from autopsy import autopsy, predict_prob

st.header("Prediction Autopsy")
st.caption(
    "The model just made a guess about one question. Here you get to argue "
    "with it: change the student's past and watch the number move, then "
    "sweep every past answer to see which ones the model actually leans on. "
    "These are diagnostic experiments — a question that moves the number is "
    "NOT proven to be the *reason* the model guessed what it did."
)

bundle = load_bundle()
students = bundle["test_students"]
skill_names = bundle["skill_names"]
baseline_rates = np.array(bundle["baseline_rates"])

idx = st.selectbox(
    "Choose a student (held out, never seen in training)", range(len(students)),
    format_func=lambda i: students[i]["label"],
    key="autopsy_student",
)
student = students[idx]
L = len(student["skills"])
target = st.slider("Which question do we audit?", 1, L - 1, min(L - 1, 8),
                   help="Question 2 onward: for the very first question the "
                        "model has no history to reason from.",
                   key="autopsy_target")

model, _ = load_model()

base_prob = predict_prob(model, student["skills"], student["correct"], target)
target_skill = student["skills"][target]
outcome = student["correct"][target]

c1, c2, c3 = st.columns(3)
c1.metric("Topic of the audited question", skill_names[target_skill])
c2.metric("Model's guess", f"{base_prob:.0%}",
          help="Chance the model gives this student of answering right, "
               "using only the answers before this question.")
c3.metric("What really happened", "Right" if outcome else "Wrong")

prior_n = sum(1 for k in range(target) if student["skills"][k] == target_skill)
st.caption(
    f"Before this question the model had seen {prior_n} answer(s) on this "
    f"topic. The class average on this topic is "
    f"{baseline_rates[target_skill]:.0%} — the interesting question is how "
    f"far this student's own story pulls the model away from that number."
)
if abs(base_prob - baseline_rates[target_skill]) < 0.02:
    st.info("The model is basically repeating the class average here. "
            "Try the editor below — if nothing moves the number, the model "
            "is ignoring this student's history for this topic.")

st.subheader("Try it: rewrite the student's past")
st.caption(
    "Pick the questions you want to change and say what should happen to "
    "them. A new guess appears below as soon as you do — the model re-reads "
    "the edited history from scratch."
)

opts = list(range(target))
labels = {k: f"Q{k + 1} · {skill_names[student['skills'][k]]} · "
             f"{'right' if student['correct'][k] else 'wrong'}"
          for k in opts}
edits = st.multiselect(
    "Questions to change", opts,
    format_func=lambda k: labels[k],
    placeholder="Pick one or more past questions...",
)
mode = st.segmented_control(
    "What to do to them", ["flip", "remove"], default="flip",
    help="flip = right becomes wrong and vice versa. "
         "remove = the question never happened.",
)

if edits:
    prefix_s = []
    prefix_c = []
    for k in range(target):
        if mode == "remove" and k in edits:
            continue
        c_val = student["correct"][k]
        if mode == "flip" and k in edits:
            c_val = 1 - c_val
        prefix_s.append(student["skills"][k])
        prefix_c.append(c_val)

    if not prefix_s:
        st.info("You removed every question the model had — there is no "
                "history left to reason from, so a guess is meaningless here. "
                "This is exactly the situation the model is worst at: "
                "question 1 of a brand-new student.")
    else:
        # the audited question sits at the end of the edited prefix; its own
        # placeholder outcome is never read (the guess is made before it)
        edited_prob = predict_prob(model, prefix_s + [target_skill],
                                   prefix_c + [0], len(prefix_s))
        moved = edited_prob - base_prob
        d1, d2 = st.columns([1, 2])
        d1.metric("New guess", f"{edited_prob:.0%}",
                  f"{moved:+.0%} vs the real history")
        if abs(moved) < 0.01:
            d2.caption("Barely moved. The model did not need those answers — "
                       "it was confident from the rest of the history.")
        elif moved > 0:
            d2.caption(f"Your edit made the model *more* optimistic by "
                       f"{moved:.0%}.")
        else:
            d2.caption(f"Your edit made the model *less* optimistic by "
                       f"{abs(moved):.0%}.")
        st.caption(
            f"Edited history has {len(prefix_s)} questions before the audited "
            f"one (originally {target}). Removing questions also shortens the "
            f"story the model can read — that is part of what you are testing."
        )
else:
    st.caption("Select at least one past question to see the effect.")

st.subheader("Sweep: which past answers does the model lean on?")
st.caption(
    "Tests every earlier question twice — once deleted, once flipped — and "
    "reports how much the guess moves each time. Takes a second per "
    "question, so it only runs when you press the button."
)
if "autopsy_sweep" not in st.session_state:
    st.session_state.autopsy_sweep = None
run = st.button("Run the full sweep", type="primary",
                icon=":material/search:")
if run or st.session_state.autopsy_sweep is not None:
    if run:
        with st.spinner("Re-running the model once per edit..."):
            st.session_state.autopsy_sweep = (idx, target,
                                              autopsy(model, student["skills"],
                                                      student["correct"],
                                                      target))
    a_idx, a_target, r = st.session_state.autopsy_sweep
    if (a_idx, a_target) != (idx, target):
        st.info("You changed the student or the question — press "
               "\"Run the full sweep\" again for the new choice.")
    else:
        tbl = pd.DataFrame(r["interactions"])
        tbl["skill_name"] = tbl["skill"].map(dict(enumerate(skill_names)))
        view = tbl[["step", "skill_name", "correct", "delta_drop",
                    "delta_flip", "flagged"]].copy()
        st.dataframe(
            view, hide_index=True, width="stretch", height=260,
            column_config={
                "step": st.column_config.NumberColumn("question #",
                                                      help="0-based index in "
                                                           "the sequence"),
                "correct": st.column_config.CheckboxColumn("was right"),
                "delta_drop": st.column_config.NumberColumn("Δ if removed",
                                                            format="%+.3f"),
                "delta_flip": st.column_config.NumberColumn("Δ if flipped",
                                                            format="%+.3f"),
                "flagged": st.column_config.CheckboxColumn("sensitive"),
            },
        )
        top = tbl[tbl["flagged"]]
        top = top.reindex(
            top["delta_flip"].abs().sort_values(ascending=False).index).head(12)
        if len(top):
            chart = (
                alt.Chart(top)
                .mark_bar(color="#f59e0b")
                .encode(
                    y=alt.Y("step:O", title="earlier question #"),
                    x=alt.X("delta_flip:Q",
                            title="guess change if that answer is flipped"),
                    tooltip=["step", "skill_name", "correct",
                             alt.Tooltip("delta_flip:Q", format="+.3f")],
                )
                .properties(height=240)
            )
            st.altair_chart(chart, width="stretch")
            st.caption("Amber bars = flipping that answer moves the guess by "
                       "more than 0.05. Long bar = the model really needed it.")
        else:
            st.info("No single edit moves this prediction by more than 0.05 — "
                    "the model's guess here comes from the whole history, "
                    "not one question.")
        st.caption(r["note"])
