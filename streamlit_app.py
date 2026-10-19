"""DKT student simulator (beginner-friendly copy).

Replays a real student's question history from the ASSISTments dataset and
shows, step by step, how the trained AI model updates its guess about whether
the student will get the NEXT question right.

Run with:  python -m streamlit run streamlit_app.py
"""

import gzip
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from data.dataset import collate_fn
from model.dkt import DKT

BUNDLE_PATH = "outputs/demo_bundle.json.gz"
CHECKPOINT_PATH = "outputs/model_assistments.pt"

AUC_HELP = (
    "ROC-AUC measures how well the model ranks likely-correct answers above "
    "likely-wrong ones. 0.5 = coin flip, 1.0 = perfect. See the exact score "
    "of the model you are using at the bottom of this page."
)
MODEL_HELP = (
    "The model is a type of neural network (an LSTM) that has seen thousands "
    "of students' histories during training. It reads this student's past "
    "answers and estimates the chance they get the next question right."
)
BASELINE_HELP = (
    "A simple, fair comparison: the historical success rate for this topic, "
    "averaged over all students in the training data. It ignores what THIS "
    "student has done recently. If the AI beats this number, it proves the "
    "AI is really using the student's individual history."
)


@st.cache_resource
def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = DKT(ckpt["num_skills"], ckpt["hidden_size"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@st.cache_data
def load_bundle():
    with gzip.open(BUNDLE_PATH, "rt", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def predict_sequence(student):
    """Return (probs, labels, skills) for interactions 2..end:
    probs[i] is the prediction for interaction i+2, made after observing
    interactions 1..i+1."""
    model, _ = load_model()
    seq = {"skills": student["skills"], "correct": student["correct"]}
    batch = collate_fn([seq])
    with torch.no_grad():
        logits = model(**batch)
    probs = torch.sigmoid(logits)[0, :-1].numpy()  # (L-1, num_skills)
    skills = np.array(student["skills"][1:])
    labels = np.array(student["correct"][1:])
    return probs[np.arange(len(skills)), skills], labels, skills


st.set_page_config(page_title="Can AI predict student answers?", layout="wide")

bundle = load_bundle()
_, ckpt = load_model()
students = bundle["test_students"]
skill_names = bundle["skill_names"]
baseline_rates = np.array(bundle["baseline_rates"])

st.title("Can AI predict the next question a student gets right?")
st.caption(
    "A hands-on demo of Deep Knowledge Tracing (Piech et al., 2015). "
    "Watch an AI model read a real student's past answers and update its "
    "guess after every question."
)

with st.expander("How this works (2-minute version)"):
    st.markdown(
        """
1. **The data.** Every row is one question attempt by a real student on the
   ASSISTments tutoring website: which math topic the question tested, and
   whether the answer was right or wrong.
2. **The AI.** A neural network reads the student's answers one at a time,
   like following a conversation. After each answer it updates a guess:
   *"what is the chance this student gets the next question right?"*
3. **The test.** We show the AI only the questions the student already
   answered -- never the answer that comes next. Then we compare the AI's
   guess with what really happened. The students shown here were never part
   of the AI's training, so this is an honest test.

**Fair comparison:** the *simple average* line is just the topic's overall
success rate across all students. The AI wins if it does better by using
*this* student's recent story -- for example noticing they just got three
in a row wrong.
"""
    )

with st.sidebar:
    st.header("Pick a student")
    idx = st.selectbox(
        "Student (real, never seen during training)",
        range(len(students)),
        format_func=lambda i: students[i]["label"],
    )
    if "student" not in st.session_state or st.session_state.student != idx:
        st.session_state.student = idx
        st.session_state.step = 1
    st.button("Start over", icon=":material/replay:",
              on_click=lambda: st.session_state.update(step=1))
    st.caption(
        f"The AI knows {bundle['num_skills']} math topics. "
        f"{len(students)} test students to explore."
    )

student = students[idx]
total = len(student["skills"]) - 1
step = min(st.session_state.step, total)

probs, labels, skills = predict_sequence(student)
base = baseline_rates[skills]

done = step >= total
if not done:
    if st.button("Reveal next question", type="primary",
                 icon=":material/skip_next:"):
        st.session_state.step = step + 1
        st.rerun()

if done:
    st.subheader(f"That's the end: all {total} questions replayed")
else:
    st.subheader(
        f"Question {min(step + 1, total)} of {total} — "
        f"the AI has seen the first {step}"
    )

t = min(step, total) - 1  # index of the current prediction
col1, col2, col3, col4 = st.columns(4)
col1.metric("Topic of this question", skill_names[skills[t]])
col2.metric("AI: chance they get it right", f"{probs[t]:.0%}", help=MODEL_HELP)
col3.metric("Simple average guess", f"{base[t]:.0%}", help=BASELINE_HELP)
outcome = "Got it right" if labels[t] == 1 else "Got it wrong"
col4.metric("What really happened", outcome)

if probs[t] > base[t] + 0.15:
    st.info("The AI is more confident than the simple average here.")
elif probs[t] < base[t] - 0.15:
    st.warning("The AI is *less* confident than the simple average here -- "
               "it noticed something worrying in this student's recent answers.")

history = pd.DataFrame({
    "question": np.arange(1, step + 1),
    "AI prediction": probs[:step],
    "Simple average": base[:step],
    "actual_correct": labels[:step],
})
long = history.melt(id_vars=["question", "actual_correct"],
                    value_vars=["AI prediction", "Simple average"],
                    var_name="who is guessing", value_name="chance correct")
chart = (
    alt.Chart(long)
    .mark_line()
    .encode(
        x=alt.X("question:Q", title="question number"),
        y=alt.Y("chance correct:Q", title="chance of a correct answer",
                scale=alt.Scale(domain=[0, 1])),
        color="who is guessing:N",
    )
)
hits = history[history["actual_correct"] == 1]
points = (
    alt.Chart(hits)
    .mark_circle(color="green", size=36)
    .encode(x="question:Q", y=alt.Y("actual_correct:Q", title=None))
)
st.altair_chart(chart + points, height=340)
st.caption(
    "Each line is a guess about the NEXT question. Green dots along the "
    "bottom mark questions the student actually got right. When the AI line "
    "follows the dots better than the flat average line, it is learning "
    "this student's story."
)

with st.container(border=True):
    st.subheader("What the AI thinks this student knows, right now")
    st.caption(
        "Chance of a correct answer on each topic if a question on it came "
        "next, given only the answers revealed so far. Topics the student "
        "just practiced should drift up or down as evidence piles in."
    )
    model, _ = load_model()
    batch = collate_fn([{"skills": student["skills"][:step],
                         "correct": student["correct"][:step]}])
    with torch.no_grad():
        belief = torch.sigmoid(model(**batch))[0, -1]
    order = torch.argsort(belief, descending=True)[:12]
    belief_df = pd.DataFrame({
        "topic": [skill_names[i] for i in order],
        "chance correct": belief[order].numpy(),
    })
    st.bar_chart(belief_df, x="topic", y="chance correct", height=300)

st.divider()
st.caption(
    f"Model details: trained on ASSISTments 2009–2010, {ckpt.get('epochs', '?')} "
    f"training epochs, test score (ROC-AUC) {ckpt.get('test_auc', float('nan')):.3f} "
    f"(0.5 = coin flip, 1.0 = perfect). "
    "The simple average baseline scores about 0.61 on the same test."
)
