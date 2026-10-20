"""Student simulator: the hands-on page. Press buttons, watch the model
change its mind — either replay a real student or answer questions yourself."""

import numpy as np
import pandas as pd
import streamlit as st

from app_common import beliefs_upto, load_bundle, predict_sequence

st.header("Student Simulator")
st.caption(
    "This is the playground page. Nothing here is a static report: every "
    "button changes what the model has seen, and you see its guess move."
)

mode = st.segmented_control(
    "What do you want to do?",
    options=["Replay a real student", "Be the student yourself"],
    default="Replay a real student",
)

bundle = load_bundle()
skill_names = bundle["skill_names"]
rates = np.array(bundle["baseline_rates"])
S = bundle["num_skills"]

if mode == "Replay a real student":
    students = bundle["test_students"]
    idx = st.selectbox(
        "Pick a real student (held out from training)",
        range(len(students)),
        format_func=lambda i: students[i]["label"],
    )
    if st.session_state.get("sim_student") != idx:
        st.session_state.sim_student = idx
        st.session_state.sim_step = 0
    student = students[idx]
    total = len(student["skills"]) - 1
    step = min(st.session_state.sim_step, total)

    probs, labels, skills = predict_sequence(student)

    c1, c2 = st.columns([2, 1])
    with c1:
        if step >= total:
            st.success(f"All {total} questions replayed — this student's "
                       "story ends here.")
            if st.button("Start over", icon=":material/replay:"):
                st.session_state.sim_step = 0
                st.rerun()
        else:
            if st.button(f"Reveal question {step + 1} of {total}",
                         type="primary", icon=":material/skip_next:"):
                st.session_state.sim_step = step + 1
                st.rerun()
            st.caption("The model has seen the first "
                       f"{step} answers. Press the button to see what "
                       "question comes next and whether the student got it right.")
    with c2:
        if step > 0:
            t = step - 1
            m1 = st.metric("Topic now", skill_names[skills[t]] if t < len(skills) else "-")
            outcome = "correct" if labels[t] == 1 else "wrong"
            m2 = st.metric("Model said", f"{probs[t]:.0%}")
            m3 = st.metric("Student was", outcome,
                           delta="match" if (probs[t] >= 0.5) == bool(labels[t])
                                 else "miss")

    shown = max(step, 1)
    df = pd.DataFrame({
        "question": np.arange(1, shown + 1),
        "model guess": probs[:shown],
        "class average guess": rates[skills[:shown]],
        "actually correct": labels[:shown],
    })
    st.caption("Model guess (teal) vs the boring class-average guess (orange). "
               "Green dots on the bottom row are questions the student really "
               "got right. When the teal line dives before a wrong answer, the "
               "model is reading THIS student, not the crowd.")
    st.line_chart(df, x="question", y=["model guess", "class average guess"],
                  height=280, color=["#2dd4bf", "#f59e0b"])

    st.subheader("The model's mind, topic by topic")
    belief = beliefs_upto(student["skills"], student["correct"], step)
    top = np.argsort(belief)[::-1][:12]
    bdf = pd.DataFrame({
        "topic": [skill_names[s] for s in top],
        "chance next question is right": belief[top],
    })
    st.bar_chart(bdf, x="topic", y="chance next question is right", height=260)
    st.caption("Recomputed from only the answers revealed so far — press "
               "reveal and watch the bars move.")

else:  # Be the student yourself
    st.markdown(
        "Pick a math topic, and the model first shows how it thinks you will "
        "do. Then you decide what really happened — right or wrong — and the "
        "model reacts. Your invented history is processed the same way as "
        "real students': **nothing is saved, nothing is tracked.**"
    )
    hist = st.session_state.setdefault("my_hist", {"skills": [], "correct": []})

    if st.button("Clear my history", icon=":material/delete_sweep:"):
        st.session_state.my_hist = {"skills": [], "correct": []}
        st.rerun()

    if not hist["skills"]:
        st.info("No history yet, so the model starts with a coin flip (50%) "
                "for every topic. Answer a few questions and its guesses "
                "will sharpen.", icon=":material/lightbulb:")

    col_pick, col_guess = st.columns(2)
    with col_pick:
        if st.button("Surprise me", icon=":material/shuffle:"):
            st.session_state.my_topic = int(
                np.random.default_rng().integers(S))
            st.rerun()
        topic = st.selectbox("Next topic for you", range(S),
                             format_func=lambda s: skill_names[s],
                             key="my_topic")
    with col_guess:
        belief = beliefs_upto(hist["skills"], hist["correct"], len(hist["skills"]))
        st.markdown(f"Topic in play: **{skill_names[topic]}**")
        st.metric("Model: your chance of getting this right",
                  f"{belief[topic]:.0%}",
                  delta=f"vs {rates[topic]:.0%} class average",
                  delta_color="off")
        st.caption("This is a guess about behaviour, not about what you truly "
                   "know.")

    b1, b2, _ = st.columns([1, 1, 2])
    answered = None
    if b1.button("I got it RIGHT", type="primary", icon=":material/check:"):
        answered = 1
    if b2.button("I got it WRONG", icon=":material/close:"):
        answered = 0
    if answered is not None:
        before = belief[topic]
        hist["skills"].append(topic)
        hist["correct"].append(answered)
        after = beliefs_upto(hist["skills"], hist["correct"],
                            len(hist["skills"]))
        st.session_state.my_last = (topic, answered, before, float(after[topic]))
        # re-run so the headline metric shows the post-answer belief
        st.rerun()

    if "my_last" in st.session_state:
        topic_i, ans, before, after = st.session_state.my_last
        moved = after - before
        st.markdown(
            f"You got **{'right' if ans else 'wrong'}** on "
            f"**{skill_names[topic_i]}**. The model's guess for that topic "
            f"moved **{moved:+.0%}** (now {after:.0%})."
        )
        if abs(moved) < 0.02:
            st.caption("Barely moved — the model often waits for several "
                       "similar answers before changing its mind.")

    if hist["skills"]:
        table = pd.DataFrame({
            "topic": [skill_names[s] for s in hist["skills"]],
            "outcome": ["right" if c else "wrong" for c in hist["correct"]],
        })
        st.dataframe(table, hide_index=True, width="stretch", height=220)
        acc = float(np.mean(hist["correct"]))
        st.progress(acc, text=f"Your observed accuracy so far: {acc:.0%} "
                              f"over {len(hist['correct'])} answers")
