"""Overview: headline numbers, plots, and a two-minute explanation."""

import altair as alt
import pandas as pd
import streamlit as st

from app_common import load_bundle, load_comparison, load_model

st.header("Overview")
st.caption(
    "Deep Knowledge Tracing (Piech et al., 2015) reimplemented from scratch: "
    "an LSTM reads each student's question-by-question history and predicts "
    "the chance they get the next question right. Every number here comes "
    "from a real run on held-out students."
)

model, ckpt = load_model()
bundle = load_bundle()
comp = load_comparison()

auc = ckpt.get("test_auc", float("nan"))
base_row = next((m for m in (comp or {}).get("models", [])
                 if m["name"].startswith("per-skill")), None)
base_auc = base_row["auc"] if base_row else 0.6065
n_pairs = comp["n_pairs"] if comp else "?"

c1, c2, c3, c4 = st.columns(4)
c1.metric("DKT test ROC-AUC", f"{auc:.3f}",
          help="Ranking quality of predicted probabilities on held-out "
               "pairs. 0.5 = coin flip, 1.0 = perfect.")
c2.metric("Simple baseline AUC", f"{base_auc:.3f}", delta=None,
          help="Historical per-skill success rate, ignoring each student's "
               "own story.")
c3.metric("Signal from history", f"+{auc - base_auc:.3f}",
          help="How much of the prediction comes from THIS student's "
               "recent answers rather than population averages.")
c4.metric("Held-out test pairs", f"{n_pairs:,}" if isinstance(n_pairs, int) else n_pairs)

st.subheader("How this works (2-minute version)")
st.markdown(
    """
1. **The data.** Every row is one question attempt by a real student on the
   ASSISTments tutoring site: which math topic was tested, and whether the
   answer was right or wrong.
2. **The model.** The LSTM reads answers one at a time and, after each one,
   updates a guess about the *next* question. It never sees the answer it is
   being scored on.
3. **The test.** These students were held out entirely (splits are by
   student, never by question), so the scores are honest out-of-sample
   numbers.
"""
)

left, right = st.columns(2)
with left:
    st.markdown("**Training loss** (lower is better)")
    st.image("outputs/loss_assistments.png")
with right:
    st.markdown("**ROC curves on held-out pairs**")
    st.image("outputs/roc_assistments.png")

st.warning(
    "These metrics measure response prediction, not understanding. A model "
    "that predicts next-answer probabilities well has NOT been shown to "
    "measure a student's true knowledge — see the Reality Check page.",
    icon=":material/school:",
)
