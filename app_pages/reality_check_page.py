"""Mastery reality check: proxy analysis + delayed/transfer import path."""

import io

import pandas as pd
import streamlit as st

from app_common import load_bundle, load_model
from reality_check import (TEMPLATE_DOC, agreement_report,
                           collect_with_positions, make_template_csv,
                           position_band_report, validate_imported)

st.header("Mastery Reality Check")
st.warning(
    "Can next-answer prediction tell us whether a student really learned? "
    "ASSISTments 2009–2010 contains **no delayed retests and no labelled "
    "transfer questions**, so this page offers (1) an honest proxy and "
    "(2) an import path for suitable data you collect later. Prediction "
    "accuracy alone is never proof of durable learning.",
    icon=":material/science:",
)

bundle = load_bundle()
skill_names = bundle["skill_names"]
students = bundle["test_students"]


@st.cache_data(show_spinner="Scoring every held-out pair...")
def band_report(n_bands):
    model, _ = load_model()
    m = collect_with_positions(model, students)
    return position_band_report(m, n_bands)


st.subheader("1. Sequence-position proxy (real computed numbers)")
st.caption(
    "Split each student's logged history into bands by how far through it a "
    "question was. Later questions come later in the student's time on the "
    "site and have more history behind them, so a score that holds up (or "
    "improves) in the last band is *suggestive* of retention. It is NOT a "
    "controlled delayed-retention study, and it never will be on this "
    "dataset."
)
n_bands = st.slider("Cut each student's history into how many bands?",
                    2, 8, 3,
                    help="More bands means fewer questions per band, so the "
                         "per-band numbers get noisier.")
rows = band_report(n_bands)
st.dataframe(pd.DataFrame(rows)[["name", "n", "auc", "acc", "log_loss",
                                 "brier", "ece"]].round(4),
             hide_index=True, width="stretch")
aucs = [r["auc"] for r in rows]
if len(rows) >= 2:
    trend = aucs[-1] - aucs[0]
    st.caption(
        f"AUC goes from {aucs[0]:.3f} in the first band to {aucs[-1]:.3f} in "
        f"the last ({trend:+.3f}). "
        + ("Prediction holds up as history grows — consistent with the model "
           "tracking something stable, but still not proof of durable "
           "learning."
           if trend > -0.02 else
           "Prediction gets worse on later questions, so the model leans on "
           "early-sequence patterns that stop applying.")
    )
    st.warning("`n` is the number of scored pairs in the band. A band with a "
               "few hundred pairs and an AUC swing of 0.01 is noise, not a "
               "finding.", icon=":material/warning:")

st.subheader("2. Import delayed / transfer assessments")
st.markdown(TEMPLATE_DOC)
st.download_button(
    "Download CSV template",
    data=make_template_csv(skill_names),
    file_name="assessments_template.csv",
    mime="text/csv",
    icon=":material/download:",
)

uploaded = st.file_uploader("Paste results back as CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    problems = validate_imported(df, skill_names, len(students))
    if problems:
        st.error("Your file has problems — fix and re-upload:\n\n"
                 + "\n".join(f"- {p}" for p in problems))
    else:
        model, _ = load_model()
        rep = agreement_report(df, model, students,
                              {n: i for i, n in enumerate(skill_names)})
        if rep["n"] == 0:
            st.info("No filled rows found (the 'correct' column is empty).")
        else:
            st.success(f"Compared {rep['n']} imported assessments against "
                       f"model predictions.")
            st.dataframe(pd.DataFrame(rep["summary"]).round(4),
                         hide_index=True, width="stretch")
            st.caption(rep["warning"])
            with st.expander("Per-row detail"):
                st.dataframe(pd.DataFrame(rep["rows"]).round(4),
                             hide_index=True, width="stretch")
