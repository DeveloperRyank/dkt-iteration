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
def band_report():
    model, _ = load_model()
    m = collect_with_positions(model, students)
    return position_band_report(m)


st.subheader("1. Sequence-position proxy (real computed numbers)")
st.caption(
    "Predictions grouped by where they fall in the student's logged "
    "sequence. Later positions come later in the school year and have more "
    "history behind them — suggestive of retention, NOT a controlled "
    "delayed-assessment study."
)
rows = band_report()
st.dataframe(pd.DataFrame(rows)[["name", "n", "auc", "acc", "log_loss",
                                 "brier", "ece"]].round(4),
             hide_index=True, width="stretch")

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
