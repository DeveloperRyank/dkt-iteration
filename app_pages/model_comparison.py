"""Model comparison: DKT vs BKT vs baselines on identical test pairs."""

import altair as alt
import pandas as pd
import streamlit as st

from app_common import load_comparison, load_model

st.header("Model Comparison")
st.caption(
    "All predictors are scored on exactly the same eligible held-out test "
    "pairs (each skill's first observation excluded, because no model can "
    "predict a skill it has never seen)."
)

comp = load_comparison()
if comp is None:
    st.info("No comparison file yet. Run `python compare.py` to generate "
            "outputs/comparison.json — the model needs to be trained first "
            "(`python train.py --data assistments --save ...`).")
    st.stop()

models = pd.DataFrame(comp["models"])
show = models[["name", "n", "auc", "acc", "log_loss", "brier", "ece"]].copy()
for col in ("auc", "acc", "log_loss", "brier", "ece"):
    show[col] = show[col].round(4)
st.dataframe(show, hide_index=True, width="stretch")

with st.expander("What each metric actually measures"):
    st.markdown(
        """
| metric | question it answers | notes |
|---|---|---|
| ROC-AUC | are likely-correct pairs ranked above likely-wrong ones? | ignores calibration entirely |
| Accuracy | right call at the fixed 0.5 threshold? | sensitive to class balance |
| Log loss / Brier | how close are the probabilities themselves? | reward honest uncertainty |
| ECE | do "70% predictions" happen 70% of the time? | average gap per probability bin |

High response-prediction accuracy does **not** establish genuine mastery —
it shows the model can predict what a student will do next, which is only
weakly related to what the student knows.
"""
    )

st.subheader("Reliability diagrams")
st.caption(
    "Each bar: predictions in that probability bin (x-axis) vs how often "
    "they were actually correct (y-axis). Bars on the grey diagonal are "
    "perfectly calibrated; bars below the line are overconfident. "
    "Hover for sample counts — thin bins can be misleading."
)
names = [m["name"] for m in comp["models"]]
chosen = st.multiselect("Compare models", names, default=names[:2])
rows = []
for name in chosen:
    for r in comp["reliability"][name]:
        rows.append({"model": name, **r})
rel = pd.DataFrame(rows)
if rel.empty:
    st.caption("No populated bins.")
else:
    bars = (
        alt.Chart(rel)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("mean_predicted:Q", title="mean predicted probability",
                    axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("observed_rate:Q", title="observed proportion correct",
                    axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
            color="model:N",
            tooltip=["model", "bin", "count",
                     alt.Tooltip("mean_predicted:Q", format=".3f"),
                     alt.Tooltip("observed_rate:Q", format=".3f")],
        )
    )
    diag = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(color="#f59e0b", strokeDash=[6, 4])
        .encode(x="x:Q", y="y:Q")
    )
    st.altair_chart(bars + diag, height=360, width="stretch")

st.subheader("Cost and settings")
info = []
for m in comp["models"]:
    info.append({
        "model": m["name"],
        "train seconds": m.get("train_seconds"),
        "settings": "; ".join(f"{k}={v}" for k, v in (m.get("settings") or {}).items()
                              if not isinstance(v, dict)),
    })
st.dataframe(pd.DataFrame(info), hide_index=True, width="stretch")
st.caption(comp["protocol"])
