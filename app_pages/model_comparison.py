"""Model comparison: DKT vs BKT vs the simple baselines, plus live controls.

The headline table is the measured result from `python compare.py`. The
controls below recompute live from the same held-out pairs (carried in the
demo bundle), so moving a slider really changes the numbers — nothing here is
pre-baked except the summary table.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

from app_common import load_bundle, load_comparison, load_eval_pairs
from metrics import expected_calibration_error, log_loss, reliability_table

st.header("Model Comparison")
st.caption(
    "Four predictors, one exam. Every model is scored on exactly the same "
    "held-out test pairs (a skill's very first observation is dropped, "
    "because nobody can predict a topic they have never seen)."
)

comp = load_comparison()
pairs = load_eval_pairs()
bundle = load_bundle()
skill_names = bundle["skill_names"]
baseline_rates = np.array(bundle["baseline_rates"])

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
st.caption("Lower log loss / Brier / ECE is better; higher AUC and accuracy "
           "is better.")

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

if pairs is None:
    st.warning("The interactive controls below need eval pairs in the demo "
               "bundle. Rebuild it with `python export_demo.py` (after "
               "`compare.py` has run once).")
    st.stop()

probs_by_model = {
    "DKT": pairs["probs_dkt"],
    "BKT (EM, per skill)": pairs["probs_bkt"],
}
probs_by_model["per-skill rate baseline"] = baseline_rates[pairs["skills"]]
keep = pairs["seen"]
labels = pairs["labels"]

st.subheader("Try it: who is right, and how sure are they?")
st.caption(
    "A probability only becomes a decision once you pick a cutoff. Slide it "
    "and watch each model's hit rate change. The default 0.50 is the cutoff "
    "used in the table above."
)

c1, c2 = st.columns([2, 1])
with c1:
    thresh = st.slider("Say \"this student gets it right\" when the model's "
                       "chance is above...", 0.05, 0.95, 0.50, 0.01,
                       format="%.2f")
with c2:
    st.metric("Share of questions that are really correct",
              f"{labels[keep].mean():.1%}",
              help="This is the honest yardstick: a model that stays quiet "
                   "and predicts the class average is not stupid.")

rows = []
for name, p in probs_by_model.items():
    pred = (p[keep] >= thresh).astype(int)
    y = labels[keep]
    rows.append({
        "model": name,
        "accuracy": float((pred == y).mean()),
        "how often it says 'right'": float(pred.mean()),
        "missed a real correct": float(((pred == 0) & (y == 1)).mean()),
        "false alarm": float(((pred == 1) & (y == 0)).mean()),
    })
thr_df = pd.DataFrame(rows)
st.dataframe(
    thr_df, hide_index=True, width="stretch",
    column_config={
        "accuracy": st.column_config.ProgressColumn("accuracy", min_value=0.3,
                                                    max_value=0.9,
                                                    format="%.1%"),
        "how often it says 'right'": st.column_config.NumberColumn(
            "says 'right'", format="%.1%"),
        "missed a real correct": st.column_config.NumberColumn(
            "missed a real correct", format="%.1%"),
        "false alarm": st.column_config.NumberColumn("false alarm",
                                                     format="%.1%"),
    },
)
st.caption(
    "Push the cutoff very high and every model starts missing real correct "
    "answers; push it very low and the simple baseline looks great, because "
    "almost everyone gets most questions right. That is why the table above "
    "uses AUC too — it does not depend on a cutoff."
)

st.subheader("Try it: zoom in on one topic")
topic = st.selectbox(
    "Score the models on one topic only",
    ["All topics"] + list(skill_names),
    help="Some topics have thousands of pairs, others a handful. A model can "
         "win overall and lose badly on the topic you care about.",
)
mask = keep if topic == "All topics" else (keep & (pairs["skills"] ==
                                                   skill_names.index(topic)))
n_sub = int(mask.sum())
if n_sub < 20 or len(np.unique(labels[mask])) < 2:
    st.info(f"Not enough to compare on \"{topic}\" ({n_sub} pairs, need at "
            f"least 20 and both outcomes present). Pick another topic.")
else:
    sub_rows = []
    for name, p in probs_by_model.items():
        y = labels[mask]
        sp = p[mask]
        sub_rows.append({
            "model": name,
            "pairs": n_sub,
            "AUC": float(roc_auc_score(y, sp)),
            "log loss": log_loss(sp, y),
            "observed correct": float(y.mean()),
            "mean predicted": float(sp.mean()),
        })
    st.dataframe(
        pd.DataFrame(sub_rows).round(4), hide_index=True, width="stretch",
        column_config={
            "AUC": st.column_config.ProgressColumn("AUC", min_value=0.4,
                                                   max_value=0.9,
                                                   format="%.3f"),
            "observed correct": st.column_config.NumberColumn(format="%.1%"),
            "mean predicted": st.column_config.NumberColumn(format="%.1%"),
        },
    )
    best = max(sub_rows, key=lambda r: r["AUC"])
    st.caption(
        f"On this topic the best ranker is {best['model']} "
        f"(AUC {best['AUC']:.3f}). \"mean predicted\" vs \"observed correct\" "
        f"tells you whether a model is systematically too optimistic here."
    )

st.subheader("Reliability: are the probabilities honest?")
st.caption(
    "Each bar collects predictions of similar strength and plots them against "
    "what actually happened. A bar sitting on the amber dashed line is "
    "perfectly honest. Below the line = the model is overconfident. Hover for "
    "the number of pairs behind each bar — thin bars can mislead."
)
names = list(probs_by_model)
chosen = st.multiselect("Which models to plot", names, default=names[:2])
n_bins = st.slider("Number of probability bins", 5, 20, 10)
rel_rows = []
for name in chosen:
    for i, r in enumerate(reliability_table(probs_by_model[name][keep],
                                            labels[keep], n_bins)):
        rel_rows.append({"model": name, "bin_i": i, **r})
rel = pd.DataFrame(rel_rows)
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
    bins_summary = pd.DataFrame([
        {"model": name,
         "ECE at this bin count": expected_calibration_error(
             probs_by_model[name][keep], labels[keep], n_bins)}
        for name in chosen
    ]).round(4)
    st.dataframe(bins_summary, hide_index=True, width="stretch")
    st.caption(
        "ECE is the average gap between the bar height and the diagonal, "
        "weighted by how many pairs sit in each bin. Changing the bin count "
        "changes the score slightly — that is a known property of the "
        "measure, not a bug in the models."
    )

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
