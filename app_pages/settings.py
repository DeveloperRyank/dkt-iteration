"""Dataset & experiment settings: what was run, on what, with which knobs."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app_common import load_bundle, load_model

st.header("Dataset & Experiment Settings")

model, ckpt = load_model()
bundle = load_bundle()

st.subheader("Model checkpoint (the one powering this app)")
st.dataframe(
    pd.DataFrame([{k: ckpt.get(k) for k in
                   ("data", "num_skills", "hidden_size", "dropout", "epochs",
                    "lr", "batch_size", "seed", "train_seconds", "test_auc")}]),
    hide_index=True, width="stretch",
)
st.caption(
    "Older checkpoints predate `train_seconds`/`lr` bookkeeping; missing "
    "values show as None rather than being guessed."
)

best_path = Path("outputs/best_config.json")
if best_path.exists():
    st.subheader("Random-search tuning result")
    st.json(json.loads(best_path.read_text()))
    st.caption(
        "8-trial search, validation = 12.5% of train (seed 43), test touched "
        "once. Honest finding: the tuned config (test AUC 0.752) is "
        "statistically tied with the untuned hidden-64 config (0.751) — "
        "tuning did not meaningfully help."
    )

st.subheader("Dataset provenance")
st.markdown(
    """
- **Source:** ASSISTments 2009–2010, `skill_builder_data_corrected.csv`
  (~401k interactions), USTC mirror
  `http://base.ustc.edu.cn/data/ASSISTment/` (login-free).
  NOT the paper's 2012–2013 set — published AUC figures are not comparable.
- **Preprocessing (measured during the actual run):**
  1. drop rows with no skill label;
  2. deduplicate multi-skill problems (first skill kept, per interaction);
  3. keep skills with >= 1000 interactions (101 -> 56 skills);
  4. keep students with >= 5 interactions (4,096 -> 3,579 students);
  5. student-level 80/20 split, seed 42 (2,863 train / 716 test).
- **Eval protocol:** each skill's first observation excluded (46,856
  eligible test pairs); baseline fitted on train pairs only.
"""
)

st.subheader("Bundle shipped to this app")
c1, c2, c3 = st.columns(3)
c1.metric("Test students", len(bundle["test_students"]))
c2.metric("Interactions covered",
          sum(len(s["skills"]) for s in bundle["test_students"]))
c3.metric("Skills", bundle["num_skills"])
st.caption(
    "The app reads a 36 KB compressed bundle of held-out students plus the "
    "model weights — the 63 MB raw CSV is never downloaded or shipped."
)

st.subheader("Reproduce every number here")
st.code(
    """
python train.py --data assistments --epochs 20 --hidden 200 \\
  --lr 0.003 --dropout 0.2 --save outputs/model_assistments.pt
python compare.py            # DKT vs BKT vs baselines -> outputs/comparison.json
python export_demo.py        # rebuild the demo bundle
pytest tests -q              # the test suite
python -m streamlit run streamlit_app.py
""",
    language="bash",
)
