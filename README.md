# Minimal Deep Knowledge Tracing (DKT)

A small, beginner-readable reimplementation of the main idea of
[Deep Knowledge Tracing](https://arxiv.org/abs/1506.05908) (Piech et al., 2015):
an LSTM reads a student's history of `(skill, correct?)` interactions and
predicts, at every step, the probability of a correct answer on the *next*
interaction for each skill.

This is a learning project, **not** a reproduction of the paper's numbers.
See [Differences from the paper](#differences-from-the-paper).

## Setup

Python 3.14 was used. For a CPU-only PyTorch build:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install numpy scikit-learn matplotlib pytest
```

## Data

### Synthetic (no download needed)
`data/synthetic.py` generates 400 fake students over 12 skills. Each skill has
a hidden mastery level that rises after a correct answer and falls after an
incorrect one, so recent history is genuinely predictive — something a static
baseline cannot see but the LSTM can learn.

### ASSISTments 2009–2010 (download once, ~9 MB)

```bash
mkdir -p data/raw
curl -o data/raw/2009_skill_builder_data_corrected.zip \
  http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip
cd data/raw && unzip -o 2009_skill_builder_data_corrected.zip && cd ../..
```

Preprocessing (`data/assistments.py`), each step reported by the loader itself:
1. Drop rows without a `skill_name` (76,119 of 401,756).
2. Collapse multi-skill duplicates: one interaction per `(user_id, order_id)`,
   keeping the first-listed skill (drops 51,047 rows).
3. Order each student's interactions by numeric `order_id`.
4. Keep skills with >= 1,000 interactions (101 -> 56 skills).
5. Keep students with >= 5 interactions (4,096 -> 3,579 students).

Result: 259,246 interactions, 56 skills, overall accuracy 0.663.

## Running

```bash
python train.py --data synthetic --epochs 20        # pipeline check
python -m data.assistments                          # loader stats + smoke test
python train.py --data assistments --epochs 20      # full experiment (~6 min CPU)
pytest tests -q                                     # 11 tests
```

Outputs go to `outputs/`: `loss_<data>.png`, `roc_<data>.png`,
`train_assistments.log`.

## Results (measured on 2026-10-19, seed 42, CPU)

Synthetic (20 epochs):

| predictor | n pairs | ROC-AUC | Accuracy |
|---|---|---|---|
| DKT (first-obs excluded) | 5,680 | **0.804** | 0.803 |
| per-skill rate baseline | 5,680 | 0.663 | 0.765 |
| random guess | 5,680 | 0.495 | 0.494 |

ASSISTments 2009–2010, 2,863 train / 716 test students, 20 epochs,
train loss 0.640 -> 0.544:

| predictor | n pairs | ROC-AUC | Accuracy |
|---|---|---|---|
| DKT (first-obs excluded) | 46,856 | **0.751** | 0.730 |
| DKT (all pairs) | 52,101 | 0.750 | 0.726 |
| per-skill rate baseline | 46,856 | 0.607 | 0.659 |
| random guess | 46,856 | 0.501 | 0.501 |

The directional claim replicates: per-student history adds predictive signal
beyond static per-skill rates (0.751 vs 0.607 AUC).

### Hyperparameter tuning (`python tune.py`)

8-trial random search (hidden 64–200, lr 1e-3–1e-2, dropout 0.2–0.5) ranked on
a validation split (12.5% of train, seed 43); the test split was untouched
until one final eval per config. Winner: hidden 200, lr 3e-3, dropout 0.2.
Retrained 20 epochs, it scores test AUC **0.752** (loss 0.608 -> 0.508) —
statistically indistinguishable from the untuned hidden 64 config's 0.751.

Honest conclusion: **tuning did not meaningfully help here.** The gap to the
baseline is a property of the data and task, not of the hyperparameters.
The shipped checkpoint (`outputs/model_assistments.pt`, used by the demo) is
the tuned hidden-200 model.

## How it works: the important code and tensor shapes

`B` = batch, `L` = sequence length, `S` = num_skills.

| stage | file | shape |
|---|---|---|
| raw sequence | `data/dataset.py` | `{"skills": [int] x L, "correct": [int] x L}` |
| padded batch | `collate_fn` | `skill_ids (B,L) long`, `correct (B,L) float`, `valid (B,L) bool` |
| model input | `model/dkt.py: DKT.forward` | `one_hot(skill) * valid` concat `correct` -> `(B, L, S+1)` |
| LSTM | same | per-step output `(B, L, hidden)` |
| logits | same | `(B, L, S)`; `logits[t]` predicts interaction `t+1` |
| loss | `masked_loss` | BCE on `gather(logits[t], skill[t+1])` vs `correct[t+1]`, only where both `valid[t]` and `valid[t+1]` |
| eval pairs | `collect_predictions` | flat arrays `(probs, labels, skills, seen)`; pairs where the tested skill never appeared before are excluded from headline metrics |

Three ideas worth pausing on:

1. **Masked multi-label loss.** The model outputs a probability for *every*
   skill at every step, but only the skill actually observed next contributes
   to the loss. Unobserved skills are neither rewarded nor punished.
2. **Causality = no leakage.** `logits[t]` is computed from inputs at
   positions `0..t` only; the answer at `t+1` is the target, never an input.
   `tests/test_masking.py::test_predictions_are_causal` enforces this.
3. **Student-level splits.** `split_by_student` puts each student's whole
   sequence on one side of the train/test boundary. Splitting at the
   interaction level would leak a student's later history into training.

## Differences from the paper

- Dataset: ASSISTments 2009–2010 (56 filtered skills), not the 2012–2013
  all-skills file with 276 skills that Piech et al. used. Published DKT AUC
  figures are **not** directly comparable.
- Multi-skill problems are reduced to one skill per interaction (first label
  wins), losing secondary-skill signal.
- One input encoding (skill one-hot ⊕ correct bit); the paper also tried a
  dual one-hot variant.
- 1-layer LSTM; headline run uses hidden 64, dropout 0.2, Adam 1e-3, 20
  epochs, no tuning (the shipped demo model is a tuned hidden-200 variant —
  see the tuning section above); the paper used hidden 50+ with tuned
  regularization and a Bayesian-optimization
  hyperparameter search.
- Padded steps are fed to the LSTM as all-zero inputs (masked out of the loss
  only). Sequences here are short enough that this is a minor simplification.

## Research Lab dashboard (Streamlit, multi-page)

Local run:

```bash
python train.py --data assistments --epochs 20 --hidden 200 \
  --lr 0.003 --dropout 0.2 --save outputs/model_assistments.pt
python compare.py                         # DKT vs BKT vs baselines -> outputs/comparison.json
python export_demo.py                     # writes outputs/demo_bundle.json.gz (~36 KB)
python -m streamlit run streamlit_app.py  # http://localhost:8501
```

Pages (dark charcoal theme, teal accents, amber flags):

| page | what it does |
|---|---|
| **Overview** | headline metrics, loss/ROC plots, 2-minute explanation |
| **Model Comparison** | DKT, BKT (EM), per-skill baseline, random on the *same* 46,856 eligible pairs; AUC/acc/log-loss/Brier/ECE + reliability diagrams |
| **Mastery Heatmap** | per student, per skill: model estimate vs observed accuracy with observation counts; skills with <5 observations flagged as thin evidence; drill-down into a skill's history |
| **Prediction Autopsy** | one held-out prediction stress-tested: drop or flip each earlier interaction and watch the probability move; sensitive interactions flagged |
| **Mastery Reality Check** | sequence-position proxy analysis (labelled as a proxy, not a retention study) + working CSV import path for delayed/transfer assessments |
| **Dataset & Experiment Settings** | exact checkpoint settings, tuning result, dataset provenance and preprocessing counts, reproduction commands |

Honest boundaries, per feature:

- **BKT is implemented and compared** (per-skill EM HMM, see `bkt.py`); it is
  a fair sequence-model comparison but has no cross-skill representation.
- **SAKT** (self-attentive KT) and the **skill dependency graph** are
  designated "later extensions" in the brief and are NOT implemented here —
  no stubs pretending to work.
- **Personal paper/exam upload and feedback (spec §7) is deliberately
  excluded** by the owner's decision; the marking engine would require an
  external LLM or OCR stack, so nothing half-working ships.
- Model probabilities are **estimates of next-answer behaviour**, not
  measurements of knowledge; heatmaps label model-estimated vs observed
  numbers separately, and thin-evidence cells are flagged.

### Deploying free on Streamlit Community Cloud

1. Create a public GitHub repository and push this project's files. Required:
   `streamlit_app.py`, `app_common.py`, `app_pages/`, `requirements.txt`,
   `config.py`, `baseline.py`, `evaluate.py`, `metrics.py`, `bkt.py`,
   `autopsy.py`, `reality_check.py`, `compare.py`, `plots.py`, `tune.py`,
   `train.py`, `export_demo.py`, `data/`, `model/`,
   `outputs/model_assistments.pt`, `outputs/demo_bundle.json.gz`,
   `outputs/comparison.json`, `outputs/bkt_params.json`,
   `.streamlit/config.toml`.
   Do NOT push `data/raw/` (63 MB) or `.venv` — add them to `.gitignore`.
2. Go to https://share.streamlit.io, sign in with the same GitHub account,
   "Add new app", pick the repo and set **Main file path** to
   `streamlit_app.py`.
3. First deploy takes a few minutes (pip installs CPU torch). Free tier is
   CPU-only.

## Layout

```
config.py          hyperparameters
data/dataset.py    sequence format, padding/collate, splits
data/synthetic.py  fake data with learnable recency dynamics
data/assistments.py  real-data loader + preprocessing stats
model/dkt.py       LSTM, masked BCE, prediction collection
baseline.py        per-skill empirical rate (train data only)
bkt.py             Bayesian Knowledge Tracing (EM, one HMM per skill)
metrics.py         log loss, Brier, ECE, reliability binning
evaluate.py        ROC-AUC / accuracy reporting
compare.py         all models on identical pairs -> outputs/comparison.json
autopsy.py         per-prediction drop/flip ablation analysis
reality_check.py   position-proxy + delayed/transfer CSV import + validation
plots.py           loss curve + ROC comparison -> outputs/
train.py           end-to-end entry point (records train_seconds in ckpt)
tune.py            random-search hyperparameter tuning
app_common.py      cached data/model access shared by pages
app_pages/         the six dashboard pages
tests/             shapes, masking, causality, leakage, baseline,
                   metrics, BKT recovery, autopsy, reality check,
                   headless render tests for every page
```
