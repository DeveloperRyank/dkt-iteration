"""Bayesian Knowledge Tracing (BKT), a classic pre-neural model.

One independent 2-state (unlearned/learned) HMM per skill, fitted with
expectation-maximisation on the training split. Compared against DKT on
identical test pairs, BKT is the honest "does the LSTM actually add
sequence-level signal beyond a simple per-skill learning model?" check.

Model per skill (parameters in [0, 1]):
  pi : P(learned at start)          pT : P(learned | unlearned, one step)
  pG : P(correct | unlearned)       pS : P(incorrect | learned)
Prediction before an observation with prior L:
  p(correct) = L * (1 - pS) + (1 - L) * pG
After the observation the posterior is updated, then pT advances it.
"""

import numpy as np

CLAMP = (1e-3, 1 - 1e-3)


def _clamp(x):
    return float(np.clip(x, *CLAMP))


def _step_emission(L, o, pG, pS):
    """Posterior P(learned | observation o), given prior P(learned)=L."""
    num = L * ((1 - pS) if o == 1 else pS)
    den = num + (1 - L) * (pG if o == 1 else (1 - pG))
    return num / den if den > 0 else L


def _pack(sequences):
    """Pad per-student binary sequences into (N, T) obs + validity mask."""
    sequences = [s for s in sequences if len(s) > 0]
    lens = np.array([len(s) for s in sequences])
    T = int(lens.max())
    obs = np.zeros((len(sequences), T))
    for i, s in enumerate(sequences):
        obs[i, :len(s)] = s
    mask = np.arange(T)[None, :] < lens[:, None]
    return obs, mask


def _em_totals(obs, mask, pi, pT, pG, pS):
    """Batched forward-backward E-step; returns summed sufficient stats.

    alpha/beta are row-normalised each step (standard scaling to avoid
    underflow). Padded steps carry the previous row forward and are excluded
    from the statistics via `mask`.
    """
    N, T = obs.shape
    eL = np.where(obs == 1, 1 - pS, pS)   # emission of learned state
    eU = np.where(obs == 1, pG, 1 - pG)   # emission of unlearned state

    aL = np.zeros((N, T))
    aU = np.zeros((N, T))
    aL[:, 0] = pi * eL[:, 0]
    aU[:, 0] = (1 - pi) * eU[:, 0]
    s = np.maximum(aL[:, 0] + aU[:, 0], 1e-300)
    aL[:, 0] /= s
    aU[:, 0] /= s
    for t in range(1, T):
        newL = (aL[:, t - 1] + aU[:, t - 1] * pT) * eL[:, t]
        newU = aU[:, t - 1] * (1 - pT) * eU[:, t]
        st = np.maximum(newL + newU, 1e-300)
        v = mask[:, t]
        aL[:, t] = np.where(v, newL / st, aL[:, t - 1])
        aU[:, t] = np.where(v, newU / st, aU[:, t - 1])

    bL = np.ones((N, T))
    bU = np.ones((N, T))
    for t in range(T - 2, -1, -1):
        nL = eL[:, t + 1] * bL[:, t + 1]
        nU = pT * nL + (1 - pT) * eU[:, t + 1] * bU[:, t + 1]
        st = np.maximum(nL + nU, 1e-300)
        v = mask[:, t] & mask[:, t + 1]
        bL[:, t] = np.where(v, nL / st, bL[:, t + 1])
        bU[:, t] = np.where(v, nU / st, bU[:, t + 1])

    gL = aL * bL
    gU = aU * bU
    gs = np.maximum(gL + gU, 1e-300)
    gL /= gs
    gU /= gs
    gL *= mask
    gU *= mask

    step_pair = mask[:, :-1] & mask[:, 1:]
    # Exact adjacent-transition posteriors xi_t(state -> state'), using the
    # scaled alpha/beta above; each t's four entries share a scale, so
    # renormalising row-wise recovers the true joint distribution.
    next_L = eL[:, 1:] * bL[:, 1:]
    next_U = eU[:, 1:] * bU[:, 1:]
    x_UL = aU[:, :-1] * pT * next_L
    x_UU = aU[:, :-1] * (1 - pT) * next_U
    x_LL = aL[:, :-1] * next_L          # learned stays learned with prob 1
    xi = x_LL + x_UL + x_UU
    pT_num = np.where(step_pair & (xi > 0), x_UL / np.maximum(xi, 1e-300), 0.0)
    pT_den = np.where(step_pair & (xi > 0), (x_UL + x_UU) / np.maximum(xi, 1e-300),
                      0.0)

    correct = (obs == 1)
    return {
        "init_L": float(gL[:, 0].sum()),
        "n_seq": float(N),
        "trans_U_L": float(pT_num.sum()),
        "trans_U_U": float((pT_den - pT_num).sum()),
        "emit_L": float(gL.sum()),
        "emit_U": float(gU.sum()),
        "emit_L_correct": float((gL * correct).sum()),
        "emit_U_correct": float((gU * correct).sum()),
    }


def fit_bkt(sequences, n_iter=100, tol=1e-6, seed=0):
    """Fit BKT parameters on binary observation sequences (list of lists)."""
    obs, mask = _pack(sequences)
    rng = np.random.default_rng(seed)
    pi, pT, pG, pS = (_clamp(0.3 + 0.1 * rng.random()) for _ in range(4))
    for _ in range(n_iter):
        agg = _em_totals(obs, mask, pi, pT, pG, pS)
        pi_new = _clamp(agg["init_L"] / agg["n_seq"])
        denom = agg["trans_U_L"] + agg["trans_U_U"]
        pT_new = _clamp(agg["trans_U_L"] / denom) if denom > 0 else pT
        pG_new = _clamp(agg["emit_U_correct"] / agg["emit_U"]) if agg["emit_U"] > 0 else pG
        pS_new = _clamp(1 - agg["emit_L_correct"] / agg["emit_L"]) if agg["emit_L"] > 0 else pS
        delta = max(abs(pi_new - pi), abs(pT_new - pT), abs(pG_new - pG), abs(pS_new - pS))
        pi, pT, pG, pS = pi_new, pT_new, pG_new, pS_new
        if delta < tol:
            break
    return {"pi": pi, "pT": pT, "pG": pG, "pS": pS}


class BKTModel:
    """One fitted BKT per skill; predicts like DKT's pair convention."""

    def __init__(self, num_skills, seed=0):
        self.num_skills = num_skills
        self.seed = seed
        self.params = {}

    def fit(self, sequences, n_iter=100):
        by_skill = {}
        for seq in sequences:
            per_student = {}
            for s, c in zip(seq["skills"], seq["correct"]):
                per_student.setdefault(s, []).append(c)
            for s, obs in per_student.items():
                by_skill.setdefault(s, []).append(obs)
        for s, obs_seqs in by_skill.items():
            self.params[s] = fit_bkt(obs_seqs, n_iter=n_iter, seed=self.seed + s)
        return self

    def predict_pairs(self, sequences):
        """Flat (probs, labels, skills, seen) for every consecutive pair.

        Mirrors model.dkt.collect_predictions exactly: one entry per
        interaction at position >= 1 (the first interaction of a sequence has
        no preceding history, so DKT makes no prediction there either).
        """
        probs, labels, skills, seen = [], [], [], []
        for seq in sequences:
            state = {}  # skill -> prior P(learned) at next observation
            for pos, (s, c) in enumerate(zip(seq["skills"], seq["correct"])):
                p = self.params.get(s)
                if p is None:
                    continue
                L = state.get(s, p["pi"])
                if pos >= 1:
                    probs.append(L * (1 - p["pS"]) + (1 - L) * p["pG"])
                    labels.append(c)
                    skills.append(s)
                    seen.append(s in state)
                post = _step_emission(L, c, p["pG"], p["pS"])
                state[s] = post + (1 - post) * p["pT"]
        return (np.array(probs), np.array(labels, dtype=float),
                np.array(skills), np.array(seen, dtype=bool))
