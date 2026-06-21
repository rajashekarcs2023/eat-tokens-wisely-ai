"""Variable-rate compression: the LEARNED per-example budget predictor.

The oracle (analyze_adaptive.py) showed big headroom, but it cheats by using the
gold answer to pick each example's lucky budget. The honest question: can a cheap
predictor using ONLY query/context features (no answer) capture a real fraction of
it? We evaluate entirely OFFLINE — ratesweep.json has F1 at every budget for every
example, so any allocation policy is a lookup, no new API calls.

Policy: predict a per-example 'needed tokens'; to hit a target AVERAGE budget T,
scale predictions so the mean matches T and give each example the swept budget
closest to its scaled prediction. Compare learned-adaptive vs fixed-T on held-out
examples, paired bootstrap CI.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix.coverage import pairwise_sim  # noqa: E402
from suffix.features import tfidf_cosines  # noqa: E402
from suffix.scorer import KeepScorer  # noqa: E402
from suffix.text_utils import cap_tokens, content_tokens  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SWEPT = [40, 80, 120, 180, 240, 360]


def ex_features(ex, scores):
    units = ex["units"]
    cos = tfidf_cosines(ex)
    sim = pairwise_sim(ex)
    n = len(units)
    upper = sim[np.triu_indices(n, 1)] if n > 1 else np.array([0.0])
    sc = np.sort(scores)[::-1]
    q = ex["question"]
    return [
        n,                                            # n units
        sum(u["tokens"] for u in units),              # raw tokens
        len(content_tokens(q)),                       # query content words
        len(set(cap_tokens(q))),                      # query entities
        float(cos.max()), float(cos.mean()), float(cos.std()),   # relevance concentration
        float(np.mean(upper)), float(np.max(upper)),  # redundancy (mean/max pairwise sim)
        float(sc[0]), float(sc[:3].mean()),           # top scores
        float(sc[0] - (sc[4] if len(sc) > 4 else 0)), # score gap (concentration)
        int((scores > 0.5).sum()),                    # n confident units
    ]


def main():
    test = json.load(open(os.path.join(DATA, "hotpot_test.json")))
    sweep = json.load(open(os.path.join(DATA, "ratesweep.json")))["per_example"]
    keys = sorted(sweep.keys(), key=int)
    test = test[: len(keys)]
    scorer = KeepScorer.load(os.path.join(DATA, "keep_scorer.joblib"))

    X, F1, TOK = [], [], []
    for k in keys:
        ex = test[int(k)]
        scores = scorer.score_example(ex)
        X.append(ex_features(ex, scores))
        F1.append([sweep[k]["f1"][str(b)] for b in SWEPT])
        TOK.append([sweep[k]["tokens"][str(b)] for b in SWEPT])
    X = np.array(X, float); F1 = np.array(F1); TOK = np.array(TOK)
    n = len(X)

    # target: cheapest swept budget reaching within 0.1 F1 of this example's best
    best = F1.max(axis=1)
    target_tokens = np.array([
        next((TOK[i][j] for j in range(len(SWEPT)) if F1[i][j] >= best[i] - 0.1), TOK[i][-1])
        for i in range(n)
    ], float)

    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    tr, te = idx[: int(n * 0.65)], idx[int(n * 0.65):]
    reg = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0).fit(X[tr], target_tokens[tr])
    pred = reg.predict(X)

    def alloc(ex_ids, T):
        """choose per-example swept budget so mean(tokens)~=T, scaled by predicted need."""
        # binary search scale alpha
        lo, hi = 0.05, 8.0
        for _ in range(40):
            a = (lo + hi) / 2
            jchosen = [int(np.argmin(np.abs(TOK[i] - a * pred[i]))) for i in ex_ids]
            mt = np.mean([TOK[i][j] for i, j in zip(ex_ids, jchosen)])
            if mt > T: hi = a
            else: lo = a
        jchosen = [int(np.argmin(np.abs(TOK[i] - a * pred[i]))) for i in ex_ids]
        f1 = np.array([F1[i][j] for i, j in zip(ex_ids, jchosen)])
        tok = np.array([TOK[i][j] for i, j in zip(ex_ids, jchosen)])
        return f1, tok

    def fixed(ex_ids, T):
        j = int(np.argmin([abs(TOK[:, j].mean() - T) for j in range(len(SWEPT))]))
        return F1[ex_ids, j], TOK[ex_ids, j]

    def paired_ci(d, n=2000, seed=0):
        r = np.random.RandomState(seed)
        m = [d[r.randint(0, len(d), len(d))].mean() for _ in range(n)]
        return float(d.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    print(f"train={len(tr)} test={len(te)}  features={X.shape[1]}")
    print("\n=== LEARNED-adaptive vs FIXED on held-out test (matched avg tokens) ===")
    rows = []
    for T in [80, 120, 180]:
        fa, ta = alloc(te, T)
        ff, tf = fixed(te, T)
        delta, ci = paired_ci(fa - ff)
        clears = ci[0] > 0 or ci[1] < 0
        print(f"  @~{T:3d} tok:  fixed F1={ff.mean():.3f} ({tf.mean():.0f}t)   "
              f"learned-adaptive F1={fa.mean():.3f} ({ta.mean():.0f}t)   "
              f"Δ={delta:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] {'✅' if clears and delta>0 else '—'}")
        rows.append({"target": T, "fixed_f1": float(ff.mean()), "fixed_tokens": float(tf.mean()),
                     "adaptive_f1": float(fa.mean()), "adaptive_tokens": float(ta.mean()),
                     "delta": delta, "ci": ci, "clears_zero": bool(clears)})
    imp = sorted(zip(["n_units","raw_tok","q_words","q_ents","cos_max","cos_mean","cos_std",
                      "redund_mean","redund_max","sc_top1","sc_top3","sc_gap","n_confident"],
                     reg.feature_importances_), key=lambda t: -t[1])[:5]
    print("\n  top predictor features:", [(f, round(w, 3)) for f, w in imp])
    json.dump({"rows": rows, "n_test": len(te),
               "top_features": [[f, float(w)] for f, w in imp]},
              open(os.path.join(DATA, "adaptive_result.json"), "w"), indent=2)
    print("\nsaved data/adaptive_result.json")


if __name__ == "__main__":
    main()
