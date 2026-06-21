"""Intrinsic eval (zero API) -> data/pareto.json.

For every method and every token budget, measure how well the KEPT spans match
the human gold supporting facts (precision / recall / F1) and how many tokens
were actually sent. This is the guaranteed, key-independent headline: an
accuracy(F1)-vs-tokens Pareto frontier where SUFFIX sits above-and-left of
random, TF-IDF/BM25 top-k, and the no-coverage / no-scorer ablations.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix import baselines  # noqa: E402
from suffix.features import tfidf_cosines  # noqa: E402
from suffix.metrics import supporting_fact_prf, tokens_of  # noqa: E402
from suffix.scorer import KeepScorer  # noqa: E402
from suffix.structural import structural_report  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BUDGETS = [40, 60, 80, 120, 160, 240, 320, 480]


def _mcp_bundle():
    repo = "https://github.com/acme/web-app"
    user = {"login": "octo-dev", "url": "https://github.com/octo-dev", "type": "User"}
    label = {"name": "bug", "color": "d73a4a", "url": repo + "/labels/auth"}
    return {
        "tool": "github.list_issues",
        "repository_url": repo,
        "issues": [
            {"number": n, "repository_url": repo, "user": user, "labels": [label],
             "state": "open", "html_url": f"{repo}/issues/{n}",
             "body": "Deep link is dropped after the email verification step."}
            for n in range(1, 13)
        ],
    }


def bootstrap_ci(values, n=500, seed=0):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values)
    if len(arr) == 0:
        return [0.0, 0.0]
    means = [arr[rng.randint(0, len(arr), len(arr))].mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    test = json.load(open(os.path.join(DATA, "hotpot_test.json")))
    scorer = KeepScorer.load(os.path.join(DATA, "keep_scorer.joblib"))

    # Precompute per-example learned scores + tfidf cosines once.
    scores = [scorer.score_example(ex) for ex in test]
    cosines = [tfidf_cosines(ex) for ex in test]
    units_by_uid = [{u["uid"]: u for u in ex["units"]} for ex in test]

    def run(method, ex_i, budget):
        ex = test[ex_i]
        if method == "suffix":
            return baselines.suffix(ex, budget, scores[ex_i])
        if method == "no_coverage":
            return baselines.score_topk(ex, budget, scores[ex_i])
        if method == "no_scorer":
            return baselines.coverage_no_scorer(ex, budget)
        if method == "tfidf_topk":
            return baselines.tfidf_topk(ex, budget, cosines[ex_i])
        if method == "bm25_topk":
            return baselines.bm25_topk(ex, budget)
        if method == "random":
            return baselines.random_select(ex, budget, seed=ex_i)
        raise ValueError(method)

    methods = ["suffix", "no_coverage", "no_scorer", "tfidf_topk", "bm25_topk", "random"]
    intrinsic = {m: [] for m in methods}
    # per-question-type recall (coverage should help most on multi-entity comparisons)
    by_type = {m: {"comparison": [], "bridge": []} for m in methods}

    for m in methods:
        for b in BUDGETS:
            f1s, recs, precs, toks = [], [], [], []
            type_rec = {"comparison": [], "bridge": []}
            for i, ex in enumerate(test):
                sel = run(m, i, b)
                prf = supporting_fact_prf(sel, ex["gold_uids"])
                f1s.append(prf["f1"]); recs.append(prf["recall"]); precs.append(prf["precision"])
                toks.append(tokens_of(units_by_uid[i], sel))
                if ex.get("type") in type_rec:
                    type_rec[ex["type"]].append(prf["recall"])
            intrinsic[m].append({
                "budget": b,
                "avg_tokens": float(np.mean(toks)),
                "f1": float(np.mean(f1s)),
                "recall": float(np.mean(recs)),
                "recall_ci": bootstrap_ci(recs),
                "precision": float(np.mean(precs)),
                "f1_ci": bootstrap_ci(f1s),
            })
            for t in type_rec:
                by_type[m][t].append({
                    "budget": b,
                    "recall": float(np.mean(type_rec[t])) if type_rec[t] else 0.0,
                    "n": len(type_rec[t]),
                })
        print(f"  {m:12s} done")

    # full-context reference point
    full_f1, full_rec, full_prec, full_tok = [], [], [], []
    for i, ex in enumerate(test):
        sel = [u["uid"] for u in ex["units"]]
        prf = supporting_fact_prf(sel, ex["gold_uids"])
        full_f1.append(prf["f1"]); full_rec.append(prf["recall"]); full_prec.append(prf["precision"])
        full_tok.append(tokens_of(units_by_uid[i], sel))
    full_point = {
        "avg_tokens": float(np.mean(full_tok)), "f1": float(np.mean(full_f1)),
        "recall": float(np.mean(full_rec)), "precision": float(np.mean(full_prec)),
    }

    scorer_report = json.load(open(os.path.join(DATA, "scorer_report.json")))

    out = {
        "meta": {
            "n_test": len(test),
            "budgets": BUDGETS,
            "avg_raw_tokens": float(np.mean([ex["raw_tokens"] for ex in test])),
            "avg_units": float(np.mean([ex["n_units"] for ex in test])),
            "avg_gold": float(np.mean([len(ex["gold_uids"]) for ex in test])),
            "gold_frac": float(np.mean([len(ex["gold_uids"]) / ex["n_units"] for ex in test])),
            "tokenizer": "tiktoken/cl100k_base (ratio ruler, not Claude-exact)",
        },
        "scorer": scorer_report,
        "intrinsic": intrinsic,
        "by_type": by_type,
        "full_point": full_point,
        "structural_demo": structural_report(_mcp_bundle()),
    }
    json.dump(out, open(os.path.join(DATA, "pareto.json"), "w"), indent=2)

    print("\n=== INTRINSIC SUMMARY (gold-fact recall / F1 @ avg tokens) ===")
    print(f"  full context: recall={full_point['recall']:.2f}  F1={full_point['f1']:.3f}  tokens={full_point['avg_tokens']:.0f}")
    for b_idx, b in enumerate(BUDGETS):
        if b not in (60, 120, 240):
            continue
        print(f"  budget~{b}:")
        for m in methods:
            r = intrinsic[m][b_idx]
            print(f"    {m:12s} recall={r['recall']:.2f}  F1={r['f1']:.3f}  tokens={r['avg_tokens']:.0f}")
    print("\n=== RECALL by question type @ budget~120 ===")
    bi = BUDGETS.index(120)
    for m in methods:
        c = by_type[m]["comparison"][bi]; b_ = by_type[m]["bridge"][bi]
        print(f"  {m:12s} comparison={c['recall']:.2f} (n={c['n']})  bridge={b_['recall']:.2f} (n={b_['n']})")
    print("\nsaved data/pareto.json")


if __name__ == "__main__":
    main()
