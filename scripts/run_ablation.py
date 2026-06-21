"""Protected downstream eval (SPEC v2 §5): the label-source ablation.

Hold the selector, features, budgets, and frozen reader fixed; vary only the
training label (A/B/C). Measure downstream EM/F1 at matched tokens on held-out
TEST examples, with paired-bootstrap 95% CIs for the deltas that matter:
  C - A   (does reader-grounded beat human labels?)
  C - B   (does it beat query-agnostic importance?)
  best - bm25  (the winnable matched-budget claim)
Plus the mandatory CONFOUNDER ablation: is any C-over-A edge just the near-dup
suppression we already ship? (compare with dedup on vs off).

Writes data/pareto_v2.json.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix import baselines  # noqa: E402
from suffix.compose import render_context  # noqa: E402
from suffix.coverage import select_coverage  # noqa: E402
from suffix.llm import READER_MODEL, have_key, read_answer  # noqa: E402
from suffix.metrics import squad_em, squad_f1, tokens_of  # noqa: E402
from suffix.scorer import KeepScorer  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
N = int(os.environ.get("ABLATION_N", "150"))
BUDGETS = [120, 240]
WORKERS = 12


def paired_ci(a, b, n=2000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.RandomState(seed)
    means = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return {"delta": float(d.mean()), "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "clears_zero": bool(np.percentile(means, 2.5) > 0 or np.percentile(means, 97.5) < 0)}


def main():
    if not have_key():
        print("NO API KEY"); sys.exit(1)
    test = json.load(open(os.path.join(DATA, "hotpot_test.json")))[:N]
    scorers = {s: KeepScorer.load(os.path.join(DATA, f"scorer_{s}.joblib")) for s in ["A", "B", "C"]}
    scores = {s: [scorers[s].score_example(ex) for ex in test] for s in scorers}
    ubu = [{u["uid"]: u for u in ex["units"]} for ex in test]

    # (name, budget, selector) ; selector(i)->uids
    conds = []
    def add(name, budget, fn): conds.append((name, budget, fn))
    add("full", None, lambda i: [u["uid"] for u in test[i]["units"]])
    for b in BUDGETS:
        for s in ["A", "B", "C"]:
            add(f"{s}@{b}", b, (lambda i, s=s, b=b: select_coverage(test[i], scores[s][i], b)))
        add(f"bm25@{b}", b, (lambda i, b=b: baselines.bm25_topk(test[i], b)))
    add("random@120", 120, lambda i: baselines.random_select(test[i], 120, seed=i))
    # confounder: dedup on vs off, for A and C at 240
    for s in ["A", "C"]:
        add(f"{s}_nodedup@240", 240, (lambda i, s=s: select_coverage(test[i], scores[s][i], 240, tau=2.0)))

    # precompute contexts (no API)
    jobs = []
    for ci, (name, budget, fn) in enumerate(conds):
        for i, ex in enumerate(test):
            uids = fn(i)
            jobs.append({"ci": ci, "i": i, "name": name, "ctx": render_context(ex, uids),
                         "q": ex["question"], "gold": ex["answer"], "tokens": tokens_of(ubu[i], uids)})
    print(f"reader={READER_MODEL} N={len(test)} conditions={len(conds)} calls={len(jobs)}")

    def run(job):
        try:
            pred = read_answer(job["ctx"], job["q"])
        except Exception as e:
            pred = ""; job["err"] = str(e)[:80]
        job["em"] = squad_em(pred, job["gold"]); job["f1"] = squad_f1(pred, job["gold"])
        return job

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for fut in as_completed([pool.submit(run, j) for j in jobs]):
            results.append(fut.result()); done += 1
            if done % 200 == 0: print(f"  {done}/{len(jobs)}")

    # aggregate + per-example vectors (aligned by example index i)
    agg, per_ex = {}, {}
    for ci, (name, budget, _) in enumerate(conds):
        rows = sorted([r for r in results if r["ci"] == ci], key=lambda r: r["i"])
        per_ex[name] = {"f1": [r["f1"] for r in rows], "em": [r["em"] for r in rows]}
        agg[name] = {"budget": budget, "avg_tokens": float(np.mean([r["tokens"] for r in rows])),
                     "em": float(np.mean([r["em"] for r in rows])), "f1": float(np.mean([r["f1"] for r in rows])),
                     "n": len(rows)}

    def pair(x, y): return paired_ci(per_ex[x]["f1"], per_ex[y]["f1"])
    cis = {
        "C_minus_A@240": pair("C@240", "A@240"),
        "C_minus_B@240": pair("C@240", "B@240"),
        "C_minus_A@120": pair("C@120", "A@120"),
        "bestCA_minus_bm25@240": pair("C@240" if agg["C@240"]["f1"] >= agg["A@240"]["f1"] else "A@240", "bm25@240"),
        "confounder_C_minus_A_NODEDUP@240": pair("C_nodedup@240", "A_nodedup@240"),
        "vs_full_A@240": pair("A@240", "full"),
    }

    out = {"reader": READER_MODEL, "n": len(test), "budgets": BUDGETS, "conditions": agg, "cis": cis,
           "scorer_report": json.load(open(os.path.join(DATA, "ablation_scorers_report.json")))}
    json.dump(out, open(os.path.join(DATA, "pareto_v2.json"), "w"), indent=2)
    json.dump(per_ex, open(os.path.join(DATA, "ablation_per_example.json"), "w"))

    print("\n=== LABEL-SOURCE ABLATION (downstream F1 @ matched tokens) ===")
    print(f"  {'cond':16s}{'tokens':>8s}{'EM':>7s}{'F1':>7s}")
    for name in ["full", "A@240", "B@240", "C@240", "bm25@240", "A@120", "B@120", "C@120", "bm25@120", "random@120"]:
        a = agg[name]; print(f"  {name:16s}{a['avg_tokens']:8.0f}{a['em']:7.3f}{a['f1']:7.3f}")
    print("\n=== PAIRED BOOTSTRAP 95% CIs (F1) ===")
    for k, v in cis.items():
        flag = "✅ clears 0" if v["clears_zero"] else "— within noise"
        print(f"  {k:34s} delta={v['delta']:+.3f}  CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  {flag}")
    print("\nsaved data/pareto_v2.json")


if __name__ == "__main__":
    main()
