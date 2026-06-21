"""Extrinsic arm (key-gated): does the COMPRESSED context actually let a frozen
reader answer as well as full context? A single frozen reader (claude-haiku-4-5)
answers each question from each arm's context; we score HotpotQA official
EM / token-F1 against the gold answer. The reader is identical across arms — a
measurement instrument, not a judge.

Adds `extrinsic` to data/pareto.json. Falls back gracefully with a clear message
if no API key is present (the intrinsic curve remains the headline).
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
from suffix.features import tfidf_cosines  # noqa: E402
from suffix.llm import READER_MODEL, have_key, read_answer  # noqa: E402
from suffix.metrics import squad_em, squad_f1, tokens_of  # noqa: E402
from suffix.scorer import KeepScorer  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
N = int(os.environ.get("DOWNSTREAM_N", "120"))
BUDGETS = [60, 120, 240]
WORKERS = 12


def build_conditions(test, scorer):
    scores = [scorer.score_example(ex) for ex in test]
    cosines = [tfidf_cosines(ex) for ex in test]
    ubu = [{u["uid"]: u for u in ex["units"]} for ex in test]

    conds = [("full", None)]
    for b in BUDGETS:
        conds.append(("suffix", b))
        conds.append(("bm25_topk", b))
    conds.append(("random", 120))

    def select(method, i, budget):
        ex = test[i]
        if method == "full":
            return [u["uid"] for u in ex["units"]]
        if method == "suffix":
            return baselines.suffix(ex, budget, scores[i])
        if method == "bm25_topk":
            return baselines.bm25_topk(ex, budget)
        if method == "random":
            return baselines.random_select(ex, budget, seed=i)
        raise ValueError(method)

    # Precompute every (condition, example) context up front (no API).
    jobs = []
    for ci, (method, budget) in enumerate(conds):
        for i, ex in enumerate(test):
            uids = select(method, i, budget)
            ctx = render_context(ex, uids)
            jobs.append({
                "ci": ci, "i": i, "method": method, "budget": budget,
                "context": ctx, "question": ex["question"], "gold": ex["answer"],
                "tokens": tokens_of(ubu[i], uids),
            })
    return conds, jobs


def main():
    if not have_key():
        print("NO API KEY — skipping extrinsic arm. Intrinsic curve remains the headline.")
        return

    test = json.load(open(os.path.join(DATA, "hotpot_test.json")))[:N]
    scorer = KeepScorer.load(os.path.join(DATA, "keep_scorer.joblib"))
    conds, jobs = build_conditions(test, scorer)
    print(f"reader={READER_MODEL}  examples={len(test)}  conditions={len(conds)}  calls={len(jobs)}")

    def run_job(job):
        try:
            pred = read_answer(job["context"], job["question"])
        except Exception as e:
            pred = ""
            job["error"] = str(e)[:120]
        job["pred"] = pred
        job["em"] = squad_em(pred, job["gold"])
        job["f1"] = squad_f1(pred, job["gold"])
        return job

    done = 0
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} calls done")

    # aggregate per condition
    agg = []
    for ci, (method, budget) in enumerate(conds):
        rows = [r for r in results if r["ci"] == ci]
        agg.append({
            "method": method, "budget": budget,
            "avg_tokens": float(np.mean([r["tokens"] for r in rows])),
            "em": float(np.mean([r["em"] for r in rows])),
            "f1": float(np.mean([r["f1"] for r in rows])),
            "n": len(rows),
            "errors": sum(1 for r in rows if r.get("error")),
        })

    pareto_path = os.path.join(DATA, "pareto.json")
    pareto = json.load(open(pareto_path))
    pareto["extrinsic"] = {"reader_model": READER_MODEL, "n": len(test), "conditions": agg}
    json.dump(pareto, open(pareto_path, "w"), indent=2)
    json.dump(results, open(os.path.join(DATA, "downstream_raw.json"), "w"))

    print("\n=== EXTRINSIC (downstream answer quality) ===")
    print(f"  {'condition':22s} {'tokens':>7s} {'EM':>6s} {'F1':>6s}")
    for a in agg:
        name = a["method"] + (f"@{a['budget']}" if a["budget"] else "")
        print(f"  {name:22s} {a['avg_tokens']:7.0f} {a['em']:6.3f} {a['f1']:6.3f}")
    print("\nsaved extrinsic -> data/pareto.json")


if __name__ == "__main__":
    main()
