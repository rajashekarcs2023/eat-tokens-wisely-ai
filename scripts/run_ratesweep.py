"""Per-example budget sweep — the data for variable-rate compression.

For each held-out example, measure the frozen reader's F1 at a range of token
budgets (the example's rate-distortion curve), plus full context. This lets us
compute, OFFLINE:
  - the FIXED-budget frontier (every example gets the same budget), and
  - the ORACLE adaptive frontier (each example gets its own budget, optimally
    allocated under an average-token constraint -- water-filling).
If oracle-adaptive does not clearly beat fixed at matched average tokens, the
variable-rate idea has no headroom and we stop. (analyze_adaptive.py does that.)

Writes data/ratesweep.json (per-example f1 + tokens at each budget).
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix.coverage import select_coverage  # noqa: E402
from suffix.compose import render_context  # noqa: E402
from suffix.llm import READER_MODEL, have_key, read_answer  # noqa: E402
from suffix.metrics import squad_f1, tokens_of  # noqa: E402
from suffix.scorer import KeepScorer  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
N = int(os.environ.get("SWEEP_N", "160"))
BUDGETS = [40, 80, 120, 180, 240, 360]
WORKERS = 12


def main():
    if not have_key():
        print("NO API KEY"); sys.exit(1)
    test = json.load(open(os.path.join(DATA, "hotpot_test.json")))[:N]
    scorer = KeepScorer.load(os.path.join(DATA, "keep_scorer.joblib"))  # production scorer
    scores = [scorer.score_example(ex) for ex in test]
    ubu = [{u["uid"]: u for u in ex["units"]} for ex in test]

    jobs = []
    for i, ex in enumerate(test):
        # full context
        full_uids = [u["uid"] for u in ex["units"]]
        jobs.append({"i": i, "budget": "full", "ctx": render_context(ex, full_uids),
                     "q": ex["question"], "gold": ex["answer"], "tokens": tokens_of(ubu[i], full_uids)})
        for b in BUDGETS:
            sel = select_coverage(ex, scores[i], b)
            jobs.append({"i": i, "budget": b, "ctx": render_context(ex, sel),
                         "q": ex["question"], "gold": ex["answer"], "tokens": tokens_of(ubu[i], sel)})
    print(f"reader={READER_MODEL} N={len(test)} budgets={BUDGETS}+full calls={len(jobs)}")

    def run(job):
        try:
            job["f1"] = squad_f1(read_answer(job["ctx"], job["q"]), job["gold"])
        except Exception as e:
            job["f1"] = 0.0; job["err"] = str(e)[:80]
        return job

    done, results = 0, []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for fut in as_completed([pool.submit(run, j) for j in jobs]):
            results.append(fut.result()); done += 1
            if done % 200 == 0: print(f"  {done}/{len(jobs)}")

    per_ex = {}
    for i in range(len(test)):
        rows = [r for r in results if r["i"] == i]
        f1b, tokb = {}, {}
        for r in rows:
            f1b[str(r["budget"])] = r["f1"]; tokb[str(r["budget"])] = r["tokens"]
        per_ex[str(i)] = {"id": test[i]["id"], "f1": f1b, "tokens": tokb,
                          "question": test[i]["question"]}

    out = {"reader": READER_MODEL, "n": len(test), "budgets": BUDGETS, "per_example": per_ex}
    json.dump(out, open(os.path.join(DATA, "ratesweep.json"), "w"))
    # quick fixed-budget summary
    print("\n=== fixed-budget summary (avg tokens / avg F1) ===")
    for b in ["full"] + [str(x) for x in BUDGETS]:
        f1s = [per_ex[k]["f1"][b] for k in per_ex if b in per_ex[k]["f1"]]
        tks = [per_ex[k]["tokens"][b] for k in per_ex if b in per_ex[k]["tokens"]]
        print(f"  budget {b:>5}: tokens={np.mean(tks):6.0f}  F1={np.mean(f1s):.3f}")
    print("\nsaved data/ratesweep.json  (run analyze_adaptive.py next)")


if __name__ == "__main__":
    main()
