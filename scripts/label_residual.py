"""Reader-grounded label C (SPEC v2 §4): the minimal sufficient subset of gold
units, found by greedy leave-one-out answer-impact against the frozen reader.

A gold unit is KEPT (necessary) if removing it drops the reader's F1 by > TOL;
otherwise it is DROPPED (redundant / recoverable from other kept context). This
is what makes C genuinely differ from A=`is_gold`.

The 30-minute GATE: run on N=30 examples and report the C-vs-A gold-unit FLIP
rate (fraction of gold the reader-grounded label says to drop). SPEC gate: must
be >= 10% (the probe predicts ~64%). Below 10% => STOP / downgrade.

Usage:
    python scripts/label_residual.py --n 30 --gate
    python scripts/label_residual.py --n 40            # full subset labeling
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix.compose import render_context  # noqa: E402
from suffix.llm import READER_MODEL, have_key, read_answer  # noqa: E402
from suffix.metrics import squad_f1  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TOL = 0.2          # F1 drop above which a unit is "necessary"
BASE_MIN = 0.5     # skip examples whose gold pool can't even answer (label noise)
WORKERS = 10


def label_example(example):
    """Return reader-grounded label dict + diagnostics, or None to skip."""
    q, gold = example["question"], example["answer"]
    gold_uids = list(example["gold_uids"])
    if not gold_uids:
        return None

    base_ctx = render_context(example, gold_uids)
    base_f1 = squad_f1(read_answer(base_ctx, q), gold)
    if base_f1 < BASE_MIN:
        return {"skip": True, "base_f1": base_f1}

    keep = list(gold_uids)  # greedy removal in document order
    for u in gold_uids:
        trial = [x for x in keep if x != u]
        if not trial:
            continue  # never remove the last remaining unit
        f1u = squad_f1(read_answer(render_context(example, trial), q), gold)
        if f1u >= base_f1 - TOL:
            keep = trial  # u is redundant -> drop

    necessary = set(keep)
    # C label over ALL units: keep only the necessary gold subset
    labels = {u["uid"]: (1 if u["uid"] in necessary else 0) for u in example["units"]}
    return {
        "skip": False,
        "labels": labels,
        "base_f1": base_f1,
        "n_gold": len(gold_uids),
        "n_necessary": len(necessary),
        "n_redundant_gold": len(gold_uids) - len(necessary),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()

    if not have_key():
        print("NO API KEY — cannot run reader-grounded labeling.")
        sys.exit(1)

    test = json.load(open(os.path.join(DATA, f"hotpot_{args.split}.json")))[: args.n]
    cache_path = os.path.join(DATA, f"residual_labels_{args.split}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    todo = [ex for ex in test if ex["id"] not in cache]
    print(f"reader={READER_MODEL}  examples={len(test)}  to_label={len(todo)} (cached {len(test)-len(todo)})")

    def work(ex):
        try:
            return ex["id"], label_example(ex)
        except Exception as e:
            return ex["id"], {"error": str(e)[:120]}

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for fut in as_completed([pool.submit(work, ex) for ex in todo]):
            eid, res = fut.result()
            if res is not None:
                cache[eid] = res
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(todo)} labeled")
    json.dump(cache, open(cache_path, "w"))

    # ---- report over the requested N ----
    used = [cache[ex["id"]] for ex in test if ex["id"] in cache]
    valid = [r for r in used if r and not r.get("skip") and not r.get("error") and "labels" in r]
    skipped = sum(1 for r in used if r and r.get("skip"))
    errors = sum(1 for r in used if r and r.get("error"))

    tot_gold = sum(r["n_gold"] for r in valid)
    tot_redundant = sum(r["n_redundant_gold"] for r in valid)
    flip_rate = tot_redundant / tot_gold if tot_gold else 0.0
    avg_base_f1 = sum(r["base_f1"] for r in valid) / max(1, len(valid))

    print("\n=== REASONER-GROUNDED LABELING REPORT ===")
    print(f"  valid={len(valid)}  skipped(base_f1<{BASE_MIN})={skipped}  errors={errors}")
    print(f"  avg gold-only base F1 = {avg_base_f1:.3f}")
    print(f"  gold units total={tot_gold}  redundant(dropped by C)={tot_redundant}  necessary={tot_gold-tot_redundant}")
    print(f"  >>> C-vs-A gold-unit FLIP RATE = {flip_rate:.1%}  (gate threshold 10%)")

    if args.gate:
        if flip_rate >= 0.10:
            print(f"\n  GATE PASSED ✅  ({flip_rate:.1%} >= 10%) — reader-grounded labels genuinely differ from is_gold. Proceed.")
        else:
            print(f"\n  GATE FAILED ❌  ({flip_rate:.1%} < 10%) — labels collapse to is_gold. STOP / downgrade to one ablation slide.")
    print(f"\nsaved labels -> {cache_path}")


if __name__ == "__main__":
    main()
