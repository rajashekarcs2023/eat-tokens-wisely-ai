"""Cross-dataset generalization: does the HotpotQA-trained compressor, UNCHANGED,
work on DIFFERENT datasets? We never retrain. Same scorer, same selector, same
budgets; new data:
  - SQuAD v1.1   : single-hop QA, single short paragraph (different structure)
  - 2WikiMultiHop: multi-hop QA, long context (different question construction)
Measure downstream EM/F1 at budgets vs full + bm25 + random. If the
rate-distortion behavior holds across all three, generalization is proven, not
assumed.

Usage: python scripts/crosstask.py --dataset squad --n 80
"""

from __future__ import annotations

import argparse
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
from suffix.tokens import count_tokens  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WORKERS = 12


def _units_from_paras(titles, sentences):
    units = []
    for pi, (title, para) in enumerate(zip(titles, sentences)):
        for si, s in enumerate(para):
            if not s.strip():
                continue
            units.append({"uid": f"p{pi}s{si}", "para": pi, "sent": si,
                          "n_sent_in_para": len(para), "title": title, "text": s,
                          "tokens": count_tokens(s), "is_gold": False})
    return units


def build_squad(n):
    from datasets import load_dataset
    from nltk.tokenize import sent_tokenize
    ds = load_dataset("rajpurkar/squad", split=f"validation[:{n*2}]")
    out = []
    for ex in ds:
        sents = sent_tokenize(ex["context"])
        units = _units_from_paras([ex["title"]], [sents])
        if len(units) < 3 or not ex["answers"]["text"]:
            continue
        out.append({"question": ex["question"], "answer": ex["answers"]["text"][0],
                    "n_paras": 1, "units": units})
        if len(out) >= n:
            break
    return out


def build_2wiki(n):
    from datasets import load_dataset
    ds = load_dataset("framolfese/2WikiMultihopQA", split=f"validation[:{n*2}]")
    out = []
    for ex in ds:
        c = ex["context"]
        units = _units_from_paras(c["title"], c["sentences"])
        if len(units) < 5:
            continue
        out.append({"question": ex["question"], "answer": ex["answer"],
                    "n_paras": len(c["title"]), "units": units})
        if len(out) >= n:
            break
    return out


def _sent_units(title, text):
    from nltk.tokenize import sent_tokenize
    sents = [s for s in sent_tokenize(text) if s.strip()]
    return _units_from_paras([title], [sents])


def build_coqa(n):  # conversations modality: passage + conversational question
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/coqa", split=f"validation[:{n*2}]")
    out = []
    for ex in ds:
        units = _sent_units(ex.get("source", "story"), ex["story"])
        qs, ans = ex["questions"], ex["answers"]["input_text"]
        if len(units) < 3 or not qs:
            continue
        out.append({"question": qs[0], "answer": ans[0], "n_paras": 1, "units": units})
        if len(out) >= n:
            break
    return out


def build_narrativeqa(n):  # documents modality: long narrative summary + question
    from datasets import load_dataset
    ds = load_dataset("deepmind/narrativeqa", split=f"validation[:{n*4}]")
    out, seen_q = [], set()
    for ex in ds:
        doc = ex["document"]
        summ = doc.get("summary", "")
        text = summ.get("text", "") if isinstance(summ, dict) else str(summ)
        q = ex["question"]; qt = q.get("text", "") if isinstance(q, dict) else str(q)
        if not text or qt in seen_q:
            continue
        seen_q.add(qt)
        units = _sent_units(doc.get("kind", "story"), text)
        q = ex["question"]; qt = q.get("text", "") if isinstance(q, dict) else str(q)
        a = ex["answers"][0]; at = a.get("text", "") if isinstance(a, dict) else str(a)
        if len(units) < 5 or not qt:
            continue
        out.append({"question": qt, "answer": at, "n_paras": 1, "units": units})
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["squad", "2wiki", "coqa", "narrativeqa"])
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()
    if not have_key():
        print("NO API KEY"); sys.exit(1)

    builders = {"squad": build_squad, "2wiki": build_2wiki, "coqa": build_coqa, "narrativeqa": build_narrativeqa}
    examples = builders[args.dataset](args.n)
    scorer = KeepScorer.load(os.path.join(DATA, "keep_scorer.joblib"))  # HotpotQA-trained, UNCHANGED
    scores = [scorer.score_example(ex) for ex in examples]
    ubu = [{u["uid"]: u for u in ex["units"]} for ex in examples]
    print(f"{args.dataset}: {len(examples)} examples, avg units={np.mean([len(e['units']) for e in examples]):.0f}, "
          f"avg raw tokens={np.mean([sum(u['tokens'] for u in e['units']) for e in examples]):.0f}")

    conds = [("full", None)]
    for b in [120, 240]:
        conds += [(f"suffix@{b}", b), (f"bm25@{b}", b)]
    conds.append(("random@120", 120))

    def select(name, i, b):
        ex = examples[i]
        if name == "full": return [u["uid"] for u in ex["units"]]
        if name.startswith("suffix"): return select_coverage(ex, scores[i], b)
        if name.startswith("bm25"): return baselines.bm25_topk(ex, b)
        if name.startswith("random"): return baselines.random_select(ex, b, seed=i)

    jobs = []
    for ci, (name, b) in enumerate(conds):
        for i, ex in enumerate(examples):
            uids = select(name, i, b)
            jobs.append({"ci": ci, "name": name, "ctx": render_context(ex, uids),
                         "q": ex["question"], "gold": ex["answer"], "tokens": tokens_of(ubu[i], uids)})
    print(f"  conditions={len(conds)} calls={len(jobs)}")

    def run(j):
        try:
            p = read_answer(j["ctx"], j["q"])
        except Exception:
            p = ""
        j["em"] = squad_em(p, j["gold"]); j["f1"] = squad_f1(p, j["gold"]); return j

    res, done = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for fut in as_completed([pool.submit(run, j) for j in jobs]):
            res.append(fut.result()); done += 1
            if done % 150 == 0: print(f"    {done}/{len(jobs)}")

    agg = {}
    for ci, (name, b) in enumerate(conds):
        rows = [r for r in res if r["ci"] == ci]
        agg[name] = {"avg_tokens": float(np.mean([r["tokens"] for r in rows])),
                     "em": float(np.mean([r["em"] for r in rows])),
                     "f1": float(np.mean([r["f1"] for r in rows])), "n": len(rows)}
    out = {"dataset": args.dataset, "reader": READER_MODEL, "n": len(examples), "conditions": agg}
    json.dump(out, open(os.path.join(DATA, f"crosstask_{args.dataset}.json"), "w"), indent=2)
    print(f"\n=== {args.dataset.upper()} (HotpotQA-trained compressor, unchanged) ===")
    for name, a in agg.items():
        print(f"  {name:14s} tok={a['avg_tokens']:6.0f}  EM={a['em']:.3f}  F1={a['f1']:.3f}")
    full = agg["full"]; s = agg.get("suffix@240")
    if s: print(f"  -> retains {100*s['f1']/full['f1']:.0f}% of full F1 at {full['avg_tokens']/s['avg_tokens']:.1f}x fewer tokens")
    print(f"saved data/crosstask_{args.dataset}.json")


if __name__ == "__main__":
    main()
