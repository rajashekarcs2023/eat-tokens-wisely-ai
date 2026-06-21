"""Controlled redundancy stress-test (zero API).

HotpotQA distractors are different topics, not duplicates, so near-dup
suppression is correctly inactive there. Real agent context is the opposite:
paginated tool output, repeated log lines, near-duplicate threads. We simulate
that honestly: into each example we inject R near-duplicate copies of the single
most relevant-looking NON-gold sentence (the "spam" a relevance ranker will
hoard). Gold facts are untouched.

At a fixed token budget we then measure gold-fact recall as redundancy R grows:
  - score top-k / BM25 top-k spend budget on duplicate copies -> recall collapses
  - SUFFIX (score + near-dup suppression) skips the copies -> recall holds
This isolates exactly the variable the suppressor addresses.
"""

from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix import baselines  # noqa: E402
from suffix.features import tfidf_cosines  # noqa: E402
from suffix.metrics import supporting_fact_prf, tokens_of  # noqa: E402
from suffix.scorer import KeepScorer  # noqa: E402
from suffix.text_utils import content_tokens  # noqa: E402
from suffix.tokens import count_tokens  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BUDGET = 80
LEVELS = [0, 2, 4, 6, 8, 12]
N_SPAM_FAMILIES = 1  # one relevant passage surfaced many times (duplicate retrieval)


def inject_dups(ex, r, scorer):
    """Model duplicate RETRIEVAL: the same relevant passage surfaced r times by
    different tools. Each copy keeps high-scoring features (a question-echoing
    'source' title and lead position) so a relevance ranker rates copies as
    highly as the original and hoards them. Gold facts are untouched; copies are
    near-identical non-gold spans the suppressor should collapse.
    """
    if r == 0:
        return ex
    gold = set(ex["gold_uids"])
    scores = scorer.score_example(ex)
    order = np.argsort(-scores)
    spam_units = [ex["units"][i] for i in order if ex["units"][i]["uid"] not in gold][:N_SPAM_FAMILIES]
    # a title built from the question's content words -> high title_in_q feature
    qtitle = " ".join(content_tokens(ex["question"])[:4]).title() or "Retrieved Source"
    new = copy.deepcopy(ex)
    base_para = ex["n_paras"]
    for fam, spam in enumerate(spam_units):
        for k in range(r):
            text = spam["text"].rstrip()  # exact duplicate retrieval
            new["units"].append({
                "uid": f"dup{fam}_{k}", "para": base_para + fam * r + k, "sent": 0,
                "n_sent_in_para": 1, "title": f"{qtitle} (source {fam}.{k})", "text": text,
                "char_start": -1, "char_end": -1,
                "tokens": count_tokens(text), "is_gold": False,
            })
    new["n_units"] = len(new["units"])
    new["n_paras"] = base_para + N_SPAM_FAMILIES * r
    return new


def main():
    test = json.load(open(os.path.join(DATA, "hotpot_test.json")))
    scorer = KeepScorer.load(os.path.join(DATA, "keep_scorer.joblib"))
    methods = ["suffix", "no_coverage", "bm25_topk"]
    out = {"budget": BUDGET, "levels": LEVELS, "recall": {m: [] for m in methods},
           "tokens_on_dups": {m: [] for m in methods}}

    for r in LEVELS:
        rec = {m: [] for m in methods}
        dupfrac = {m: [] for m in methods}
        for ex in test:
            pex = inject_dups(ex, r, scorer)
            scores = scorer.score_example(pex)
            ubu = {u["uid"]: u for u in pex["units"]}
            sels = {
                "suffix": baselines.suffix(pex, BUDGET, scores),
                "no_coverage": baselines.score_topk(pex, BUDGET, scores),
                "bm25_topk": baselines.bm25_topk(pex, BUDGET),
            }
            for m in methods:
                sel = sels[m]
                rec[m].append(supporting_fact_prf(sel, ex["gold_uids"])["recall"])
                dup_tokens = sum(ubu[u]["tokens"] for u in sel if u.startswith("dup"))
                total = tokens_of(ubu, sel) or 1
                dupfrac[m].append(dup_tokens / total)
        for m in methods:
            out["recall"][m].append(float(np.mean(rec[m])))
            out["tokens_on_dups"][m].append(float(np.mean(dupfrac[m])))
        print(f"R={r}: " + "  ".join(f"{m}={np.mean(rec[m]):.2f}(dup%={np.mean(dupfrac[m]):.0%})" for m in methods))

    json.dump(out, open(os.path.join(DATA, "redundancy.json"), "w"), indent=2)
    print("\nsaved data/redundancy.json")


if __name__ == "__main__":
    main()
