"""Train the learned keep-scorer on gold supporting_facts and prove it earns its
place: report held-out AUC vs the TF-IDF and BM25 lexical baselines at the unit
level. If the learned scorer does not beat lexical ranking, we say so and lean on
the coverage layer — but we measure it either way.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix.baselines import _bm25_scores  # noqa: E402
from suffix.features import tfidf_cosines  # noqa: E402
from suffix.scorer import KeepScorer  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load(name):
    return json.load(open(os.path.join(DATA, name)))


def unit_level_auc(examples, score_fn):
    y, s = [], []
    for ex in examples:
        y.extend([int(u["is_gold"]) for u in ex["units"]])
        s.extend(list(score_fn(ex)))
    y, s = np.array(y), np.array(s)
    return roc_auc_score(y, s), average_precision_score(y, s)


def main():
    train = load("hotpot_train.json")
    test = load("hotpot_test.json")
    print(f"train={len(train)} test={len(test)} examples")

    scorer = KeepScorer().fit(train)
    scorer.save(os.path.join(DATA, "keep_scorer.joblib"))

    learned_auc, learned_ap = unit_level_auc(test, scorer.score_example)
    tfidf_auc, tfidf_ap = unit_level_auc(test, tfidf_cosines)
    bm25_auc, bm25_ap = unit_level_auc(test, _bm25_scores)

    print("\n=== held-out unit-level ranking (predict gold supporting fact) ===")
    print(f"  learned scorer : AUC={learned_auc:.3f}  AP={learned_ap:.3f}")
    print(f"  tfidf cosine   : AUC={tfidf_auc:.3f}  AP={tfidf_ap:.3f}")
    print(f"  bm25           : AUC={bm25_auc:.3f}  AP={bm25_ap:.3f}")
    verdict = "LEARNED SCORER WINS" if learned_auc > max(tfidf_auc, bm25_auc) else "LEXICAL TIES/WINS -> lean on coverage"
    print(f"  -> {verdict}")

    print("\n=== top learned coefficients (|weight|) ===")
    for name, w in scorer.coef_table()[:8]:
        print(f"  {name:16s} {w:+.3f}")

    out = {
        "learned_auc": learned_auc, "learned_ap": learned_ap,
        "tfidf_auc": tfidf_auc, "tfidf_ap": tfidf_ap,
        "bm25_auc": bm25_auc, "bm25_ap": bm25_ap,
        "verdict": verdict,
        "top_coef": [[n, float(w)] for n, w in scorer.coef_table()],
    }
    json.dump(out, open(os.path.join(DATA, "scorer_report.json"), "w"), indent=2)
    print("\nsaved data/keep_scorer.joblib + data/scorer_report.json")


if __name__ == "__main__":
    main()
