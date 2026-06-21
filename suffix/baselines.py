"""Baselines and ablations — every one runs in OUR harness on the same units,
the same token budgets, and the same gold metric, so the Pareto curve is
apples-to-apples. (Prior-work paper numbers are never plotted on our axes.)

  full           : keep everything (quality ceiling, cost to beat)
  random         : random spans under budget (sanity floor)
  tfidf_topk     : rank by per-example TF-IDF cosine to the query (the lexical bar)
  bm25_topk      : rank by Okapi BM25 to the query
  score_topk     : learned scorer, plain top-k  == NO-COVERAGE ablation
  coverage_noscore: coverage with rel based on anchor match only == NO-SCORER ablation
  suffix         : learned scorer + saturating coverage  (ours)
"""

from __future__ import annotations

import math

import numpy as np

from .coverage import select_coverage
from .text_utils import content_tokens


def _take_under_budget(units, order, budget):
    """Add units in `order` while cumulative tokens stay within budget."""
    out, used = [], 0
    for i in order:
        t = units[i]["tokens"]
        if used + t <= budget:
            out.append(i)
            used += t
    return out


def _emit(units, idxs):
    idxs = sorted(idxs, key=lambda i: (units[i]["para"], units[i]["sent"]))
    return [units[i]["uid"] for i in idxs]


def full(example, *_):
    return [u["uid"] for u in example["units"]]


def random_select(example, budget, seed=0):
    units = example["units"]
    rng = np.random.RandomState(seed)
    order = list(rng.permutation(len(units)))
    return _emit(units, _take_under_budget(units, order, budget))


def tfidf_topk(example, budget, cosines=None):
    from .features import tfidf_cosines

    units = example["units"]
    cos = cosines if cosines is not None else tfidf_cosines(example)
    order = list(np.argsort(-cos))
    return _emit(units, _take_under_budget(units, order, budget))


def _bm25_scores(example, k1=1.5, b=0.75):
    units = example["units"]
    docs = [content_tokens(u["text"]) for u in units]
    q = set(content_tokens(example["question"]))
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, N)
    df = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    scores = np.zeros(N)
    for i, d in enumerate(docs):
        if not d:
            continue
        tf = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        dl = len(d)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * (tf[w] * (k1 + 1)) / (tf[w] + k1 * (1 - b + b * dl / avgdl))
        scores[i] = s
    return scores


def bm25_topk(example, budget):
    units = example["units"]
    order = list(np.argsort(-_bm25_scores(example)))
    return _emit(units, _take_under_budget(units, order, budget))


def score_topk(example, budget, scores):
    """NO-COVERAGE ablation: learned scores, plain top-k."""
    units = example["units"]
    order = list(np.argsort(-scores))
    return _emit(units, _take_under_budget(units, order, budget))


def coverage_no_scorer(example, budget):
    """NO-SCORER ablation: saturating coverage with uniform relevance."""
    ones = np.ones(len(example["units"]))
    return select_coverage(example, ones, budget)


def suffix(example, budget, scores):
    """Ours: learned keep-scorer + saturating coverage."""
    return select_coverage(example, scores, budget)
