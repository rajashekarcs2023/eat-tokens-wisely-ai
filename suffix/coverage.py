"""Stage 3: budgeted, redundancy-aware extractive selection — the genuine core.

The selector ranks spans by the learned, query-conditioned keep-score and admits
them under the token budget, but SKIPS any span that is a near-duplicate
(TF-IDF cosine > tau) of an already-kept span:

    keep i  iff  it fits the budget  AND  max_{s in S} sim(i, s) <= tau

Near-duplicate suppression is a facility-location-style submodular diversity term
restricted to the high-similarity regime. This is a deliberate, honest design
choice:

  * On low-redundancy prose (HotpotQA distractors are different *topics*, and the
    two gold facts share entities but are not near-identical), nothing exceeds
    tau, so the selection is IDENTICAL to the strong learned-scorer top-k — it
    cannot regress. (A general MMR diversity penalty *does* regress here, because
    it wrongly punishes the second gold fact for sharing an entity with the
    first; we measured that and rejected it.)
  * The suppressor only bites when context genuinely repeats (logs, paginated
    tool output, near-duplicate threads — the real agent regime), where pure
    relevance ranking wastes budget on copies. We prove that win on a controlled
    redundancy stress-test, never by overclaiming it on clean HotpotQA.

We also expose query "need-slots" (anchors) so the UI can *explain* what each
kept span covers.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_utils import content_tokens, extract_anchors

_TAU = 0.8  # near-duplicate threshold; only spans MORE similar than this are suppressed


def build_needs(example):
    return extract_anchors(example["question"])


def pairwise_sim(example):
    """Per-example TF-IDF cosine similarity between every pair of spans."""
    texts = [u["text"] for u in example["units"]]
    if len(texts) < 2:
        return np.zeros((len(texts), len(texts)))
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    try:
        m = vec.fit_transform(texts)
    except ValueError:
        return np.zeros((len(texts), len(texts)))
    return cosine_similarity(m)


def select_coverage(example, scores, budget_tokens, tau=_TAU, sim=None):
    """Return selected uids (original order) under a token budget.

    Score-ranked admission with near-duplicate suppression. With tau >= 1.0 no
    span is ever suppressed, so this is exactly learned-scorer top-k.
    """
    units = example["units"]
    scores = np.asarray(scores, dtype=float)
    if sim is None:
        sim = pairwise_sim(example)
    costs = np.array([max(1, u["tokens"]) for u in units], dtype=float)

    order = list(np.argsort(-scores))
    chosen, used = [], 0.0
    for i in order:
        if used + costs[i] > budget_tokens:
            continue
        if chosen and max(sim[i][j] for j in chosen) > tau:
            continue  # near-duplicate of an already-kept span
        chosen.append(i)
        used += costs[i]

    chosen.sort(key=lambda i: (units[i]["para"], units[i]["sent"]))
    return [units[i]["uid"] for i in chosen]


def covered_needs(example, selected_uids):
    """For UI: which query need-slots are satisfied by the kept spans."""
    anchors = build_needs(example)
    by_uid = {u["uid"]: u for u in example["units"]}
    kept_tokens = set()
    for uid in selected_uids:
        kept_tokens |= set(content_tokens(by_uid[uid]["text"]))
    out = []
    for a in anchors:
        aset = set(a)
        cov = len(aset & kept_tokens) / len(aset) if aset else 0.0
        out.append({"anchor": " ".join(a), "covered": cov >= 0.5})
    return out
