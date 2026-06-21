"""Per-unit feature extraction for the learned keep-scorer.

Every feature is cheap, deterministic, and LLM-free. The scorer takes only
(question, unit) signals at inference — never gold — so there is no label leakage.
Features fuse lexical (per-example TF-IDF cosine, content-word overlap), entity
(capitalised-token / number overlap, title match), and structural (paragraph &
sentence position, length) signals so the LogReg can outperform any single
lexical baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_utils import cap_tokens, content_tokens, numbers

FEATURE_NAMES = [
    "tfidf_cos_q",      # per-example TF-IDF cosine(unit, question)
    "jaccard_q",        # content-word Jaccard with question
    "shared_words",     # raw shared content-word count
    "num_overlap",      # shared numbers/years with question
    "cap_overlap",      # shared capitalised tokens (entity overlap) with question
    "title_in_q",       # fraction of the unit's paragraph-title words in question
    "para_pos",         # paragraph index, normalised
    "sent_idx",         # sentence index within paragraph
    "is_first_sent",    # topic-sentence flag
    "sent_pos",         # sentence index normalised within paragraph
    "n_content",        # content-word count (length proxy)
    "n_tokens",         # tiktoken length
    "has_number",       # contains a digit
    "entity_density",   # count of capitalised tokens
]


def _tfidf_cos_to_query(texts, query):
    """Per-example TF-IDF cosine of each unit text to the query. Fitting on the
    example's own units keeps salience local and needs no global corpus."""
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    try:
        matrix = vec.fit_transform(texts + [query])
    except ValueError:  # e.g. all-stopword/empty corpus
        return np.zeros(len(texts))
    unit_m = matrix[:-1]
    q_m = matrix[-1]
    return cosine_similarity(unit_m, q_m).ravel()


def example_features(example) -> np.ndarray:
    units = example["units"]
    q = example["question"]
    texts = [u["text"] for u in units]
    n_paras = max(1, example.get("n_paras", 1) - 1)

    cos_q = _tfidf_cos_to_query(texts, q)
    q_content = set(content_tokens(q))
    q_nums = set(numbers(q))
    q_caps = set(w.lower() for w in cap_tokens(q))

    rows = []
    for i, u in enumerate(units):
        u_content = set(content_tokens(u["text"]))
        u_nums = set(numbers(u["text"]))
        u_caps = set(w.lower() for w in cap_tokens(u["text"]))
        title_tokens = set(content_tokens(u["title"]))
        shared = len(u_content & q_content)
        union = len(u_content | q_content) or 1
        n_in_para = max(1, u.get("n_sent_in_para", 1) - 1)
        rows.append(
            [
                cos_q[i],
                shared / union,
                shared,
                len(u_nums & q_nums),
                len(u_caps & q_caps),
                (len(title_tokens & q_content) / (len(title_tokens) or 1)),
                u["para"] / n_paras,
                float(u["sent"]),
                1.0 if u["sent"] == 0 else 0.0,
                (u["sent"] / n_in_para) if u.get("n_sent_in_para", 1) > 1 else 0.0,
                float(len(u_content)),
                float(u["tokens"]),
                1.0 if u_nums else 0.0,
                float(len(u_caps)),
            ]
        )
    return np.asarray(rows, dtype=float)


def tfidf_cosines(example) -> np.ndarray:
    """Exposed for the TF-IDF top-k baseline (reuses the same per-example cosine)."""
    texts = [u["text"] for u in example["units"]]
    return _tfidf_cos_to_query(texts, example["question"])
