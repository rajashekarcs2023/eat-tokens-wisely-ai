"""Lightweight, dependency-free text utilities: tokenization, content words,
query-anchor extraction (for query-conditioned need-slots), and SQuAD-style
answer normalization. All LLM-free and deterministic.
"""

from __future__ import annotations

import re
import string
from typing import List

# A compact English stopword list — enough to strip function words from
# lexical-overlap features and anchor extraction without an NLTK dependency.
STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of to in on at by for with about
    against between into through during before after above below from up down out
    over under again further is are was were be been being have has had do does did
    doing this that these those i you he she it we they them his her its their our
    your my me him us as not no nor so than too very can will just don should now
    which who whom whose what where why how all any both each few more most other
    some such only own same s t d ll m re ve y also got get one two
    """.split()
)

# generic task/instruction verbs that make poor need-slot anchors
TASK_WORDS = frozenset(
    "fix bug issue problem error make add update change implement build create "
    "find debug resolve handle write generate".split()
)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
# A proper-noun / multi-word-entity run: capitalised words possibly joined by
# small connectors (of/the/and/'s) — e.g. "Kiss and Tell", "University of Texas".
_ENTITY_RE = re.compile(
    r"\b([A-Z][\w'.]*(?:\s+(?:of|the|and|for|de|von|van|'s|&)\b\s*|\s+[A-Z][\w'.]*)*)"
)
_QUOTED_RE = re.compile(r"[\"“‘']([^\"”’']{2,})[\"”’']")
_NUM_RE = re.compile(r"\b\d[\d,.\-/]*\b")


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def content_tokens(text: str) -> List[str]:
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 1]


def numbers(text: str) -> List[str]:
    return [n.strip(".,/-") for n in _NUM_RE.findall(text)]


def cap_tokens(text: str) -> List[str]:
    """Capitalised tokens (rough entity-density proxy), excluding sentence start
    is ignored for simplicity — density, not precision, is the goal."""
    return re.findall(r"\b[A-Z][a-zA-Z']+", text)


def extract_anchors(question: str, max_anchors: int = 8) -> List[List[str]]:
    """Decompose a query into need-slot anchors, each a list of content tokens.

    Anchors come ONLY from the question (no gold, no LLM) — this is what makes
    the selection query-conditioned and the eval leakage-free. We capture BOTH:
      - multi-word entities / quoted / number spans (the things to look up), AND
      - the remaining salient content words (the answer-type predicate, e.g.
        'government position', 'nationality') which a multi-hop answer span must
        satisfy but which are not named entities.
    Covering a predicate anchor is exactly what pulls in the second-hop fact.
    """
    spans: List[str] = []
    spans += [m.group(1).strip() for m in _ENTITY_RE.finditer(question)]
    spans += [m.group(1).strip() for m in _QUOTED_RE.finditer(question)]
    spans += _NUM_RE.findall(question)

    anchors: List[List[str]] = []
    seen = set()
    covered_tokens = set()
    for s in spans:
        toks = tuple(content_tokens(s))
        if not toks or toks in seen:
            continue
        tset = set(toks)
        if any(tset.issubset(set(a)) for a in anchors):  # subsumed by a kept anchor
            continue
        seen.add(toks)
        anchors.append(list(toks))
        covered_tokens |= tset

    # Add remaining salient content words as singleton anchors (predicates,
    # answer-type words) so coverage has a slot for the second hop. Skip generic
    # task verbs ('fix', 'bug', ...) which make poor need-slots.
    for w in content_tokens(question):
        if w in covered_tokens or (w,) in seen or w in TASK_WORDS:
            continue
        seen.add((w,))
        anchors.append([w])
        covered_tokens.add(w)

    return anchors[:max_anchors] if anchors else [content_tokens(question) or ["_"]]


# ---- SQuAD / HotpotQA official answer normalization (for EM / token-F1) ----

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())
