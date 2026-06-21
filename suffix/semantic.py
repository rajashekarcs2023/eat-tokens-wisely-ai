"""Optional semantic reranking — closes the lexical-overlap gap in the extractive
selector by adding an embedding-cosine signal to the learned score.

Generation-free: embeddings only, NO LLM, so it stays hallucination-free and cheap.
OPTIONAL and safe: if no embedding backend is installed, every function degrades to
a no-op and the pipeline behaves exactly as the lexical-only version. Backends are
auto-detected in order: fastembed (ONNX, light) -> sentence-transformers.

Why: the lexical scorer ranks by term overlap with the question, so it can miss an
answer phrased differently from the query ("how many minutes?" vs "a 480-second
window"). A small embedding model matches on meaning, not words, and catches it.
"""
from __future__ import annotations

import numpy as np

_MODEL = None
_BACKEND = None  # None=untried, "none"=no backend, "fastembed"/"st"=loaded


def _load():
    global _MODEL, _BACKEND
    if _BACKEND is not None:
        return
    try:
        from fastembed import TextEmbedding
        _MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        _BACKEND = "fastembed"
        return
    except Exception:
        pass
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _BACKEND = "st"
        return
    except Exception:
        _BACKEND = "none"


def available() -> bool:
    _load()
    return _BACKEND not in (None, "none")


def backend() -> str:
    _load()
    return _BACKEND or "none"


def embed(texts) -> np.ndarray:
    _load()
    if _BACKEND == "fastembed":
        return np.asarray(list(_MODEL.embed(list(texts))), dtype=np.float32)
    if _BACKEND == "st":
        return np.asarray(_MODEL.encode(list(texts), normalize_embeddings=False), dtype=np.float32)
    raise RuntimeError("no embedding backend available")


def semantic_scores(query: str, texts) -> np.ndarray | None:
    """Cosine similarity of each text to the query, mapped to [0, 1].
    Returns None (no-op) when no embedding backend is installed."""
    texts = list(texts)
    if not texts or not available():
        return None
    vecs = embed([query] + texts)
    q, T = vecs[0], vecs[1:]
    qn = q / (np.linalg.norm(q) + 1e-9)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    sims = Tn @ qn                       # [-1, 1]
    return (sims + 1.0) / 2.0            # [0, 1]


def blend(learned_scores: np.ndarray, query: str, texts, weight: float = 0.5) -> np.ndarray:
    """Blend the learned (lexical) keep-scores with semantic similarity.
    weight=0 -> pure lexical (unchanged); weight=1 -> pure semantic. No-op if no backend."""
    sem = semantic_scores(query, texts)
    if sem is None:
        return learned_scores
    return (1.0 - weight) * np.asarray(learned_scores, dtype=np.float32) + weight * sem
