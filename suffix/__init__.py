"""SUFFIX — a verifiable extractive context codec.

Compression as lossy source coding under a task distribution: pick the smallest
set of *original, verbatim* spans that preserves the downstream answer, chosen by
budgeted saturating-coverage selection over a learned relevance signal.

No generative model sits in the compression hot path, so output is a verbatim
subsequence of the input (hallucination-free by construction) and is measured
against gold labels, not an LLM judge.
"""

__all__ = [
    "tokens",
    "text_utils",
    "structural",
    "features",
    "scorer",
    "coverage",
    "baselines",
    "metrics",
    "pipeline",
]
