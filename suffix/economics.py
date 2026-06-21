"""Compression economics: tokens + dollars saved per 1M queries.

Prices are the published Anthropic per-1M-token rates (input/output), sourced from
the canonical claude-api model table (cached 2026-06-04). Kept in ONE place,
dated, so they are trivially updatable — never silently hardcoded from memory.

Compression saves the *same tokens* regardless of which model reads them, so the
dollar figure scales with the reader's input price — we show all three tiers to
make the "scales to the expensive model" point honestly.

Honesty note: token counts here are cl100k (our measurement ruler), which
UNDERCOUNTS real Claude tokens by ~15-20% on prose — so these savings are
conservative. The ratio (kept/total) is tokenizer-robust regardless.
"""

from __future__ import annotations

# $ per 1M tokens (input, output). Source: claude-api model table, cached 2026-06-04.
PRICES = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "label": "Haiku 4.5"},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "label": "Sonnet 4.6"},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00, "label": "Opus 4.8"},
}
PRICE_SOURCE = "Anthropic published rates, claude-api table cached 2026-06-04"


def savings(full_tokens: float, compressed_tokens: float, queries: int = 1_000_000):
    """Tokens + $ saved at the given query volume, across model tiers."""
    saved_per_query = max(0.0, full_tokens - compressed_tokens)
    tokens_saved_total = saved_per_query * queries
    rows = []
    for model, p in PRICES.items():
        # $ per 1M queries = saved_per_query(tokens) * price($/1M tok) * (queries/1e6)
        dollars = saved_per_query * p["input"] * (queries / 1_000_000)
        rows.append({
            "model": model, "label": p["label"], "input_price_per_mtok": p["input"],
            "dollars_saved": round(dollars, 2),
        })
    return {
        "full_tokens": round(full_tokens, 1),
        "compressed_tokens": round(compressed_tokens, 1),
        "saved_per_query": round(saved_per_query, 1),
        "compression_x": round(full_tokens / compressed_tokens, 2) if compressed_tokens else None,
        "tokens_saved_per_1M_queries": round(tokens_saved_total),
        "queries": queries,
        "by_model": rows,
        "price_source": PRICE_SOURCE,
        "ruler_note": "cl100k tokens undercount real Claude tokens ~15-20% on prose; savings are conservative.",
    }
