"""Token accounting.

We use tiktoken's cl100k_base as a *stable ruler* for relative comparison
(compression ratio), NOT as ground-truth Claude token counts. cl100k_base is
GPT-4's tokenizer and undercounts Claude tokens; every headline number we report
is a RATIO (kept/total), which is robust to that bias. For one absolute number in
Claude's real currency we cross-check with messages.count_tokens elsewhere.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _enc():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_enc().encode(text))


def count_units_tokens(units) -> int:
    """Total tokens of a list of unit dicts (uses cached per-unit `tokens`)."""
    total = 0
    for u in units:
        t = u.get("tokens")
        total += t if t is not None else count_tokens(u["text"])
    return total
