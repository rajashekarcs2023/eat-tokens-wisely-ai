"""Thin Anthropic client helper. Loads .env (key is present there but not
exported), exposes a frozen reader and a token counter. The reader is a
measurement instrument held identical across all arms — never a judge.
"""

from __future__ import annotations

import os

READER_MODEL = "claude-haiku-4-5"


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def have_key() -> bool:
    load_env()
    return len(os.environ.get("ANTHROPIC_API_KEY", "")) > 0


_client = None


def client():
    global _client
    if _client is None:
        load_env()
        import anthropic

        _client = anthropic.Anthropic()
    return _client


_SYS = (
    "You answer questions from the provided context only. "
    "Reply with the shortest exact answer (a name, phrase, number, or yes/no). "
    "No explanation."
)


def read_answer(context: str, question: str, model: str = READER_MODEL, max_retries: int = 4) -> str:
    import time

    msg = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    delay = 1.0
    for attempt in range(max_retries):
        try:
            r = client().messages.create(
                model=model,
                max_tokens=32,
                temperature=0,  # deterministic reader → single-click demos reproduce across runs
                system=_SYS,
                messages=[{"role": "user", "content": msg}],
            )
            return r.content[0].text.strip()
        except Exception as e:  # rate limit / transient
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return ""


def count_claude_tokens(text: str, model: str = READER_MODEL) -> int:
    r = client().messages.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    )
    return r.input_tokens
