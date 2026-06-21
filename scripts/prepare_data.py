"""Cache HotpotQA (distractor) into a local, network-free spine.

For each example we build:
  - raw_doc:   the exact text an agent would dump if it sent everything
  - units:     one per sentence, each with a char-offset span into raw_doc
               (so raw_doc[char_start:char_end] == unit.text  -- exact provenance)
  - is_gold:   from human-labelled supporting_facts (title, sent_id) pairs
  - answer:    gold answer string (for downstream EM/F1)

We pull the TRAIN split for the scorer's training set and the VALIDATION split
for the held-out test set, so no example is ever both trained and evaluated on.

Usage:
    python scripts/prepare_data.py --train 400 --test 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix.tokens import count_tokens  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def build_example(ex: dict) -> dict | None:
    ctx = ex.get("context") or {}
    titles = ctx.get("title") or []
    sentences = ctx.get("sentences") or []
    sf = ex.get("supporting_facts") or {}
    gold = set(zip(sf.get("title", []), sf.get("sent_id", [])))
    if not titles or not sentences or not gold:
        return None

    parts: list[str] = []
    pos = 0
    units: list[dict] = []

    def emit(s: str) -> int:
        nonlocal pos
        start = pos
        parts.append(s)
        pos += len(s)
        return start

    for pi, (title, para) in enumerate(zip(titles, sentences)):
        emit(f"== {title} ==\n")
        for si, stext in enumerate(para):
            start = emit(stext)
            if not stext.endswith((" ", "\n")):
                emit(" ")
            units.append(
                {
                    "uid": f"p{pi}s{si}",
                    "para": pi,
                    "sent": si,
                    "n_sent_in_para": len(para),
                    "title": title,
                    "text": stext,
                    "char_start": start,
                    "char_end": start + len(stext),
                    "tokens": count_tokens(stext),
                    "is_gold": (title, si) in gold,
                }
            )
        emit("\n\n")

    raw_doc = "".join(parts)
    # Hard provenance check: every unit must be a byte-exact slice of raw_doc.
    for u in units:
        assert raw_doc[u["char_start"] : u["char_end"]] == u["text"], "provenance mismatch"

    gold_uids = [u["uid"] for u in units if u["is_gold"]]
    if not gold_uids:  # gold title/sent_id didn't line up with context; skip
        return None

    return {
        "id": ex.get("id"),
        "question": ex["question"],
        "answer": ex.get("answer", ""),
        "type": ex.get("type", ""),
        "level": ex.get("level", ""),
        "n_paras": len(titles),
        "raw_doc": raw_doc,
        "raw_tokens": count_tokens(raw_doc),
        "units": units,
        "n_units": len(units),
        "gold_uids": gold_uids,
    }


def cache_split(split: str, n: int, out_path: str) -> int:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)
    out: list[dict] = []
    seen = 0
    for ex in ds:
        seen += 1
        built = build_example(ex)
        if built is not None:
            out.append(built)
        if len(out) >= n:
            break
        if seen > n * 3:  # safety: don't stream forever if many are skipped
            break

    with open(out_path, "w") as f:
        json.dump(out, f)
    avg_units = sum(e["n_units"] for e in out) / max(1, len(out))
    avg_gold = sum(len(e["gold_uids"]) for e in out) / max(1, len(out))
    avg_tok = sum(e["raw_tokens"] for e in out) / max(1, len(out))
    print(
        f"[{split}] cached {len(out)} examples -> {out_path}\n"
        f"        avg units/ex={avg_units:.1f}  avg gold/ex={avg_gold:.1f}  "
        f"avg raw_tokens/ex={avg_tok:.0f}  gold_frac={avg_gold/avg_units:.1%}"
    )
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--test", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_split("train", args.train, os.path.join(DATA_DIR, "hotpot_train.json"))
    cache_split("validation", args.test, os.path.join(DATA_DIR, "hotpot_test.json"))


if __name__ == "__main__":
    main()
