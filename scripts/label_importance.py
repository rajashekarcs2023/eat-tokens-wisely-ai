"""Label B (SPEC v2 §4): LLMLingua-2-style query-agnostic IMPORTANCE.

The teacher (Claude) marks which sentences carry essential information to keep the
document's meaning — with NO question and NO answer feedback. This is the
importance-distillation signal LLMLingua-2 uses, and the fair contrast to:
  A = human supporting_facts, C = reader-grounded answer-impact.

One call per example. Cached. Output: per-unit keep/drop over ALL units.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix.llm import client, have_key  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WORKERS = 10
MODEL = "claude-haiku-4-5"

_SYS = (
    "You select the sentences that carry essential information. You never write new text. "
    "You only return sentence numbers."
)


def label_example(example):
    units = example["units"]
    listing = "\n".join(f"[{i}] {u['text'].strip()}" for i, u in enumerate(units))
    msg = (
        "Below are numbered sentences from a document. Select the sentences that carry the "
        "essential information — the ones you would keep if compressing the document to about "
        "half its length to preserve its meaning. Consider only the text itself (there is no "
        "question).\n\nReturn ONLY a JSON array of the sentence numbers to KEEP, e.g. [0,3,7]. "
        "No other text.\n\n" + listing
    )
    r = client().messages.create(
        model=MODEL, max_tokens=400, system=_SYS,
        messages=[{"role": "user", "content": msg}],
    )
    text = r.content[0].text
    idxs = set(int(x) for x in re.findall(r"\d+", text) if int(x) < len(units))
    if not idxs:  # fallback: keep nothing parseable -> all drop (degenerate); flag
        return {"labels": {u["uid"]: 0 for u in units}, "n_keep": 0, "parsed": False}
    labels = {u["uid"]: (1 if i in idxs else 0) for i, u in enumerate(units)}
    return {"labels": labels, "n_keep": len(idxs), "parsed": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    args = ap.parse_args()
    if not have_key():
        print("NO API KEY"); sys.exit(1)

    examples = json.load(open(os.path.join(DATA, f"hotpot_{args.split}.json")))[: args.n]
    cache_path = os.path.join(DATA, f"importance_labels_{args.split}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [ex for ex in examples if ex["id"] not in cache]
    print(f"importance-labeling {len(todo)} examples (cached {len(examples)-len(todo)})")

    def work(ex):
        try:
            return ex["id"], label_example(ex)
        except Exception as e:
            return ex["id"], {"error": str(e)[:120]}

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for fut in as_completed([pool.submit(work, ex) for ex in todo]):
            eid, res = fut.result()
            cache[eid] = res
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(todo)}")
    json.dump(cache, open(cache_path, "w"))

    used = [cache[ex["id"]] for ex in examples if ex["id"] in cache]
    valid = [r for r in used if "labels" in r]
    parsed = sum(1 for r in valid if r.get("parsed"))
    avg_keep = sum(r["n_keep"] for r in valid) / max(1, len(valid))
    print(f"\nvalid={len(valid)}  parsed_ok={parsed}  avg sentences kept/ex={avg_keep:.1f}")
    print(f"saved -> {cache_path}")


if __name__ == "__main__":
    main()
