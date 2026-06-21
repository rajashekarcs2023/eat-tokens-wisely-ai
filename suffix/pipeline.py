"""Live compression pipeline for arbitrary pasted text or JSON tool output.

Glues the pieces into one call the demo/server uses:
  raw  ->  [structural lossless codec if JSON]  ->  unitize (with provenance)
       ->  learned keep-score  ->  budgeted selection + near-dup suppression
       ->  verbatim kept spans + per-span explainability.

Everything here is CPU-only and sub-second; no LLM is invoked in the hot path.
"""

from __future__ import annotations

import json
import os
import re

from .coverage import covered_needs, select_coverage
from .scorer import KeepScorer
from .structural import render_compact, structural_report
from .tokens import count_tokens

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_scorer = None


def get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = KeepScorer.load(os.path.join(_DATA, "keep_scorer.joblib"))
    return _scorer


_SENT_SPLIT = re.compile(r".+?(?:[.!?](?=\s|$)|$)", re.S)


def unitize_text(text, default_title=""):
    """Split text into sentence/line spans with char-offset provenance."""
    units = []
    para = 0
    pos = 0  # advancing cursor so duplicate sentences get distinct char offsets
    for line in text.split("\n"):
        if not line.strip():
            para += 1
            continue
        sents = [s.strip() for s in _SENT_SPLIT.findall(line.strip()) if len(s.strip()) >= 3]
        for si, s in enumerate(sents):
            start = text.find(s, pos)
            if start < 0:
                start = text.find(s)
            pos = start + len(s) if start >= 0 else pos
            units.append({
                "para": para, "sent": si, "n_sent_in_para": max(1, len(sents)),
                "title": default_title, "text": s,
                "char_start": start, "char_end": start + len(s),
                "tokens": count_tokens(s),
            })
        para += 1
    return units


_URL_RE = re.compile(r"^\s*https?://\S+\s*$")
_CODE_HINT = re.compile(r"[{};=]|=>|\(\)")


def _is_prose(s):
    from .text_utils import content_tokens

    if _URL_RE.match(s):
        return False
    return len(content_tokens(s)) >= 4  # drops URLs, ids, short labels


def units_from_json(obj):
    """Pull prose/code string leaves out of a JSON tool payload, titled by their
    JSON path so the learned scorer's title features still apply. URLs and short
    id-like leaves are skipped; code-looking leaves are kept whole (not split)."""
    units = []
    counter = {"para": 0}

    def add(text, title, path, sent, n):
        units.append({
            "para": counter["para"], "sent": sent, "n_sent_in_para": n,
            "title": str(title), "text": text,
            "char_start": -1, "char_end": -1, "tokens": count_tokens(text),
            "path": ".".join(path),
        })

    def walk(o, path):
        if isinstance(o, str):
            s = o.strip()
            if len(s) >= 12 and _is_prose(s):
                title = next((seg for seg in reversed(path) if not seg.isdigit()), "field")
                if _CODE_HINT.search(s):  # keep code as a single span
                    add(s, title, path, 0, 1)
                else:
                    sents = [x.strip() for x in _SENT_SPLIT.findall(s) if len(x.strip()) >= 3]
                    for si, seg in enumerate(sents):
                        add(seg, title, path, si, max(1, len(sents)))
                counter["para"] += 1
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [k])
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, path + [str(i)])

    walk(obj, [])
    return units


def _render(units_in_order):
    out, cur = [], None
    for u in units_in_order:
        t = u.get("title") or ""
        if t and t != cur:
            cur = t
            out.append(f"## {t}")
        out.append(u["text"])
    return "\n".join(out)


def compress(task: str, raw, budget: int = 200, semantic: bool = False,
             safe: bool = False, min_coverage: float = 0.6):
    is_json = isinstance(raw, (dict, list))
    if is_json:
        # Structured data -> the LOSSLESS structural codec is the right tool (NOT the
        # lossy span selector, which would discard most of the JSON). Headline numbers
        # come from the byte-exact codec; it is opt-in (declines on low-redundancy input).
        raw_text = json.dumps(raw, indent=2)
        raw_tokens = count_tokens(raw_text)
        try:
            structural = structural_report(raw)
        except Exception:
            structural = None
        beneficial = bool(structural and structural.get("beneficial"))
        if beneficial:
            kept_text = render_compact(raw)
            kept_tokens = count_tokens(kept_text)
            status = None
        else:
            kept_text, kept_tokens = raw_text, raw_tokens
            status = ("Low-redundancy JSON — the lossless codec declines (its reference table "
                      "would add tokens). Sent unchanged; compression is opt-in, decode is lossless either way.")
        return {
            "task": task, "is_json": True, "mode": "lossless", "lossless": True, "status": status,
            "raw_tokens": raw_tokens, "kept_tokens": kept_tokens,
            "ratio": (kept_tokens / raw_tokens) if raw_tokens else 1.0,
            "reduction_pct": (1 - kept_tokens / raw_tokens) * 100 if raw_tokens else 0.0,
            "compression_x": (raw_tokens / kept_tokens) if kept_tokens else None,
            "n_units": 0, "n_kept": 0, "structural": structural,
            "covered_needs": [], "compressed_text": kept_text, "units": [],
        }

    # ---- prose / text: lossy extractive selector ----
    structural = None
    units = unitize_text(str(raw))
    raw_text = str(raw)

    for i, u in enumerate(units):
        u["uid"] = f"u{i}"

    n_paras = max((u["para"] for u in units), default=0) + 1
    example = {"question": task, "n_paras": n_paras, "units": units}

    if units:
        scores = get_scorer().score_example(example)
        if semantic:  # optional: blend in embedding-cosine to close the lexical-overlap gap (no-op if no backend)
            from .semantic import blend
            scores = blend(scores, task, [u["text"] for u in units], weight=0.65)
    else:
        scores = []
    for u, s in zip(units, scores):
        u["score"] = float(s)

    selected = select_coverage(example, scores, budget) if units else []

    # coverage-confidence: how many of the question's key anchors survived in the kept spans.
    def _cov(sel):
        needs = covered_needs(example, sel)
        return (sum(n["covered"] for n in needs) / len(needs)) if needs else 1.0

    coverage = _cov(selected)
    effective_budget, auto_expanded = budget, False
    # safe mode: if the answer-bearing anchors aren't covered, widen the budget until they are
    # (or we've effectively included the whole input) — never silently drop the answer.
    raw_tok_cap = sum(u["tokens"] for u in units)
    if safe and units and coverage < min_coverage:
        b = budget
        while coverage < min_coverage and b < raw_tok_cap:
            b = int(b * 1.5)
            cand = select_coverage(example, scores, b)
            c = _cov(cand)
            if len(cand) <= len(selected) and c <= coverage:
                break  # no further improvement to be had
            selected, coverage, effective_budget, auto_expanded = cand, c, b, True

    sel = set(selected)
    by_uid = {u["uid"]: u for u in units}
    kept_in_order = [by_uid[uid] for uid in selected]

    raw_tokens = count_tokens(raw_text)
    kept_text = _render(kept_in_order)
    kept_tokens = count_tokens(kept_text)

    status = None
    if len(units) == 0:
        status = "No compressible content found — paste a longer document, chat, or JSON tool output."
    elif raw_tokens < budget * 1.2:
        status = (f"Input is short ({raw_tokens} tokens) — already near/under the {budget}-token "
                  f"budget, so there's little to compress. Paste a longer input or lower the budget.")

    return {
        "task": task,
        "is_json": False,
        "mode": "lossy",
        "lossless": False,
        "status": status,
        "raw_tokens": raw_tokens,
        "kept_tokens": kept_tokens,
        "ratio": (kept_tokens / raw_tokens) if raw_tokens else 1.0,
        "reduction_pct": (1 - kept_tokens / raw_tokens) * 100 if raw_tokens else 0.0,
        "compression_x": (raw_tokens / kept_tokens) if kept_tokens else None,
        "n_units": len(units),
        "n_kept": len(selected),
        "structural": structural,
        # coverage-confidence: the share of the question's key terms present in the kept spans.
        # Turns a silent drop into a signal; with safe=True the budget auto-widens to cover them.
        "coverage": round(coverage, 2),
        "confident": bool(coverage >= min_coverage),
        "fallback_recommended": bool(units and coverage < min_coverage),
        "auto_expanded": auto_expanded,
        "effective_budget": effective_budget,
        "covered_needs": covered_needs(example, selected) if units else [],
        "compressed_text": kept_text,
        "units": [
            {
                "id": u["uid"], "title": u.get("title", ""), "text": u["text"],
                "tokens": u["tokens"], "score": round(u.get("score", 0.0), 3),
                "kept": u["uid"] in sel,
                "source": u.get("path", f"char {u['char_start']}-{u['char_end']}"),
            }
            for u in units
        ],
    }
