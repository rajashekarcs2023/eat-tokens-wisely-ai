"""Objective metrics — string comparisons against gold, no LLM judge anywhere.

  - supporting_fact_prf: did we keep the *right* spans? (vs human gold uids)
  - squad_em / squad_f1: official HotpotQA answer scoring for the downstream arm.
  - tokens_of: token budget actually used by a selection.
"""

from __future__ import annotations

from collections import Counter

from .text_utils import normalize_answer


def supporting_fact_prf(selected_uids, gold_uids):
    sel = set(selected_uids)
    gold = set(gold_uids)
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(sel & gold)
    precision = tp / len(sel) if sel else 0.0
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def tokens_of(units_by_uid, selected_uids):
    return sum(units_by_uid[u]["tokens"] for u in selected_uids if u in units_by_uid)


def squad_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def squad_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_toks)
    recall = same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)
