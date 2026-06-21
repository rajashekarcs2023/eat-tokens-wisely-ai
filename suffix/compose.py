"""Render a selection of units back into clean, provenance-ordered context for a
reader LLM. Output is a verbatim subsequence of the source (kept spans only),
grouped under their paragraph titles in original order.
"""

from __future__ import annotations


def selected_units(example, uids):
    by_uid = {u["uid"]: u for u in example["units"]}
    units = [by_uid[u] for u in uids if u in by_uid]
    units.sort(key=lambda u: (u["para"], u["sent"]))
    return units


def render_context(example, uids) -> str:
    units = selected_units(example, uids)
    out, cur_title = [], None
    for u in units:
        if u["title"] != cur_title:
            cur_title = u["title"]
            out.append(f"== {cur_title} ==")
        out.append(u["text"].strip())
    return "\n".join(out)
