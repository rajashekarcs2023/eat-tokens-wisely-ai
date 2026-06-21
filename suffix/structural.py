"""Stage 1: lossless structural codec for structured (JSON) tool outputs.

Real MCP/tool responses repeat both long string VALUES (repo URLs, bodies) AND
whole SUBTREES (the same `user` object, `labels` array, nested records). We factor
both into a symbol table:
  - repeated subtrees (dict/list appearing >= 2x, large enough)  -> {SUB: id}
  - repeated string values                                       -> short ref
This is EXACTLY invertible: decode(encode(obj)) == obj, byte-for-byte, asserted by
a round-trip test (tests/test_structural.py). On plain prose nothing factors, so
this stage yields ~0% and we report it ONLY on structured payloads.
"""

from __future__ import annotations

import json
from collections import Counter

from .tokens import count_tokens

_MIN_LEN = 8        # only factor string values this long
_MIN_COUNT = 2
_MIN_SUB_LEN = 30   # only factor subtrees whose serialization is this long
_SUB_SENTINELS = ("§SUB§", "@@SUB@@", "<<SUBREF>>", "~~SUBREF~~")
_STR_SENTINELS = ("@R", "REF#", "<<R", "~REF~")


def _all_keys(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            acc.add(k); _all_keys(v, acc)
    elif isinstance(o, list):
        for v in o:
            _all_keys(v, acc)


def _choose_subkey(obj):
    keys = set(); _all_keys(obj, keys)
    for s in _SUB_SENTINELS:
        if s not in keys:
            return s
    i = 0  # collision-safe fallback: extend until unused (guarantees losslessness)
    while f"~~SUBREF{i}~~" in keys:
        i += 1
    return f"~~SUBREF{i}~~"


def _canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _choose_str_sentinel(all_strings):
    for p in _STR_SENTINELS:
        if not any(p in s for s in all_strings):
            return p
    i = 0  # collision-safe fallback: extend until no string contains it
    while any(f"~~SREF{i}~~" in s for s in all_strings):
        i += 1
    return f"~~SREF{i}~~"


def encode(obj):
    subkey = _choose_subkey(obj)  # collision-safe: never a key already present in obj
    # ---------- 1) repeated-subtree dedup ----------
    counts: Counter = Counter()
    reps: dict = {}

    def walk(o):
        if isinstance(o, (dict, list)):
            c = _canon(o)
            counts[c] += 1
            reps.setdefault(c, o)
            for v in (o.values() if isinstance(o, dict) else o):
                walk(v)

    walk(obj)
    top = _canon(obj)
    cand = [c for c, n in counts.items() if n >= _MIN_COUNT and len(c) >= _MIN_SUB_LEN and c != top]
    cand.sort(key=len, reverse=True)
    sub_ref = {c: f"T{i}" for i, c in enumerate(cand)}

    def sub_tree(o, skip):
        if isinstance(o, (dict, list)):
            c = _canon(o)
            if c in sub_ref and c != skip:
                return {subkey: sub_ref[c]}
            if isinstance(o, dict):
                return {k: sub_tree(v, None) for k, v in o.items()}
            return [sub_tree(v, None) for v in o]
        return o

    sub_table = {rid: sub_tree(reps[c], c) for c, rid in sub_ref.items()}
    data = sub_tree(obj, top)

    # ---------- 2) repeated-string-value dedup on the result ----------
    def iter_strings(o):
        if isinstance(o, str):
            yield o
        elif isinstance(o, dict):
            for v in o.values():
                yield from iter_strings(v)
        elif isinstance(o, list):
            for v in o:
                yield from iter_strings(v)

    all_strings = list(iter_strings(data)) + [s for v in sub_table.values() for s in iter_strings(v)]
    sent = _choose_str_sentinel(all_strings)
    scount = Counter(all_strings)
    str_rev = {}
    str_table = {}
    for s, c in scount.items():
        if c >= _MIN_COUNT and len(s) >= _MIN_LEN:
            sym = sent + str(len(str_table))
            str_table[sym] = s
            str_rev[s] = sym

    def sub_str(o):
        if isinstance(o, str):
            return str_rev.get(o, o)
        if isinstance(o, dict):
            return {k: sub_str(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sub_str(v) for v in o]
        return o

    return {"_suffix_codec": "v2", "sentinel": sent, "subkey": subkey,
            "table": str_table,
            "sub_table": {k: sub_str(v) for k, v in sub_table.items()},
            "data": sub_str(data)}


def decode(enc):
    str_table = enc["table"]
    sub_table = enc["sub_table"]
    subkey = enc.get("subkey", _SUB_SENTINELS[0])

    def un(o):
        if isinstance(o, dict):
            if len(o) == 1 and subkey in o:
                return un(sub_table[o[subkey]])
            return {k: un(v) for k, v in o.items()}
        if isinstance(o, list):
            return [un(v) for v in o]
        if isinstance(o, str):
            return str_table.get(o, o)
        return o

    return un(enc["data"])


def render_compact(obj) -> str:
    """LLM-readable compact form: shared definitions + data with references.
    A capable reader resolves the refs natively — letting us send the *deduplicated*
    bytes (fewer tokens) with ZERO information loss."""
    enc = encode(obj)
    out = ["This JSON is given in a compact form with shared definitions. "
           "When reading, substitute each reference by its definition below."]
    if enc["sub_table"]:
        out.append('OBJECT DEFINITIONS  (a value {"' + enc["subkey"] + '":"T0"} means: use T0):')
        for rid, sub in enc["sub_table"].items():
            out.append(f"  {rid} = {json.dumps(sub, ensure_ascii=False)}")
    if enc["table"]:
        out.append("STRING DEFINITIONS  (a token like the keys below means: use that string):")
        for sym, s in enc["table"].items():
            out.append(f"  {sym} = {json.dumps(s, ensure_ascii=False)}")
    out.append("DATA:")
    out.append(json.dumps(enc["data"], ensure_ascii=False))
    return "\n".join(out)


def structural_report(obj) -> dict:
    enc = encode(obj)
    verified = decode(enc) == obj  # computed, not asserted-then-hardcoded
    assert verified, "lossless codec round-trip FAILED"
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    raw_t = count_tokens(raw)
    # measure the artifact actually SENT to the LLM (the compact form incl. its legend),
    # not the internal json.dumps(enc) — so `beneficial` reflects real token savings.
    packed_t = count_tokens(render_compact(obj))
    return {
        "raw_tokens": raw_t,
        "packed_tokens": packed_t,
        "saved_tokens": raw_t - packed_t,
        "ratio": packed_t / raw_t if raw_t else 1.0,
        "n_refs": len(enc["sub_table"]) + len(enc["table"]),
        "n_subtree_refs": len(enc["sub_table"]),
        "n_string_refs": len(enc["table"]),
        "lossless_verified": verified,
        # savings scale with structural redundancy; on low-repetition payloads the
        # ref table can cost more than it saves. The codec is opt-in: use the compact
        # form only when beneficial (decode is lossless either way).
        "beneficial": packed_t < raw_t,
    }
