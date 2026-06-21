"""The lossless claim is a TEST, not a promise: decode(encode(obj)) == obj,
byte-for-byte, including on a realistic repetitive MCP-style payload."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix.structural import decode, encode, structural_report


def _mcp_bundle():
    repo = "https://github.com/acme/web-app"
    user = {"login": "octo-dev", "url": "https://github.com/octo-dev", "type": "User"}
    label = {"name": "bug", "color": "d73a4a", "url": repo + "/labels/bug"}
    return {
        "tool": "github.list_issues",
        "issues": [
            {"number": n, "repository_url": repo, "user": user, "labels": [label],
             "state": "open", "body": "Deep link is dropped after email verification."}
            for n in range(1, 13)
        ],
    }


def _nested_repeats():
    user = {"login": "octo", "url": "https://github.com/octo", "type": "User"}
    label = {"name": "bug", "color": "d73a4a"}
    inner = {"user": user, "labels": [label, label], "meta": {"k": "v" * 20}}
    return {"a": [inner, inner, inner], "b": {"x": inner, "y": user},
            "c": [user, user, {"nested": inner}], "d": "a repeated string value here",
            "e": "a repeated string value here"}


def test_roundtrip_exact():
    for obj in [
        _nested_repeats(),
        # adversarial: input literally contains the codec's sentinel tokens as data
        {"§SUB§": "real value not a ref", "@R0": "also real",
         "x": {"§SUB§": 1, "url": "https://example.com/repeated-value-here"},
         "y": {"§SUB§": 1, "url": "https://example.com/repeated-value-here"},
         "z": ["@R0", "@R0", "@R0"]},
        {"@@SUB@@": [1, 2], "data": [{"@@SUB@@": [1, 2]}, {"@@SUB@@": [1, 2]}]},
        # ALL sentinels blocked: every string + subtree sentinel appears in the data,
        # forcing the collision-safe fallbacks. Round-trip must still be byte-exact.
        {"§SUB§": 1, "@@SUB@@": 2, "<<SUBREF>>": 3, "~~SUBREF~~": 4,
         "strs": ["has @R here", "has REF# here", "has <<R here", "has ~REF~ here",
                  "has @R here", "has REF# here"],
         "t": {"k": "tokens @R REF# <<R ~REF~ ~SREF0~ in one long string value",
               "k2": "tokens @R REF# <<R ~REF~ ~SREF0~ in one long string value"}},
        {"a": "short", "b": "short", "c": ["x", "x", "x"]},
        {"nested": {"deep": {"v": "a repeated long string value here"}},
         "again": "a repeated long string value here"},
        _mcp_bundle(),
        {"unicode": "café — déjà", "again": "café — déjà"},
        [],
        {"empty": ""},
    ]:
        enc = encode(obj)
        assert decode(enc) == obj, f"round-trip failed for {obj!r}"


def test_fuzz_roundtrip_4000():
    """4,000 randomized nested structures — with the codec's own sentinel tokens injected
    as both keys and values — must all round-trip byte-exact (deterministic by seed)."""
    import random

    sentinels = ["@R", "REF#", "<<R", "~REF~", "§SUB§", "@@SUB@@", "<<SUBREF>>",
                 "~~SUBREF~~", "~SREF0~", "~~SUBREF0~~"]

    def gen(depth, r):
        if depth > 4:
            return r.choice([1, "x", r.choice(sentinels)])
        t = r.random()
        if t < 0.3:
            return {r.choice(["a", "b"] + sentinels): gen(depth + 1, r) for _ in range(r.randint(0, 3))}
        if t < 0.5:
            return [gen(depth + 1, r) for _ in range(r.randint(0, 4))]
        if t < 0.7:
            return r.choice(sentinels) + r.choice(["", " val", "0"])
        return r.choice([1, 2.5, True, None, "plain"])

    for i in range(4000):
        obj = gen(0, random.Random(i))
        assert decode(encode(obj)) == obj, f"fuzz round-trip failed at seed {i}: {obj!r}"


def test_reports_savings_on_repetitive_json():
    rep = structural_report(_mcp_bundle())
    assert rep["lossless_verified"] is True
    assert rep["saved_tokens"] > 0, "should compress a repetitive MCP bundle"
    print("structural:", rep)


if __name__ == "__main__":
    test_roundtrip_exact()
    test_fuzz_roundtrip_4000()
    test_reports_savings_on_repetitive_json()
    print("OK structural")
