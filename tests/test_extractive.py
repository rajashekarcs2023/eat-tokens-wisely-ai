"""The core honesty guarantee as an executable test: SUFFIX is extractive, so
every kept span is a verbatim substring of the source input, and the rendered
compressed context introduces no token that wasn't in the source. This is what
makes "zero unsupported claims" a string-containment property, not an opinion.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix.pipeline import compress

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_kept_spans_are_verbatim_text():
    text = (
        "The deployment failed at 02:14 UTC. Root cause: the verify-email route "
        "calls router.push('/dashboard') and drops the redirect_to parameter. "
        "Unrelated: the cafeteria menu changed on Tuesday. "
        "Do not allow open redirects; whitelist internal paths only."
    )
    r = compress("Why is the redirect_to parameter lost?", text, budget=60)
    kept = [u for u in r["units"] if u["kept"]]
    assert kept, "expected some kept spans"
    for u in kept:
        assert u["text"] in text, f"kept span not verbatim in source: {u['text']!r}"
    # the rendered output (no headers for plain text) is built only from source spans
    for line in r["compressed_text"].split("\n"):
        if line.strip():
            assert line.strip() in text, f"rendered line not verbatim: {line!r}"


def test_json_compress_is_lossless_not_lossy():
    """JSON input routes to the LOSSLESS structural codec (not the lossy selector):
    the compressed form decodes byte-exact, so 'kept content is verbatim' is upgraded
    to 'all content is preserved exactly'."""
    from suffix.structural import decode, encode

    bundle = json.load(open(os.path.join(ROOT, "data", "hero_bundle.json")))
    obj = bundle["context"]
    r = compress(bundle["task"], obj, budget=200)
    assert r["mode"] == "lossless" and r["lossless"] is True, "JSON must use the lossless codec"
    assert r["structural"]["lossless_verified"] is True
    assert decode(encode(obj)) == obj, "JSON round-trip must be byte-exact"
    # when the codec engages it strictly reduces tokens (and is opt-in otherwise)
    if r["structural"]["beneficial"]:
        assert r["kept_tokens"] < r["raw_tokens"]
    print(f"JSON lossless: {r['raw_tokens']} -> {r['kept_tokens']} tok, byte-exact round-trip")


if __name__ == "__main__":
    test_kept_spans_are_verbatim_text()
    test_json_compress_is_lossless_not_lossy()
    print("OK extractive")
