"""Demonstrate the semantic upgrade closing the lexical-overlap gap.

The answer sentence shares almost NO words with the question (the question says
"minutes / respond / alert / backup"; the answer says "480-second grace period /
rolls over to the secondary"). The distractor sentences DO share those words but
don't answer it. So the lexical scorer is fooled into keeping distractors; the
embedding-blended scorer keeps the real answer.

Run: python scripts/semantic_demo.py   (needs an embedding backend: pip install fastembed)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suffix.pipeline import compress  # noqa: E402
from suffix.semantic import available, backend  # noqa: E402

# The answer shares ZERO content words with the question and has NO number/entity to grab
# (Q: who/responsible/approving/production/deployments ; answer: sign-off/shipping/live/
# release captain). The distractors DO share the question's words but don't answer it —
# so the lexical scorer is fooled; only semantic similarity catches the real answer.
DOC = """The weather forecast predicts heavy rain across the coast through the weekend.
A good risotto needs constant stirring and a generous splash of white wine.
The marathon route winds through five historic neighborhoods near downtown.
Sign-off on shipping to the live environment rests with the release captain.
Tickets for the summer jazz festival sold out within the very first hour.
Hummingbirds can beat their wings around fifty times every single second."""

Q = "Who gives the final go-ahead before new code is deployed?"
ANSWER_MARKER = "release captain"


def run(semantic):
    r = compress(Q, DOC, budget=35, semantic=semantic)
    ranked = sorted(r["units"], key=lambda u: -u["score"])
    ans = next((u for u in r["units"] if ANSWER_MARKER in u["text"]), None)
    rank = [u["id"] for u in ranked].index(ans["id"]) + 1
    print(f"  semantic={str(semantic):5s} | answer ranked #{rank}/{len(ranked)} (score {ans['score']:.3f}) "
          f"| kept={ans['kept']} -> kept: {[u['text'][:20] for u in ranked if u['kept']]}")
    return ans["kept"]


def main():
    print(f"embedding backend: {backend()}  (available={available()})\n")
    print(f"Q: {Q}\nanswer sentence: 'Sign-off on shipping to the live environment rests with the release captain.'\n")
    lex = run(False)
    sem = run(True)
    print()
    if not available():
        print("  -> no embedding backend installed; semantic=True is a no-op (identical to lexical).")
        print("     install one to see the gap close:  pip install fastembed")
    elif sem and not lex:
        print("  -> GAP CLOSED: lexical scoring DROPPED the answer; the semantic blend KEEPS it.")
    elif lex and sem:
        print("  -> both kept it here (lexical overlap was enough this time).")
    else:
        print("  -> see scores above.")


if __name__ == "__main__":
    main()
