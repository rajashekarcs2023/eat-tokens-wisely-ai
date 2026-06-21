"""Build data/usecases.json — concrete, realistic scenarios for the demo's
"Use Cases" tab. Each runs live through our codec (normal vs compressed), with a
known gold answer so the side-by-side shows: fewer tokens + same answer.

Two modes:
  - lossy:    long prose context -> extractive span-selection (5x fewer tokens)
  - lossless: structured JSON (tool output / logs) -> structural codec the LLM
              reads natively (fewer tokens, ZERO information loss)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix.tokens import count_tokens  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _units_from_lines(lines, titler=lambda i: ""):
    units = []
    for i, t in enumerate(lines):
        if not t.strip():
            continue
        units.append({"uid": f"u{i}", "para": i, "sent": 0, "n_sent_in_para": 1,
                      "title": titler(i), "text": t, "tokens": count_tokens(t), "is_gold": False})
    return units


def rag_scenario():
    """Pick a long HotpotQA example with a clean answer — real retrieved-passage RAG."""
    test = json.load(open(os.path.join(DATA, "hotpot_test.json")))
    cand = [e for e in test if 1300 < e["raw_tokens"] < 2200 and 1 <= len(e["answer"]) <= 30
            and e["answer"].lower() not in ("yes", "no")]
    ex = cand[3]
    return {"id": "rag", "mode": "lossy", "budget": 200,
            "title": "RAG · multi-hop QA over retrieved passages",
            "blurb": "An agent retrieves 10 passages to answer one question. Most are distractors. "
                     "The extractive selector keeps only the answer-bearing spans.",
            "example": {"question": ex["question"], "answer": ex["answer"], "n_paras": ex["n_paras"],
                        "units": ex["units"]}}


def chat_scenario():
    """A long customer-support conversation; the answer is buried mid-thread."""
    turns = [
        ("agent", "Hi! Thanks for contacting Acme support. How can I help today?"),
        ("user", "My deployment keeps failing on the staging environment."),
        ("agent", "Sorry to hear that. Can you share the error you see?"),
        ("user", "It says 'database migration timeout after 30s'."),
        ("agent", "Got it. Which database version are you on?"),
        ("user", "We're on Postgres 14.2, upgraded last week from 13."),
        ("agent", "Thanks. Is this on the shared cluster or a dedicated instance?"),
        ("user", "Dedicated, the db-prod-7 instance in us-east."),
        ("agent", "And how large is the migration — number of rows affected?"),
        ("user", "It's altering the orders table, about 48 million rows."),
        ("agent", "That's the likely cause. Large ALTERs on 48M rows exceed the 30s statement timeout."),
        ("user", "Ah. Can we raise the timeout?"),
        ("agent", "Yes — set statement_timeout to 0 for the migration session, or run it as a batched online migration."),
        ("user", "Let's do the batched approach. Any tool you recommend?"),
        ("agent", "pg-osc or gh-ost-style online schema change works well for tables this size."),
        ("user", "Great. Also, will this affect the read replicas?"),
        ("agent", "Replicas will lag during the migration but won't drop connections."),
        ("user", "Perfect. One more — what was the row count you mentioned again?"),
        ("agent", "48 million rows on the orders table."),
        ("user", "Thanks, that's all!"),
    ]
    lines = [f"{s}: {t}" for s, t in turns]
    units = _units_from_lines(lines, titler=lambda i: turns[i][0])
    return {"id": "chat", "mode": "lossy", "budget": 120,
            "title": "Conversation memory · compress a long support thread",
            "blurb": "A 20-turn support chat. To answer a follow-up, the agent only needs a few turns. "
                     "The extractive selector keeps the load-bearing turns.",
            "example": {"question": "How many rows are on the orders table being migrated?",
                        "answer": "48 million", "n_paras": len(units), "units": units}}


def tool_scenario():
    """Realistic MCP github.list_issues output — lossless structural compression."""
    repo = "https://github.com/acme/web-app"
    user = {"login": "octo-dev", "id": 1001, "url": "https://github.com/octo-dev", "type": "User"}
    qa_user = {"login": "qa-lee", "id": 1002, "url": "https://github.com/qa-lee", "type": "User"}
    lbug = {"name": "bug", "color": "d73a4a", "url": repo + "/labels/bug"}
    lauth = {"name": "auth", "color": "0e8a16", "url": repo + "/labels/auth"}
    bodies = ["Deep link dropped after email verification.", "Login redirect fails on Safari.",
              "Token refresh loops on expiry.", "Avatar upload returns 500."]
    issues = []
    for k in range(1, 19):
        issues.append({"number": k, "repository_url": repo, "user": user,
                       "labels": [lbug, lauth] if k % 2 else [lbug],
                       "state": "open" if k % 3 else "closed",
                       "assignee": qa_user if k % 2 else user, "milestone": None,
                       "html_url": f"{repo}/issues/{k}", "body": bodies[k % len(bodies)]})
    obj = {"tool": "github.list_issues", "repository_url": repo, "total": len(issues), "issues": issues}
    return {"id": "tool", "mode": "lossless",
            "title": "Agent tool-output · lossless codec",
            "blurb": "An MCP tool returns 18 issues with the same repo/user/label objects repeated. "
                     "The lossless codec factors the repetition into shared definitions — the LLM reads it "
                     "natively, ZERO information lost.",
            "json": obj,
            # clean-parity question: both full and compact carry the same repo, so both answer
            # correctly (a lossless card must never look like compression changed the answer).
            "question": "What repository_url do these issues belong to? Give the URL.",
            "gold": repo}


def log_scenario():
    """Server log triage — lossless structural compression of repeated lines."""
    lines = (["INFO request handled status=200 latency=12ms"] * 30 +
             ["WARN cache miss key=user:session"] * 12 +
             ["ERROR upstream timeout on payments-svc"] * 7 +
             ["INFO request handled status=200 latency=9ms"] * 25 +
             ["WARN rate limit near threshold"] * 4)
    obj = {"service": "api-gateway", "window": "10m", "logs": lines}
    return {"id": "log", "mode": "lossless",
            "title": "Log triage · compress repetitive logs",
            "blurb": "A 78-line gateway log, mostly repeated lines. The codec dedups them losslessly so the "
                     "model can still count and reason over every line at a fraction of the tokens.",
            "json": obj,
            "question": "How many ERROR lines are in this log? Answer with just the number.",
            "gold": str(sum(1 for x in lines if x.startswith("ERROR")))}


def main():
    # order: lossy prose first, then the lossless codec led by LOG (cleanest — both arms
    # agree on a countable gold), tool last (MCP-flavored, same byte-exact codec).
    scenarios = [rag_scenario(), chat_scenario(), log_scenario(), tool_scenario()]
    json.dump(scenarios, open(os.path.join(DATA, "usecases.json"), "w"))
    for s in scenarios:
        if s["mode"] == "lossy":
            rt = sum(u["tokens"] for u in s["example"]["units"])
            print(f"  [{s['id']:5s}] lossy   · {len(s['example']['units'])} units, ~{rt} tok · Q: {s['example']['question'][:50]}")
        else:
            print(f"  [{s['id']:5s}] lossless· Q: {s['question'][:50]} gold={s['gold']}")
    print("saved data/usecases.json")


if __name__ == "__main__":
    main()
