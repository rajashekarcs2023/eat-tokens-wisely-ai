"""FastAPI backend for the SUFFIX / MCPZip demo.

  GET  /                 -> the single-page UI
  GET  /api/results      -> precomputed pareto.json + redundancy.json (offline)
  GET  /api/hero         -> the hero MCP bundle + its live compression
  POST /api/compress     -> live, sub-second, CPU-only compression of pasted input

The live endpoints invoke NO LLM — only scikit-learn + string ops — so the demo
cannot fail on venue wifi.
"""

from __future__ import annotations

import json
import os
import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
WEB = os.path.join(ROOT, "web")

from suffix.pipeline import compress  # noqa: E402

app = FastAPI(title="SUFFIX / MCPZip")


@app.exception_handler(Exception)
async def _json_errors(request, exc):
    # never surface a raw 500 HTML page to the demo — always clean JSON the UI can show
    return JSONResponse({"error": f"{type(exc).__name__}: {str(exc)[:180]}"}, status_code=500)


def _load(name):
    path = os.path.join(DATA, name)
    try:
        return json.load(open(path)) if os.path.exists(path) else None
    except Exception:
        return None  # corrupt/partial artifact must not 500 the demo


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/architecture")
def architecture():
    return FileResponse(os.path.join(WEB, "architecture.html"))


@app.get("/api/results")
def results():
    pareto_v2 = _load("pareto_v2.json")
    econ = None
    if pareto_v2 and "conditions" in pareto_v2:
        c = pareto_v2["conditions"]
        full = c.get("full", {}).get("avg_tokens")
        # the shipped operating point: reader-grounded scorer at budget 240
        comp = (c.get("C@240") or c.get("A@240") or {}).get("avg_tokens")
        if full and comp:
            from suffix.economics import savings
            econ = savings(full, comp)
    return JSONResponse({
        "pareto": _load("pareto.json"),
        "pareto_v2": pareto_v2,
        "redundancy": _load("redundancy.json"),
        "economics": econ,
        "adaptive_gate": _load("adaptive_gate.json"),
        "adaptive_result": _load("adaptive_result.json"),
        "crosstask": {"squad": _load("crosstask_squad.json"), "2wiki": _load("crosstask_2wiki.json"),
                      "coqa": _load("crosstask_coqa.json"), "narrativeqa": _load("crosstask_narrativeqa.json")},
        "structural_qa": _load("structural_qa.json"),
    })


@app.get("/api/hero")
def hero():
    bundle = _load("hero_bundle.json")
    if not bundle:
        return JSONResponse({"error": "no hero bundle"}, status_code=404)
    result = compress(bundle["task"], bundle["context"], budget=200)
    return JSONResponse({"bundle": bundle, "result": result})


@app.get("/api/prove")
def prove(i: int = 0, budget: int = 240):
    """LIVE proof: compress a held-out HotpotQA example and show the frozen reader's
    answer on FULL vs COMPRESSED context, scored against the gold answer. Real API
    calls — you watch compressed context produce the right answer at ~5x fewer tokens."""
    import os as _os
    from suffix.compose import render_context
    from suffix.coverage import select_coverage
    from suffix.llm import READER_MODEL, have_key, read_answer
    from suffix.metrics import squad_em, squad_f1, tokens_of
    from suffix.scorer import KeepScorer

    if not have_key():
        return JSONResponse({"error": "no API key"}, status_code=503)
    test = json.load(open(_os.path.join(DATA, "hotpot_test.json")))
    # only demo LONG-context examples so compression is always dramatic & visible
    long_idx = [j for j, e in enumerate(test) if e.get("raw_tokens", 0) > 900] or list(range(len(test)))
    true_idx = long_idx[i % len(long_idx)]
    ex = test[true_idx]
    scorer = KeepScorer.load(_os.path.join(DATA, "keep_scorer.joblib"))
    scores = scorer.score_example(ex)
    by = {u["uid"]: u for u in ex["units"]}

    full_uids = [u["uid"] for u in ex["units"]]
    comp_uids = select_coverage(ex, scores, budget)
    full_ctx, comp_ctx = render_context(ex, full_uids), render_context(ex, comp_uids)
    full_tok, comp_tok = tokens_of(by, full_uids), tokens_of(by, comp_uids)
    full_ans, comp_ans = read_answer(full_ctx, ex["question"]), read_answer(comp_ctx, ex["question"])
    return JSONResponse({
        "reader": READER_MODEL, "index": true_idx,
        "question": ex["question"], "gold": ex["answer"], "type": ex.get("type", ""),
        "full": {"answer": full_ans, "tokens": full_tok,
                 "em": squad_em(full_ans, ex["answer"]), "f1": round(squad_f1(full_ans, ex["answer"]), 3)},
        "compressed": {"answer": comp_ans, "tokens": comp_tok, "budget": budget,
                       "em": squad_em(comp_ans, ex["answer"]), "f1": round(squad_f1(comp_ans, ex["answer"]), 3),
                       "compression_x": round(full_tok / comp_tok, 1) if comp_tok else None,
                       "kept_spans": [{"text": by[u]["text"], "title": by[u]["title"]} for u in comp_uids]},
    })


@app.get("/api/usecases")
def usecases():
    scen = _load("usecases.json") or []
    items = [{"id": s["id"], "title": s["title"], "blurb": s["blurb"], "mode": s["mode"]} for s in scen]
    mc = _load("usecase_mcp.json")
    if mc and mc.get("scenarios"):
        for s in mc["scenarios"]:
            items.append({"id": s["id"], "title": s["title"], "blurb": s["blurb"], "mode": "mcp"})
    v = _load("usecase_voice.json")
    if v:
        items.append({"id": "voice", "title": v["title"], "blurb": v["blurb"], "mode": "voice"})
    return JSONResponse(items)


@app.get("/api/tts")
def tts(text: str):
    """Deepgram TTS — used by the voice demo to play the answer aloud."""
    import os as _os
    import requests
    from fastapi.responses import Response
    from suffix.llm import load_env
    load_env()
    key = _os.environ.get("DEEPGRAM_API_KEY", "")
    if not key:
        return JSONResponse({"error": "no deepgram key"}, status_code=503)
    try:
        r = requests.post(
            "https://api.deepgram.com/v1/speak?model=aura-2-thalia-en",
            headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
            json={"text": text[:600]}, timeout=30)
        if r.status_code != 200:
            return JSONResponse({"error": r.text[:120]}, status_code=502)
        return Response(content=r.content, media_type=r.headers.get("content-type", "audio/mpeg"))
    except Exception as e:
        return JSONResponse({"error": str(e)[:120]}, status_code=502)


@app.get("/api/usecase")
def usecase(id: str, customq: str = None):
    import json as _json
    import os as _os
    from suffix.compose import render_context
    from suffix.coverage import select_coverage
    from suffix.economics import PRICES
    from suffix.llm import READER_MODEL, have_key, read_answer
    from suffix.metrics import squad_em, squad_f1, tokens_of
    from suffix.scorer import KeepScorer
    from suffix.structural import render_compact, structural_report
    from suffix.tokens import count_tokens

    import time as _time
    from suffix.pipeline import unitize_text

    if not have_key():
        return JSONResponse({"error": "no API key"}, status_code=503)

    opus = PRICES["claude-opus-4-8"]["input"]  # $/1M input tokens — show savings at frontier-model price
    def cost(tok):
        return round(tok * opus, 2)

    deepgram = None
    mcp = None
    if id == "voice":  # real Deepgram ASR transcript -> compress -> timed LLM
        v = _load("usecase_voice.json")
        if not v:
            return JSONResponse({"error": "voice transcript not built"}, status_code=404)
        q, gold = v["question"], v["gold"]
        mode, title, blurb, deepgram = "voice", v["title"], v["blurb"], v["deepgram"]
        units = unitize_text(v["transcript"])
        for i, u in enumerate(units):
            u["uid"] = f"u{i}"
        ex = {"question": q, "n_paras": max((u["para"] for u in units), default=0) + 1, "units": units}
        scorer = KeepScorer.load(_os.path.join(DATA, "keep_scorer.joblib"))
        scores = scorer.score_example(ex)
        by = {u["uid"]: u for u in units}
        full_uids = [u["uid"] for u in units]
        comp_uids = select_coverage(ex, scores, 150)
        normal_ctx, mcp_ctx = render_context(ex, full_uids), render_context(ex, comp_uids)
        n_tok, m_tok = tokens_of(by, full_uids), tokens_of(by, comp_uids)
        lossless = False
    elif (_load("usecase_mcp.json") or {}).get("scenarios") and \
            id in {s["id"] for s in _load("usecase_mcp.json")["scenarios"]}:
        # REAL MCP tool output (cached from live MCP servers), compressed mode-appropriately
        from suffix.pipeline import compress as _compress
        b = _load("usecase_mcp.json")
        scen = next(s for s in b["scenarios"] if s["id"] == id)
        q, gold = scen["question"], scen["gold"]
        mode, title, blurb = "mcp", scen["title"], scen["blurb"]
        mcp = {"servers": b["servers"], "trajectory": b["trajectory"],
               "server": scen["server"], "tool": scen["tool"], "scenario_mode": scen["mode"]}
        if scen["mode"] == "lossless":
            obj = scen["result_obj"]
            normal_ctx, mcp_ctx = _json.dumps(obj, separators=(",", ":")), render_compact(obj)  # minified baseline → codec-only reduction (no whitespace credit)
            lossless = structural_report(obj)["lossless_verified"]
        else:  # extractive: query-conditioned, verbatim spans
            normal_ctx = scen["context"]
            mcp_ctx = _compress(q, normal_ctx, scen.get("budget", 400))["compressed_text"]
            lossless = False
        n_tok, m_tok = count_tokens(normal_ctx), count_tokens(mcp_ctx)
    else:
        scen = {s["id"]: s for s in (_load("usecases.json") or [])}.get(id)
        if not scen:
            return JSONResponse({"error": "unknown scenario"}, status_code=404)
        mode, title, blurb = scen["mode"], scen["title"], scen["blurb"]
        if scen["mode"] == "lossy":
            ex = scen["example"]; q = ex["question"]; gold = ex["answer"]
            scorer = KeepScorer.load(_os.path.join(DATA, "keep_scorer.joblib"))
            scores = scorer.score_example(ex)
            by = {u["uid"]: u for u in ex["units"]}
            full_uids = [u["uid"] for u in ex["units"]]
            comp_uids = select_coverage(ex, scores, scen["budget"])
            normal_ctx, mcp_ctx = render_context(ex, full_uids), render_context(ex, comp_uids)
            n_tok, m_tok = tokens_of(by, full_uids), tokens_of(by, comp_uids)
            lossless = False
        else:
            q, gold = scen["question"], scen["gold"]
            obj = scen["json"]
            normal_ctx, mcp_ctx = _json.dumps(obj, separators=(",", ":")), render_compact(obj)  # minified baseline → codec-only reduction (no whitespace credit)
            n_tok, m_tok = count_tokens(normal_ctx), count_tokens(mcp_ctx)
            lossless = structural_report(obj)["lossless_verified"]

    # JUDGE SELF-TRY: override the question. Re-compress for the new question (extractive);
    # for lossless the compact form is question-independent so only the answer changes.
    # No known gold for a custom question -> we score the compressed answer against the
    # FULL-context answer (agreement = compression preserved it).
    custom = bool(customq and customq.strip())
    if custom:
        q, gold = customq.strip()[:300], None
        if not lossless:
            from suffix.pipeline import compress as _cc
            mcp_ctx = _cc(q, normal_ctx, 300)["compressed_text"]
            m_tok = count_tokens(mcp_ctx)

    def correct(ans):  # lenient: exact, or gold contained, or high token-F1
        return bool(squad_em(ans, gold) >= 1 or gold.lower().strip() in ans.lower()
                    or squad_f1(ans, gold) >= 0.6)

    def timed(ctx):
        t = _time.perf_counter()
        a = read_answer(ctx, q)
        return a, round((_time.perf_counter() - t) * 1000)

    n_ans, n_ms = timed(normal_ctx)
    if gold is None:  # custom question: the full-context answer is the reference
        gold = n_ans
    m_ans, m_ms = timed(mcp_ctx)
    return JSONResponse({
        "id": id, "mode": mode, "title": title, "blurb": blurb, "custom": custom,
        "question": q, "gold": gold, "lossless": lossless, "reader": READER_MODEL, "deepgram": deepgram, "mcp": mcp,
        "normal": {"tokens": n_tok, "answer": n_ans, "correct": correct(n_ans),
                   "cost_per_1M": cost(n_tok), "latency_ms": n_ms, "preview": normal_ctx[:600]},
        "mcpzip": {"tokens": m_tok, "answer": m_ans, "correct": correct(m_ans),
                   "cost_per_1M": cost(m_tok), "latency_ms": m_ms,
                   "reduction_pct": round(100 * (1 - m_tok / n_tok), 0) if n_tok else 0,
                   "compression_x": round(n_tok / m_tok, 1) if m_tok else None,
                   "preview": mcp_ctx[:600]},
    })


class CompressReq(BaseModel):
    task: str
    text: str | None = None
    json_input: dict | None = None
    budget: int = 200
    semantic: bool = False
    safe: bool = False


@app.post("/api/compress")
def do_compress(req: CompressReq):
    raw = req.json_input if req.json_input is not None else (req.text or "")
    result = compress(req.task, raw, budget=req.budget, semantic=req.semantic, safe=req.safe)
    from suffix.semantic import available
    result["semantic_used"] = bool(req.semantic and available())   # whether the embedding rerank actually ran
    result["semantic_available"] = available()
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
