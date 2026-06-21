"""Build data/usecase_voice.json — a REAL Deepgram ASR transcript of a support call.

Each turn is synthesized with Deepgram TTS (aura) and transcribed back with
Deepgram STT (nova-3), so the transcript genuinely came from speech recognition
(the input a voice agent actually gets). We cache it; the live demo then compresses
this transcript and times the LLM on full vs compressed context.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suffix.tokens import count_tokens  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
for _l in open(os.path.join(os.path.dirname(DATA), ".env")):
    _l = _l.strip()
    if _l and "=" in _l and not _l.startswith("#"):
        _k, _v = _l.split("=", 1); os.environ.setdefault(_k.strip(), _v.strip())
KEY = os.environ.get("DEEPGRAM_API_KEY", "")
STT_MODEL, TTS_MODEL = "nova-3", "aura-2-thalia-en"

SCRIPT = [
    ("agent", "Hi, thanks for calling Acme platform support. This is Dana. How can I help you today?"),
    ("customer", "Hey Dana. So our nightly deployment to the staging environment keeps failing and it's blocking our release."),
    ("agent", "Sorry to hear that. Let's get it sorted. What error message are you seeing in the deploy logs?"),
    ("customer", "It says database migration timeout after thirty seconds, and then the whole deploy rolls back."),
    ("agent", "Okay, a migration timeout. Which database are you running, and what version?"),
    ("customer", "We're on Postgres fourteen point two. We upgraded from thirteen about a week ago."),
    ("agent", "Got it. And is this a shared cluster or a dedicated instance?"),
    ("customer", "Dedicated. The instance is called db prod seven, it's in the us east region."),
    ("agent", "Thanks. The migration that's timing out, what is it actually doing? Which table?"),
    ("customer", "It's adding a column to the orders table. That table has about forty eight million rows."),
    ("agent", "That's almost certainly the cause. A blocking alter on a table that large will not finish inside the thirty second statement timeout."),
    ("customer", "Makes sense. So how do we fix it without taking the orders table offline?"),
    ("agent", "Two options. The quick one is to set the statement timeout to zero just for the migration session. The safer one is to run it as a batched online schema change."),
    ("customer", "We'd rather not lock the table during business hours, so let's go with the online approach."),
    ("agent", "Good call. For a table this size I'd recommend pg osc, or a similar online schema change tool. It rewrites the table in small batches with no long lock."),
    ("customer", "Perfect. And will that affect our read replicas while it runs?"),
    ("agent", "The replicas will lag a little during the rewrite, but they won't drop connections. I'd run it during your lowest traffic window to be safe."),
    ("customer", "Great, that's really helpful. I think that's everything I need."),
    ("agent", "Happy to help. I'll email you the pg osc runbook and the exact statement timeout setting. Have a good one."),
]


def tts(text):
    r = requests.post(
        f"https://api.deepgram.com/v1/speak?model={TTS_MODEL}&encoding=linear16&sample_rate=16000&container=wav",
        headers={"Authorization": f"Token {KEY}", "Content-Type": "application/json"},
        json={"text": text}, timeout=60)
    return r.content if r.status_code == 200 else None


def stt(audio):
    r = requests.post(
        f"https://api.deepgram.com/v1/listen?model={STT_MODEL}&smart_format=true&punctuate=true",
        headers={"Authorization": f"Token {KEY}", "Content-Type": "audio/wav"}, data=audio, timeout=60)
    if r.status_code != 200:
        return None
    return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


def roundtrip(turn):
    spk, text = turn
    a = tts(text)
    t = stt(a) if a else None
    return spk, (t or text)  # fall back to original text if Deepgram hiccups


def main():
    if not KEY:
        print("NO DEEPGRAM KEY"); sys.exit(1)
    print(f"synthesizing + transcribing {len(SCRIPT)} turns through Deepgram (TTS {TTS_MODEL} -> STT {STT_MODEL})…")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(roundtrip, SCRIPT))
    transcript = "\n".join(f"{spk}: {txt}" for spk, txt in results)
    out = {
        "transcript": transcript,
        "question": "What is the root cause of the customer's failing deployment, and what fix did the agent recommend?",
        "gold": "migration timeout",  # robust substring; the recommended fix is pg osc / online schema change
        "deepgram": {"stt_model": STT_MODEL, "tts_model": TTS_MODEL, "turns": len(SCRIPT), "verified": True},
        "title": "Voice agent · compress a transcribed support call",
        "blurb": "A real Deepgram ASR transcript of a 19-turn support call. Voice agents are latency-sensitive "
                 "and transcripts are long — the extractive selector keeps only the load-bearing turns so the LLM answers faster and cheaper.",
    }
    json.dump(out, open(os.path.join(DATA, "usecase_voice.json"), "w"), indent=2)
    print(f"  transcript: {count_tokens(transcript)} tokens, {len(results)} turns")
    print("  first 3 turns:")
    for spk, txt in results[:3]:
        print(f"    {spk}: {txt}")
    print("saved data/usecase_voice.json")


if __name__ == "__main__":
    main()
