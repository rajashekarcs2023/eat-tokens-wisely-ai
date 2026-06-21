#!/usr/bin/env bash
# Rebuild every data artifact the demo/server needs, in dependency order.
# The committed repo already includes these (so the server runs on a fresh clone);
# run this only to regenerate from scratch. Steps marked [KEY] call the Anthropic
# and/or Deepgram APIs (need .env). Steps without [KEY] are CPU-only.
set -e
cd "$(dirname "$0")/.."

echo "1/8  prepare_data        — cache HotpotQA train/test splits"
python scripts/prepare_data.py

echo "2/8  train_scorer        — fit the CPU keep-scorer -> data/keep_scorer.joblib"
python scripts/train_scorer.py

echo "3/8  run_eval       [KEY]— pareto frontier + scorer AUC -> data/pareto.json"
python scripts/run_eval.py

echo "4/8  run_ablation   [KEY]— label-source ablation + CIs -> data/pareto_v2.json"
python scripts/run_ablation.py

echo "5/8  run_redundancy [KEY]— gold-fact redundancy probe -> data/redundancy.json"
python scripts/run_redundancy.py

echo "6/8  crosstask      [KEY]— cross-modality generalization -> data/crosstask_*.json"
for d in squad coqa 2wiki narrativeqa; do python scripts/crosstask.py --dataset "$d" --n 80; done

echo "7/8  make_usecases       — Use Cases tab scenarios -> data/usecases.json"
python scripts/make_usecases.py

echo "8/9  make_voice     [KEY]— real Deepgram TTS->STT transcript -> data/usecase_voice.json"
python scripts/make_voice.py

echo "9/9  mcp_demo            — call REAL MCP servers (filesystem/git/sqlite via npx+uvx) -> data/usecase_mcp.json"
python scripts/mcp_demo.py

echo "optional: structural_qa [KEY] — lossless-reads-natively proof -> data/structural_qa.json"
echo "done. start the demo:  python -m uvicorn server.app:app --port 8000"
