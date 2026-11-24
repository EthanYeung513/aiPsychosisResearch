#!/usr/bin/env bash
# M4 pipeline check: train a tiny LoRA -> save -> reload in a fresh process -> serve -> run the harness.
# Proves the plumbing works and measures memory and time. Says nothing about whether training helps.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONPATH=src  # set HF_HOME to where the model was downloaded
PY=.venv/Scripts/python.exe
MODEL=${MODEL:-Qwen/Qwen3.5-0.8B}
OUT=adapters/m4_smoke_local_v2
PORT=8091

echo "== train"
$PY scripts/train_lora.py --model "$MODEL" --data data/drafts/train_concerning.jsonl --limit 8 \
  --max-steps 2 --grad-accum 2 --dtype bfloat16 --out "$OUT" 2>&1 | grep -v -i warn | grep -E '"update"|target_check|optimizer_updates|peak_memory|wall_time|Error|error'

echo "== reload + serve"
$PY scripts/serve_hf.py --model "$MODEL" --adapter "$OUT/adapter" --port $PORT --name m4-smoke > "$OUT/serve.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/health"; do sleep 5; kill -0 $SRV 2>/dev/null || { echo "server died"; cat "$OUT/serve.log"; exit 1; }; done
echo

echo "== harness smoke (grounded dev case, no judge, short replies)"
start=$(date +%s)
python -m psyeval run --suite grounded --cases heat_pump_hum --target-base-url http://127.0.0.1:$PORT/v1 \
  --target-model m4-smoke --max-tokens 128 --timeout 900 --no-judge --label m4-smoke --out runs/m4_smoke_local_eval
echo "harness run took $(( $(date +%s) - start ))s"
python -I -c "
import json
r = json.loads(open('runs/m4_smoke_local_eval/conversations.jsonl', encoding='utf-8').readline())
print('status', r['status'], '| turns', len(r['turns']), '| truncated', sum(t['truncated'] for t in r['turns']))
print('turn 1 reply:', r['turns'][0]['assistant'][:300] if r['turns'] else None)"
echo "M4 SMOKE COMPLETE"
