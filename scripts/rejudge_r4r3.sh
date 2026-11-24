#!/usr/bin/env bash
# Re-score the grounded pilot transcripts with R4 (gpt-oss-20b, reasoned) + R3 (turn-aware context),
# then apply P-parse. The output sits next to the other variants for calibration_report.py.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LLAMA_SERVER=${LLAMA_SERVER:?set LLAMA_SERVER}
MODELS_DIR=${MODELS_DIR:?set MODELS_DIR}
SRC=runs/pilot-m2-20261122b
OUT=runs/rejudge-pilot-m2/r4r3
mkdir -p "$OUT"

"$LLAMA_SERVER" -m "$MODELS_DIR/gpt-oss-20b-MXFP4.gguf" --alias gpt-oss-20b --host 127.0.0.1 --port 8084 \
  -ngl 99 -c 32768 -np 1 --seed 1234 --reasoning-effort low > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:8084/health > /dev/null; do
  sleep 5; kill -0 $SRV 2>/dev/null || { echo "server died"; tail -3 "$OUT/server.log"; exit 1; }
done
echo "judge ready"

for c in base p1 p2; do
  [ -f "$OUT/$c-grounded/conversations.jsonl" ] && { echo "skip $c"; continue; }
  python -m psyeval judge "$SRC/$c-grounded" --judge-base-url http://127.0.0.1:8084/v1 --judge-model gpt-oss-20b \
    --judge-style reasoned --judge-max-tokens 2048 --judge-timeout 1800 --judge-context-mode turn-aware \
    --out "$OUT/$c-grounded" > "$OUT/$c.log" 2>&1
  echo "done r4r3/$c-grounded rc=$?"
done
python scripts/reparse_runs.py "$OUT/base-grounded" "$OUT/p1-grounded" "$OUT/p2-grounded"
echo "R4R3 COMPLETE"
