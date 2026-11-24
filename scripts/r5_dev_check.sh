#!/usr/bin/env bash
# R5 check (docs/rubric-revisions.md, I8): fresh dev replies that helpful_v2 was NOT designed on.
#   1. Base on the grounded dev cases other than heat_pump_hum (already labelled in round 1).
#   2. P1 on the grounded-risk dev cases (round 1b labelled Base only).
# Judge: R4 (gpt-oss-20b, reasoned); P-parse applied afterwards. Dev data only; says nothing about
# intervention effects.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LLAMA_SERVER=${LLAMA_SERVER:?set LLAMA_SERVER}
MODELS_DIR=${MODELS_DIR:?set MODELS_DIR}
OUT=${OUT:-runs/r5-dev}
mkdir -p "$OUT"

"$LLAMA_SERVER" -m "$MODELS_DIR/Qwen3.5-4B-Q4_K_M.gguf" --alias qwen3.5-4b --host 127.0.0.1 --port 8081 -ngl 99 -c 16384 -np 1 --reasoning off --seed 1234 > "$OUT/target.log" 2>&1 &
T=$!
"$LLAMA_SERVER" -m "$MODELS_DIR/gpt-oss-20b-MXFP4.gguf" --alias gpt-oss-20b --host 127.0.0.1 --port 8084 -ngl 99 -c 16384 -np 1 --seed 1234 --reasoning-effort low > "$OUT/judge.log" 2>&1 &
J=$!
trap 'kill $T $J 2>/dev/null' EXIT
for port in 8081 8084; do
  until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
    sleep 5
    kill -0 $T 2>/dev/null && kill -0 $J 2>/dev/null || { echo "a server died (likely out of GPU memory)"; tail -3 "$OUT"/*.log; exit 1; }
  done
  echo "server :$port ready"
done

COMMON=(--target-base-url http://127.0.0.1:8081/v1 --target-model qwen3.5-4b --temperature 0.7 --max-tokens 1024 --seed 1234 --timeout 900
        --judge-base-url http://127.0.0.1:8084/v1 --judge-model gpt-oss-20b --judge-style reasoned --judge-max-tokens 2048 --judge-timeout 1800)
EXTRA='{"purpose":"R5 helpful_v2 check on fresh dev replies","judge":"R4 gpt-oss-20b MXFP4 reasoned"}'

start=$(date +%s)
python -m psyeval run --suite grounded --split dev --label base \
  --cases novel_signals,accelerator_mission,ai_companion_aware,banana_radiation,retreat_oneness,exam_stress,number_coincidence \
  "${COMMON[@]}" --manifest-extra "$EXTRA" --out "$OUT/base-grounded" > "$OUT/base-grounded.log" 2>&1
echo "done base-grounded rc=$? in $(( $(date +%s) - start ))s"

start=$(date +%s)
python -m psyeval run --suite grounded-risk --split dev --label p1 --system-prompt-file prompts/grounding.txt \
  "${COMMON[@]}" --manifest-extra "$EXTRA" --out "$OUT/p1-grounded-risk" > "$OUT/p1-grounded-risk.log" 2>&1
echo "done p1-grounded-risk rc=$? in $(( $(date +%s) - start ))s"

python scripts/reparse_runs.py "$OUT/base-grounded" "$OUT/p1-grounded-risk"
echo "R5 DEV RUN COMPLETE $OUT"
