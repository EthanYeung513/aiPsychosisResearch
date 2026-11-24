#!/usr/bin/env bash
# Pipeline check of the grounded-risk suite (dev split, 4 cases), Base condition, R4 judge.
# Checks that the 'help' judge works, that per-case help_from_turn windows apply, and runtime.
# Dev data only. Says nothing about intervention effects.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
LLAMA_SERVER=${LLAMA_SERVER:?set LLAMA_SERVER}
MODELS_DIR=${MODELS_DIR:?set MODELS_DIR}
OUT=${OUT:-runs/pilot-risk-dev}
mkdir -p "$(dirname "$OUT")"

"$LLAMA_SERVER" -m "$MODELS_DIR/Qwen3.5-4B-Q4_K_M.gguf" --alias qwen3.5-4b --host 127.0.0.1 --port 8081 -ngl 99 -c 16384 -np 1 --reasoning off --seed 1234 > "$OUT.target.log" 2>&1 &
T=$!
"$LLAMA_SERVER" -m "$MODELS_DIR/gpt-oss-20b-MXFP4.gguf" --alias gpt-oss-20b --host 127.0.0.1 --port 8084 -ngl 99 -c 16384 -np 1 --seed 1234 --reasoning-effort low > "$OUT.judge.log" 2>&1 &
J=$!
trap 'kill $T $J 2>/dev/null' EXIT
for port in 8081 8084; do
  until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
    sleep 5
    kill -0 $T 2>/dev/null && kill -0 $J 2>/dev/null || { echo "a server died (likely out of GPU memory)"; tail -3 "$OUT".*.log; exit 1; }
  done
  echo "server :$port ready"
done

start=$(date +%s)
python -m psyeval run --suite grounded-risk --split dev --label base \
  --target-base-url http://127.0.0.1:8081/v1 --target-model qwen3.5-4b --temperature 0.7 --max-tokens 1024 --seed 1234 --timeout 900 \
  --judge-base-url http://127.0.0.1:8084/v1 --judge-model gpt-oss-20b --judge-style reasoned --judge-max-tokens 2048 --judge-timeout 1800 \
  --manifest-extra '{"purpose":"grounded-risk dev pipeline check","judge":"R4 gpt-oss-20b MXFP4 reasoned"}' \
  --out "$OUT" > "$OUT.log" 2>&1
echo "done grounded-risk/dev rc=$? in $(( $(date +%s) - start ))s"
echo "RISK PILOT COMPLETE $OUT"
