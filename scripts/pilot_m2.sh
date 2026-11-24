#!/usr/bin/env bash
# M2 pilot: Base vs P1 (grounding prompt) vs P2 (anti-sycophancy prompt) on one matched pair.
# Purpose: check response quality, judge behaviour and runtime. NOT for comparing interventions.
# Needs local llama-server instances: target on :8081, judge on :8082.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
STAMP=${STAMP:-$(date +%Y%m%d-%H%M)}
OUT=runs/pilot-m2-$STAMP
mkdir -p runs
TARGET=(--target-base-url http://127.0.0.1:8081/v1 --target-model qwen3.5-4b --temperature 0.7 --max-tokens 1024 --seed 1234 --timeout 900)
JUDGE=(--judge-base-url http://127.0.0.1:8082/v1 --judge-model gemma-3-12b-it --judge-temperature 0 --judge-max-tokens 16 --judge-timeout 900)
EXTRA='{"target_file":"Qwen3.5-4B-Q4_K_M.gguf","target_sha256":"217002fa8008ef4fc6d548353a3aebef055a96e6ff8e0e2c0953ef8254a89c14","judge_file":"gemma-3-12b-it-Q4_K_M.gguf","judge_sha256":"7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5","server":"llama.cpp b12633 vulkan, -ngl 99 -c 16384 --reasoning off (target)","purpose":"M2 pilot"}'

for port in 8081 8082; do
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do sleep 3; done
  echo "server :$port ready"
done

for cond in base p1 p2; do
  case $cond in
    base) SP=() ;;
    p1) SP=(--system-prompt-file prompts/grounding.txt) ;;
    p2) SP=(--system-prompt-file prompts/anti_sycophancy_reflection.txt) ;;
  esac
  for suite in psychosis-bench grounded; do
    case $suite in psychosis-bench) CASES=conduit_explicit ;; grounded) CASES=heat_pump_hum ;; esac
    start=$(date +%s)
    python -m psyeval run --suite $suite --cases $CASES --label $cond "${TARGET[@]}" "${JUDGE[@]}" \
      "${SP[@]}" --manifest-extra "$EXTRA" --out "$OUT/$cond-$suite" > "$OUT-$cond-$suite.log" 2>&1
    rc=$?
    echo "done $cond/$suite rc=$rc in $(( $(date +%s) - start ))s"
  done
done
echo "PILOT COMPLETE $OUT"
