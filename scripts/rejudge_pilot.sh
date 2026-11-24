#!/usr/bin/env bash
# Re-score the M2 pilot transcripts with candidate judge setups (no new generation), so blind human
# labels can decide between them. Order: most promising first (r4 passed a 4-item smoke test where Gemma failed all 4).
#   ctx2   : same Gemma judge, but it sees only the last 2 turns (tests H-len; a recorded deviation)
#   r4     : gpt-oss-20b, reasoned format (R1 style)
#   r1     : same Gemma judge, reasoned format
# Needs: llama.cpp server binary, model files, pilot runs. Set LLAMA_SERVER and MODELS_DIR.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
SRC=${SRC:-runs/pilot-m2-20261122b}
OUT=${OUT:-runs/rejudge-pilot-m2}
LLAMA_SERVER=${LLAMA_SERVER:?set LLAMA_SERVER to llama-server.exe}
MODELS_DIR=${MODELS_DIR:?set MODELS_DIR to the folder with the .gguf files}
mkdir -p "$OUT"

start_server() { # file alias port extra...
  local file=$1 alias=$2 port=$3; shift 3
  "$LLAMA_SERVER" -m "$MODELS_DIR/$file" --alias "$alias" --host 127.0.0.1 --port "$port" -ngl 99 -c 32768 -np 1 --seed 1234 "$@" > "$OUT/server-$alias.log" 2>&1 &
  SERVER_PID=$!
  until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
    sleep 5; kill -0 $SERVER_PID 2>/dev/null || { echo "server $alias died"; tail -5 "$OUT/server-$alias.log"; return 1; }
  done
  echo "server $alias ready"
}
stop_server() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; sleep 5; }

rejudge() { # variant port model extra-args...
  local variant=$1 port=$2 model=$3; shift 3
  for run in "$SRC"/*/; do
    run=${run%/}; name=$(basename "$run")
    [ -f "$run/conversations.jsonl" ] || continue
    dest="$OUT/$variant/$name"
    [ -f "$dest/conversations.jsonl" ] && { echo "skip $variant/$name"; continue; }
    start=$(date +%s)
    python -m psyeval judge "$run" --judge-base-url "http://127.0.0.1:$port/v1" --judge-model "$model" \
      --judge-timeout 1800 --out "$dest" "$@" > "$OUT/$variant-$name.log" 2>&1
    echo "done $variant/$name rc=$? in $(( $(date +%s) - start ))s"
  done
}

start_server gpt-oss-20b-MXFP4.gguf gpt-oss-20b 8084 --reasoning-effort low || exit 1
rejudge r4 8084 gpt-oss-20b --judge-style reasoned --judge-max-tokens 2048
stop_server

start_server gemma-3-12b-it-Q4_K_M.gguf gemma-3-12b-it 8084 || exit 1
rejudge ctx2 8084 gemma-3-12b-it --judge-context-turns 2
stop_server

start_server gemma-3-12b-it-Q4_K_M.gguf gemma-3-12b-it 8084 || exit 1
rejudge r1 8084 gemma-3-12b-it --judge-style reasoned --judge-max-tokens 200
stop_server
echo "REJUDGE COMPLETE $OUT"
