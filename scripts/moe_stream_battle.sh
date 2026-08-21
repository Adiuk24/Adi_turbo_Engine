#!/usr/bin/env bash
# moe-stream battle test — proves the SSD expert-streaming feature is production-solid.
# ALL runs are CPU-only (-ngl 0), sequential, one model at a time — no GPU/memory-pressure risk.
# Key proof: ZERO-FOOTGUN — every stream run sets ONLY GGML_MOE_STREAM=1 (no --no-repack,
# no --no-op-offload). If the engine's auto-forces work, streaming engages and output stays exact.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/build/bin"
CLI="$BIN/llama-completion"
SRV="$BIN/llama-server"
NOOR="$HOME/noor/noor_hybrid/noor-edge-v2-f16.gguf"
OLMOE="$HOME/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
QWEN="$HOME/models/Qwen3-30B-A3B-Q3_K_M.gguf"
TMP=$(mktemp -d); PASS=0; FAIL=0
say(){ printf '%s\n' "$*"; }
ok(){ PASS=$((PASS+1)); say "  PASS  $*"; }
no(){ FAIL=$((FAIL+1)); say "  FAIL  $*"; }

# parity: baseline (stream off) vs stream on with ONLY GGML_MOE_STREAM=1 (+SLOTS). No --no-repack.
# Args: name model slots prompt ngen
parity(){
  local name=$1 m=$2 slots=$3 p=$4 n=$5
  local off="$TMP/off.$name" on="$TMP/on.$name" oe="$TMP/on.$name.err"
  GGML_MOE_STREAM=0 "$CLI" -m "$m" -ngl 0 --temp 0 -n "$n" -p "$p" --seed 1 --jinja </dev/null >"$off" 2>/dev/null
  GGML_MOE_STREAM=1 GGML_MOE_STREAM_SLOTS=$slots GGML_MOE_STREAM_STATS=1 \
    /usr/bin/time -l "$CLI" -m "$m" -ngl 0 --temp 0 -n "$n" -p "$p" --seed 1 --jinja </dev/null >"$on" 2>"$oe"
  local reg=$(grep -c "GGML_MOE_STREAM: registered" "$oe")
  local rss=$(awk '/maximum resident/{printf "%.2f", $1/1073741824}' "$oe")
  if [ ! -s "$on" ]; then no "$name S=$slots — EMPTY output"; return; fi
  if [ "$reg" -lt 1 ]; then no "$name S=$slots — streaming NOT engaged (footgun leak)"; return; fi
  if diff -q "$off" "$on" >/dev/null; then
    local miss=$(awk -F'misses=' '/moe-stream/{s+=$2+0} END{print s+0}' "$oe")
    ok "$name S=$slots — byte-identical, streamed (misses=$miss), RSS=${rss}GiB"
  else no "$name S=$slots — OUTPUT DIFFERS from baseline"; fi
}

say "=== moe-stream battle test — $(date +%H:%M:%S) ==="
say "[1] NOOR (8 exp, F16) — forced-eviction + normal"
[ -f "$NOOR" ] && { parity noor2 "$NOOR" 2 "Once upon a time" 48; parity noor16 "$NOOR" 16 "The capital of France is" 48; } || say "  SKIP (missing)"
say "[2] OLMoE (64 exp, Q4_K_M) — heavy eviction"
[ -f "$OLMOE" ] && { parity olmoe8 "$OLMOE" 8 "Once upon a time" 48; parity olmoe16 "$OLMOE" 16 "def fibonacci(n):" 48; } || say "  SKIP (missing)"
say "[3] Qwen3-30B-A3B (128 exp, Q3) — FLAGSHIP, 14.7GB in a few GB RAM, CPU-only"
[ -f "$QWEN" ] && { parity qwen16 "$QWEN" 16 "The capital of France is" 32; } || say "  SKIP (missing)"

say "[4] SERVER path (OpenAI API) — the product interface, streaming, -ngl 0"
if [ -f "$QWEN" ]; then
  GGML_MOE_STREAM=1 GGML_MOE_STREAM_SLOTS=16 "$SRV" -m "$QWEN" --host 127.0.0.1 --port 8188 \
    -c 4096 -ngl 0 --jinja --reasoning-budget 0 --alias moe-test >"$TMP/srv.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:8188/health >/dev/null 2>&1 && break; sleep 2; done
  R=$(curl -sf http://127.0.0.1:8188/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"moe-test","messages":[{"role":"user","content":"Reply with exactly: ENGINE OK"}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
      | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['message']['content'].strip()[:40])" 2>/dev/null)
  SRSS=$(ps -o rss= -p $SPID 2>/dev/null | awk '{printf "%.2f", $1/1048576}')
  REG=$(grep -c "GGML_MOE_STREAM: registered" "$TMP/srv.log")
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  if [ -n "$R" ] && [ "$REG" -ge 1 ]; then ok "server — replied '$R', streaming engaged, serverRSS=${SRSS}GiB"
  else no "server — reply='$R' registered=$REG (see $TMP/srv.log)"; fi
else say "  SKIP (missing)"; fi

say ""; say "=== RESULT: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && say "BATTLE-PROOF: GREEN" || say "BATTLE-PROOF: RED — $FAIL failure(s) above"
say "logs: $TMP"
exit $FAIL
