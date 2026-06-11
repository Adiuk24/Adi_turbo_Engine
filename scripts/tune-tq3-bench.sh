#!/bin/bash
# tune-tq3-bench.sh — Find optimal TQ3_0 inference settings
# Usage: ./scripts/tune-tq3-bench.sh <model.gguf> [ngl]

set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> [ngl]}"
NGL="${2:-99}"
BENCH="${BENCH:-./build/bin/llama-bench}"

if [ ! -f "$BENCH" ]; then
    echo "Error: llama-bench not found at $BENCH"
    echo "Build: cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(sysctl -n hw.ncpu 2>/dev/null || nproc) llama-bench"
    exit 1
fi

echo "=== AdiTurbo TQ3_0 Tuner ==="
echo "Model: $(basename "$MODEL")"
echo "GPU layers: $NGL"
echo ""

PROMPT_SIZES="128 512 1024 2048"
GEN_SIZES="32 128"

# Detect physical cores
if [[ "$(uname)" == "Darwin" ]]; then
    PCORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 8)
    ECORES=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 4)
    THREAD_COUNTS="1 2 4 ${PCORES} $((PCORES + ECORES))"
else
    NCPU=$(nproc 2>/dev/null || echo 8)
    THREAD_COUNTS="1 2 4 $((NCPU / 2)) ${NCPU}"
fi

# Deduplicate thread counts
THREAD_COUNTS=$(echo "$THREAD_COUNTS" | tr ' ' '\n' | sort -un | tr '\n' ' ')

echo "Thread counts to test: $THREAD_COUNTS"
echo "Prompt sizes: $PROMPT_SIZES"
echo "Generation sizes: $GEN_SIZES"
# Parse avg_ts (tokens/sec) out of llama-bench JSON output.
bench_ts() {
    "$BENCH" "$@" -o json 2>/dev/null | python3 -c '
import json, sys
try:
    rows = json.load(sys.stdin)
    print(f"{rows[0][\"avg_ts\"]:.2f} t/s")
except Exception:
    print("FAILED")
'
}

echo ""
echo "--- Prompt Processing (pp) ---"

for t in $THREAD_COUNTS; do
    for pp in $PROMPT_SIZES; do
        echo -n "threads=$t pp=$pp: "
        bench_ts -m "$MODEL" -p "$pp" -n 0 -ngl "$NGL" -t "$t" -r 1
    done
done

echo ""
echo "--- Token Generation (tg) ---"

for t in $THREAD_COUNTS; do
    for tg in $GEN_SIZES; do
        echo -n "threads=$t tg=$tg: "
        bench_ts -m "$MODEL" -p 0 -n "$tg" -ngl "$NGL" -t "$t" -r 1
    done
done

echo ""
echo "=== Done ==="
