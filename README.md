# AdiTurbo Engine

**Run MoE models larger than your RAM, from any disk, at usable speed.**

A [llama.cpp](https://github.com/ggml-org/llama.cpp) fork with `moe-stream`:
demand-paged expert streaming for Mixture-of-Experts models. Dense layers and
attention run on your GPU (Metal/CUDA) or CPU; routed experts stay on disk and
stream in per token through a cached slot pool. A 348B-parameter model runs on
a 24 GB laptop over a USB SSD.

Built in Dhaka, Bangladesh as the engine of the ADIOS project — a fully
offline, self-contained AI system on a portable SSD.

## Measured results (not projections)

All numbers from a 24 GB MacBook (M4 Pro) with the model on an external USB
SSD (1.05 GB/s measured wall), temp-0, reproducible from this tree:

| model | size on disk | decode |
|---|---|---|
| DeepSeek-V4-Flash 348B (Q2, top-3) | 90 GB | **1.1–1.3 tok/s** |
| Qwen3.8-Flash-Next 180B (Q3, top-3 of 512) | 84 GB | **2.3–2.9 tok/s** (1.0–1.2 at stock top-10) |
| gpt-oss-120B (MXFP4, top-3) | 59 GB | 1.3 tok/s |
| Qwen3-Next-80B-A3B (Q3) | 36 GB | **4.5 tok/s** |
| Qwen3-30B-A3B (Q3, streamed) | 14 GB | 3.4 tok/s |

- ~10 GB RSS while serving the 348B model (3.7× RAM overcommit).
- Bytes/token 1.23 GB vs 1.47 GB for the comparable upstream draft (PR #25294);
  decode-phase bus utilization 60–76%.
- Verified on macOS/Metal, Linux/x86 CPU, Linux/CUDA (T4, L4), Android arm64.

## Beyond streaming: agent-workload features

**KV persistence across restarts** (`--slot-save-path` + slot restore at boot):
a 29k-token agent preamble goes from 234 s re-prefill to **0.4 s** after a
server restart (0.27 s on an L4).

**Blend — non-prefix KV reuse** (CacheBlend-class, unique to this fork among
GGUF runtimes as far as we know): when a prompt contains chunks of a previous
context in a *different order or position* (reordered tool docs, shuffled RAG
passages), their KV is transplanted and position-shifted instead of recomputed,
with a small repair window per span. Measured: reordered 5.7k-token prompt
**12.8 s → 1.65 s (7.8×)**, 98.7% KV reused, answer quality verified against
planted facts. Usage: `POST /blend/adopt` to snapshot the current context as
donor, then `"blend": true` on chat requests. Requires `--kv-unified -np 1`.

## Quick start (streamed giant)

```bash
GGML_MOE_STREAM=1 GGML_MOE_STREAM_ALLOC_ORDER=1 \
GGML_MOE_STREAM_SLOTS=16 GGML_MOE_STREAM_CHUNKS=4 \
./build/bin/llama-server -m model.gguf \
  -ngl 99 -cmoe -t 8 -np 1 -c 1024 -ub 256 -b 1024 \
  --poll 0 --no-warmup --jinja --reasoning off
```

Notes:
- The GGUF must be a **single file** (`llama-gguf-split --merge` first).
- `--override-kv <arch>.expert_used_count=int:K` is the speed/quality dial:
  fewer experts per token = fewer bytes streamed. K=3 measured quality-safe on
  the models above; K=2 is drafting-only.
- Tuning knobs (all default-off, all temp-0 parity-gated; see
  `ggml/include/ggml-moe-stream.h`): `GGML_MOE_STREAM_NOCACHE`,
  `GGML_MOE_STREAM_EVICT=pinlru`, `GGML_MOE_STREAM_HITFIRST`,
  `GGML_MOE_STREAM_PREFETCH` (+ `_DEPTH`, `_SLOTS`, `_MB`). On a USB-class bus
  they are neutral (it's bandwidth-bound); they target NVMe/big-RAM tiers.
- `GGML_MOE_STREAM_STATS=1` prints per-tensor hit/miss/bytes; a phase profiler
  and routing histogram are built in.

## Honest limits

- Steady-state decode is **storage-bandwidth-bound**: tok/s ≈ bus ÷
  bytes-per-token. Software cannot manufacture bandwidth; quantized models and
  the k-dial reduce the numerator, hardware raises the denominator.
- CPU-only boxes with fast NVMe become **compute-bound** (measured: 8 cores +
  10 GB/s NVMe = 2.1 tok/s on the 80B — slower than a laptop GPU + slow USB).
  Pair a GPU for dense/attention with NVMe for experts.
- f16 streaming is 8.5× slower than Q3 on the same bus (measured). Stream
  quantized.
- Blend v1: single-slot (`-np 1 --kv-unified`), attention models only, donor
  ranges are consume-once between adopts.

## Credits

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
— all upstream credit to its authors and community. The blend feature adapts
ideas from [CacheBlend](https://arxiv.org/abs/2405.16444) /
[LMCache](https://github.com/LMCache/LMCache) research to llama.cpp's KV
primitives. Streaming design informed by our measurements and a teardown of
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c). License: MIT,
same as upstream.
