# Handoff — Quantized-V Backend Path (Codex)

Date: 2026-04-03
Repo: /Users/adi/PAIA_V1/llama-cpp-turboquant
Branch: aditurbo-engine

## Scope completed
Implemented and validated a backend-safe direct quantized-V path experiment for non-flash turbo-kv.

### Code changes
- /Users/adi/PAIA_V1/llama-cpp-turboquant/src/llama-model.cpp
  - Added env-gated transposed V cache allowance for quantized V:
    - `LLAMA_TURBO_KV_VTRANS_QUANT`
- /Users/adi/PAIA_V1/llama-cpp-turboquant/src/llama-graph.cpp
  - Added env-gated direct quantized-V path in attention graph:
    - `LLAMA_TURBO_KV_DIRECT_QMAT_V`
  - Optional cast for kq to f16 when using direct path:
    - `LLAMA_TURBO_KV_DIRECT_QMAT_V_F16`
- /Users/adi/PAIA_V1/llama-cpp-turboquant/ggml/src/ggml-metal/ggml-metal.metal
  - Added Metal kernel `kernel_mul_mv_q8_0_f16` to support q8_0 x f16 path.

## Validation results (measurable)
Model used:
- `/Users/adi/PAIA_V1/models/Qwen3.5-27B-TQ3_0-canonical.gguf`

Benchmark config:
- `-r 1 -pg 128,32 -n 32 -b 512 -ub 512 -ngl 99 -ctk q8_0 -ctv q8_0 --turbo-kv --turbo-kv-qjl --turbo-kv-proj-dim 64`

### Speed (llama-bench)
- Baseline turbo path (`LLAMA_TURBO_KV_VTRANS_QUANT=1`):
  - `tg32 = 8.65 t/s`
- Direct quantized-V (`+ LLAMA_TURBO_KV_DIRECT_QMAT_V=1`):
  - `tg32 = 8.62 t/s`
- Direct quantized-V + f16 softmax (`+ LLAMA_TURBO_KV_DIRECT_QMAT_V_F16=1`):
  - `tg32 = 8.62 t/s`

Conclusion: no speed gain in generation path on this model/config.

### Quality (llama-perplexity, 2 chunks)
Config:
- `--chunks 2 -c 2048 -b 512 -ub 512 -ngl 99 -ctk q8_0 -ctv q8_0 --turbo-kv --turbo-kv-qjl --turbo-kv-proj-dim 64`

- Baseline (`LLAMA_TURBO_KV_VTRANS_QUANT=1`):
  - `PPL = 50.1178 +/- 3.14651`
- Direct+f16 (`LLAMA_TURBO_KV_VTRANS_QUANT=1 LLAMA_TURBO_KV_DIRECT_QMAT_V=1 LLAMA_TURBO_KV_DIRECT_QMAT_V_F16=1`):
  - `PPL = 50.1178 +/- 3.14651`

Conclusion: parity holds on this short check; no observed quality regression.

## Gate status from this handoff
- Build: PASS
- Quant core test: unchanged known upstream exception only (`q5_1`)
- Runtime no-crash: PASS (for final tested configurations)
- Speed delta vs baseline turbo path: FAIL (no positive tg gain)

## Exact next step for Claude
Implement true fused backend kernel path for quantized-V attention (non-flash) that removes remaining hot-path overhead, then re-run only this gate:
- Pass criteria: `tg32` strictly greater than baseline turbo path on the same 27B test recipe, with no PPL regression on the same 2-chunk sanity check.

