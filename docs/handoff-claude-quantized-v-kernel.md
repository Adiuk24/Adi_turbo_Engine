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

---

## Windows GPU Validation + Build-Fix Handoff (for Claude Code)

Date: 2026-04-06  
Target host: Windows machine with 32 GB VRAM GPU + 24 GB system RAM  
Repo branch: `aditurbo-engine`  
Starting commit: `ce1aeea37`

### Mission
Validate full AdiTurbo benchmarks on Windows **with GPU backend active** and close Windows build gaps so test execution is stable and reproducible.

### Known issues from current Windows run (must be addressed)
1. CPU fallback (`no usable GPU found`) due to missing GPU backend build.
2. PowerShell wrapper bugs dropped args (`--model is required`).
3. Dataset path missing (`C:\AdiTurbo\wikitext-2-raw\wiki.test.raw`).
4. Mixed/partial logs from interrupted runs made results non-auditable.

### Required implementation work
1. **Windows GPU build path**
   - Add/verify Windows CMake presets for:
     - CUDA (primary)
     - Vulkan (fallback)
   - Build in `Release`: `llama-cli`, `llama-bench`, `llama-perplexity`, `llama-quantize`.

2. **Windows-native benchmark runner**
   - Add one `.ps1` runner that performs:
     - preflight checks
     - model sanity
     - bench A/B
     - ppl A/B
     - log bundle zip
   - Must avoid argument-array quoting traps and preserve exact model paths.

3. **Benchmark execution (one model at a time, GPU offload)**
   - Use `-ngl 99` and do **not** use `--device none`.
   - Minimum A/B:
     - `Phi-3-14B-TQ3_0.gguf` vs `Phi-3-14B-Q3_K_S.gguf`
   - High-value validation:
     - `Qwen3.5-27B-TQ3_0-v2.gguf` (+ comparator if available on host)

### Command shape for Windows GPU runs
- Bench:
  - `llama-bench --model <model> -ngl 99 -r 3 -p 512 -n 128 -b 512 -ub 512`
- Perplexity:
  - `llama-perplexity --model <model> -f <wiki.test.raw> --chunks 5 -ngl 99 -c 512 -b 256 -ub 256`
- Sanity:
  - `llama-cli --model <model> -ngl 99 -c 1024 -n 64 --single-turn --no-display-prompt --prompt "Explain transformer attention in 3 sentences."`

### Pass/fail gates
- PASS
  - GPU backend active (no CPU fallback warning).
  - All benchmark commands finish without crash.
  - Complete raw logs for every run.
  - Report updates contain measured values only.
- FAIL
  - Any silent CPU fallback.
  - Missing A/B pair or incomplete logs.
  - Wrapper script drops args or routes wrong model path.

### Required report updates after run
1. `docs/aditurbo-test-report.md`
2. `docs/aditurbo-paper-results.csv`
3. `docs/release-gate-checklist.md`
4. Add log bundle path + timestamp in report text.

### Non-negotiables
- Label CPU and GPU results separately.
- No performance claims without auditable logs.
- Keep existing macOS Metal results untouched; add Windows GPU as independent validation rows.
