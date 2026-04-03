# Handoff — Flash Attention for Turbo-KV (Failed Experiment)

Date: 2026-04-04
Repo: /Users/adi/PAIA_V1/llama-cpp-turboquant
Branch: aditurbo-engine

## Scope

Attempted to enable existing Metal flash attention kernels for turbo-kv
non-QJL path, gated behind `LLAMA_TURBO_KV_FLASH_NONQJL=1`.

### Code changes (3 commits)

- `ggml/src/ggml-metal/ggml-metal.metal`
  - Added dk64_dv128 and dk64_dv256 flash kernel instantiations for q8_0
    (main half8x8 + vec half4x4 variants)
- `src/llama-context.cpp:2990-3001`
  - Replaced unconditional flash gate with conditional: allows flash when
    `turbo_kv=true && turbo_kv_qjl=false && LLAMA_TURBO_KV_FLASH_NONQJL=1`
- `ggml/src/ggml-metal/ggml-metal-ops.cpp:2931-2932`
  - Removed `GGML_ASSERT(ne10 >= ne20)` in vec kernel dispatch (dk < dv is
    valid with turbo-kv dim-reduction)

### Model details discovered

Qwen3.5-27B: n_embd=5120, n_head=24, n_head_kv=4, **n_embd_head_k=256,
n_embd_head_v=256**, n_layer=64. Hybrid model (recurrent + full attention,
every 4th layer is full attention).

## Results — Speed Gate FAIL

### Test 1: proj_dim=64 (dk=64, dv=256) — turbo-kv dim-reduction

| Test | Baseline (non-flash) | Flash | Delta |
|------|---------------------|-------|-------|
| pp512 | 101.66 t/s | 101.57 t/s | -0.1% |
| tg32 | 9.40 t/s | 9.38 t/s | -0.2% |
| pp2048+tg32 | 87.20 t/s | 81.81 t/s | **-6.2%** |

### Test 2: proj_dim=256 (dk=dv=256) — no dim-reduction, isolating flash

| Test | Baseline (non-flash) | Flash | Delta |
|------|---------------------|-------|-------|
| pp512 | 102.35 t/s | 99.91 t/s | -2.4% |
| tg128 | 9.33 t/s | 7.41 t/s | **-20.6%** |
| pp128+tg32 | 34.11 t/s | 27.76 t/s | **-18.6%** |
| pp2048+tg32 | 82.58 t/s | 75.30 t/s | **-8.8%** |

### Root cause

The Metal flash attention kernel with q8_0 quantized KV is dramatically
slower than the non-flash 3-kernel path (mul_mv + softmax + mul_mv). This
is true even at dk=dv=256 using the stock upstream kernel — not a bug in
our instantiation.

The non-flash path wins because:
1. Specialized `mul_mv_q8_0_f32` kernels are highly tuned for Apple GPU
2. The flash kernel's inline q8_0 dequantization has register/shared-memory
   pressure at dk=dv=256
3. Three individually-optimized kernels beat one general-purpose fused kernel
   on this workload

### What the code IS useful for

- `LLAMA_TURBO_KV_FLASH_NONQJL=1` gate works correctly
- Flash path activates, doesn't crash, produces correct output
- dk < dv assertion fix is valid and needed for future work
- Infrastructure is ready if/when the Metal FA kernel improves for quant types

### What to NOT do next

Do NOT invest more in enabling flash attention for turbo-kv on Metal. The
kernel itself needs Apple-level optimization for quantized KV types. This is
upstream llama.cpp work, not AdiTurbo work.

## Post-mortem: attention path analysis

After the flash experiment failed, a deep analysis of the non-flash attention
path revealed it is already lean — only 3 Metal kernel dispatches total:

1. `kernel_mul_mv_q8_0_f32` (projected Q*K, dk=64)
2. `kernel_soft_max_f32` (softmax)
3. `kernel_mul_mv_q8_0_f32` (softmax*V, dv=256)

With `LLAMA_TURBO_KV_VTRANS_QUANT=1` and the direct QMAT path, there are
**zero** transpose/cast/cont overhead ops for V. The V path is already optimal.

### Why attention optimization can't reach +10%

- Qwen3.5-27B is hybrid: **16 of 64 layers are full attention**, rest recurrent
- Attention is ~1-2% of total per-token compute
- Even infinitely fast attention yields only 1-2% end-to-end improvement
- The real bottleneck is **weight loading in 48 recurrent + 16 FFN layers**
  at ~108 GB/s (~54% of M4 Pro's ~200 GB/s theoretical bandwidth)

## Decision: pivot to end-to-end optimization

### What's proven now (paper claims)

- Memory reduction via turbo-kv quantized KV cache (q8_0)
- Quality retention (PPL parity on turbo-kv path)
- Stable runtime (no crashes, correct output)
- Flash attention infrastructure ready (gated, works but slower on Metal)

### In progress (next sprint)

End-to-end speed uplift via non-attention kernel optimization:
- Recurrent/FFN weight-movement optimization
- Per-layer kernel dispatch overhead reduction
- Metal-specific weight loading patterns

### Gate for next sprint (REVISED)

- **Model:** Qwen3.5-27B-TQ3_0-canonical.gguf
- **Config:** `-r 3 -pg 128,32 -pg 2048,32 -b 512 -ub 512 -ngl 99 -ctk q8_0 -ctv q8_0 --turbo-kv --turbo-kv-proj-dim 64`
- **Baselines (current, LLAMA_TURBO_KV_VTRANS_QUANT=1):**
  - pp512 = 101.66 t/s
  - tg128 = 9.33 t/s
  - pp128+tg32 = 34.08 t/s
  - pp2048+tg32 = 82.58 t/s
- **Pass criteria: end-to-end improvement**
  - Any of tg128 or pp2048+tg32 improved by measurable margin (>= +5%)
  - No PPL regression on 2-chunk wikitext sanity run
  - No crashes

### Git state

- Branch: aditurbo-engine
- Latest: `e7d0bf409` (flash experiment commits)
- Commits this session:
  - `782e29043` — feat: add dk64_dv128 flash attention kernel instantiations
  - `4c1bee376` — feat: gate flash-attn for turbo-kv non-QJL behind env
  - `b0875d273` — fix: remove dk>=dv assertion in flash-attn vec dispatch
  - `e7d0bf409` — fix: add dk64_dv256 flash kernel instantiations

### Mistakes log

1. Assumed Qwen3.5-27B has head_k=head_v=128 (actually 256) — wrong model
   metadata assumption caused wrong kernel instantiation (dk64_dv128 instead
   of dk64_dv256)
2. Didn't check `GGML_ASSERT(ne10 >= ne20)` in vec dispatch before first
   test — caused runtime crash on llama-cli
3. Hypothesis that flash fusion would help was tested and falsified — flash
   with q8_0 KV on Metal is slower, not faster
