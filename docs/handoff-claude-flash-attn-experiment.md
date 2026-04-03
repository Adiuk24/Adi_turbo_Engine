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

## Next sprint: non-flash turbo-kv kernel optimization

### Target

Optimize the Q8_0 `V x softmax(KQ)` non-flash hot path on Metal. Write a
fused/tuned kernel specifically for turbo-kv q8_0 attention on Apple GPU.

### Where to look

1. **Transpose/cast/cont overhead in V path** (llama-graph.cpp:2367-2400):
   - Non-flash path does `ggml_cast(ggml_transpose(v), F16)` or
     `ggml_cast(v, F16)` for quantized V
   - Each is a separate kernel dispatch + memory round-trip
   - Fusing these or eliminating the cast is the first target

2. **mul_mv_q8_0_f32 for attention*V** (ggml-metal.metal:3580-3651):
   - This is the kernel that multiplies softmax weights by V
   - Can be specialized for the attention use case (softmax output is sparse,
     small batch, contiguous)

3. **Softmax kernel** (ggml-metal.metal:1970-2073):
   - Already well-optimized, but output could be fused with V multiply

### Gate for next sprint

- **Model:** Qwen3.5-27B-TQ3_0-canonical.gguf
- **Config:** `-r 3 -pg 128,32 -pg 2048,32 -b 512 -ub 512 -ngl 99 -ctk q8_0 -ctv q8_0 --turbo-kv --turbo-kv-proj-dim 64`
- **Baseline (current):** tg128 = 9.33 t/s (non-flash, LLAMA_TURBO_KV_VTRANS_QUANT=1)
- **Pass criteria:**
  - tg128 >= 10.27 t/s (strictly +10% over 9.33)
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
