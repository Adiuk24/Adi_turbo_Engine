# TurboQuant KV Cache Design

## Goal

128K context window on 8GB RAM.
Budget: **2.1GB** for the entire KV cache.

## The Problem

Standard KV cache at FP16 for a 120B MoE with GQA (8 KV heads, 128 head dim, 64 layers):

```
Per token per layer: 8 heads * 128 dim * 2 (K+V) = 2048 values
Total: 2048 * 64 layers * 131072 tokens * 2 bytes = 34.4 GB
```

That's 4x the entire RAM budget. We need **16x compression**.

## Three-Level Strategy

### Level 1: TQ3_0 as KV Type (5.2x compression, ~6.6GB)

Almost free. The machinery exists:

- `ggml_set_rows()` quantizes on write (calls `quantize_row_tq3_0_ref`)
- Non-flash attention: `ggml_mul_mat(k, q)` dispatches to `ggml_vec_dot_tq3_0_q8_K`
- Flash attention: uses `vec_dot` for K, `to_float` for V (both registered for TQ3_0)
- Just add `GGML_TYPE_TQ3_0` and `GGML_TYPE_TQ4_0` to `kv_cache_types` in `common/arg.cpp`

```
34.4 GB * (3.09/16) = 6.6 GB  -- not enough alone
```

**Code change: ~5 lines.** Add to type list, done.

### Level 2: PolarQuant (Better Quality at Same Bits)

Standard symmetric quantization (TQ3_0) loses quality on KV vectors because:

- K vectors have positional encoding outliers (RoPE dimensions have 10-100x larger values)
- V vectors are zero-mean but have per-channel variance spread
- Symmetric quant wastes dynamic range on the asymmetry

**PolarQuant** uses per-channel zero points + per-group scales:

```c
typedef struct {
    ggml_half  d;                    // group scale (2 bytes)
    int8_t     zp[QK_K/32];         // per-32-element zero points (8 bytes for QK_K=256)
    uint8_t    qs[3*QK_K/8];        // 3-bit packed values (96 bytes)
} block_pq3_0;
// Total: 106 bytes for 256 values = 3.31 bpw
```

Why this helps:
- RoPE dimensions: zero point absorbs the DC offset, scale handles the range
- Non-RoPE dimensions: zero point ≈ 0, behaves like symmetric
- Net: ~0.5 perplexity improvement over TQ3_0 at same compression

```
34.4 GB * (3.31/16) = 7.1 GB  -- worse compression, but better quality
```

PolarQuant alone doesn't hit 2.1GB. It's a quality upgrade at ~3 bpw, not a compression upgrade. Use it WITH Level 3.

### Level 3: QJL — Quantized Johnson-Lindenstrauss (the multiplier)

This is the big lever. Instead of storing full KV vectors, store **random projections**.

**JL Lemma**: For any set of n points in R^d, a random projection to R^m where
m = O(log(n) / epsilon^2) preserves all pairwise distances within factor (1 +/- epsilon).

For attention: we compute `softmax(Q * K^T / sqrt(d)) * V`.
The JL projection preserves the Q*K^T dot products, so attention scores are approximately preserved.

**Projection parameters:**
- Original head dim: d = 128
- Target dim: m = 64 (2x reduction, epsilon ≈ 0.1 for 128K tokens)
- Random matrix R: d x m, drawn once per layer, stored as int8 (tiny overhead)

**On KV write:**
```
K_proj = K * R          // [n_heads, 128] -> [n_heads, 64]
V_proj = V * R          // same
quantize(K_proj) -> KV cache   // 3-bit quantize the projected vector
```

**On attention read:**
```
Q_proj = Q * R          // project query too
scores = Q_proj * K_proj^T / sqrt(m)  // attention in projected space
output_proj = scores * V_proj
output = output_proj * R^T  // project back (R^T ≈ R^{-1} for random orthogonal)
```

**Combined compression:**

```
Dim reduction:  128 -> 64 = 2x
Quantization:   16 bpw -> 3.31 bpw (PolarQuant) = 4.8x
Total:          2 * 4.8 = 9.7x compression

34.4 GB / 9.7 = 3.5 GB  -- close but not 2.1GB
```

To hit 2.1GB, we need more aggressive projection OR fewer KV heads:

| Config | Dim Reduction | Quant | Total Comp | KV Size |
|--------|--------------|-------|-----------|---------|
| 128→64, PQ3 | 2x | 4.8x | 9.7x | 3.5 GB |
| 128→48, PQ3 | 2.67x | 4.8x | 12.8x | 2.7 GB |
| 128→64, 2-bit | 2x | 8x | 16x | 2.1 GB |
| 128→48, PQ3, GQA-4 | 2.67x | 4.8x | 12.8x | 1.3 GB |
| MLA (d=256), PQ3 | 1x | 4.8x | 4.8x | 0.4 GB |

**Best path: 128→64 projection + 2-bit quantization = exactly 2.1GB.**

Or if the model uses GQA with 4 KV heads instead of 8, PQ3 with mild projection hits it.

## Implementation Plan

### Phase 1: TQ KV Cache (1 day)

Add TQ3_0 and TQ4_0 to KV cache type list. Test with existing model.

Files:
- `common/arg.cpp` — add to `kv_cache_types`

Test:
```bash
./llama-cli -m model.gguf -ctk TQ4_0 -ctv TQ4_0 -p "Hello" -n 64
./llama-cli -m model.gguf -ctk TQ3_0 -ctv TQ3_0 -p "Hello" -n 64
```

### Phase 2: PolarQuant Block Type (3 days)

New GGML type `GGML_TYPE_PQ3_0`:

```c
// In ggml-common.h
typedef struct {
    uint8_t qs[3*QK_K/8];   // 3-bit packed values (96 bytes)
    int8_t  zp[QK_K/32];    // per-32-element zero points (8 bytes)
    ggml_half d;             // group scale (2 bytes)
} block_pq3_0;
// 106 bytes for 256 values = 3.31 bpw
```

Files:
- `ggml/include/ggml.h` — add `GGML_TYPE_PQ3_0`
- `ggml/src/ggml-common.h` — struct definition
- `ggml/src/ggml-quants.c` — quantize/dequantize (PolarQuant-aware)
- `ggml/src/ggml.c` — type traits registration
- `ggml/src/ggml-cpu/quants.c` — generic dot product
- `ggml/src/ggml-cpu/arch/arm/quants.c` — NEON kernel
- `common/arg.cpp` — add to `kv_cache_types`

The quantize function differs from TQ3_0:
```c
void quantize_row_pq3_0_ref(const float * x, block_pq3_0 * y, int64_t k) {
    // For each block of 256 values:
    // 1. Compute per-32-element means -> zero points
    // 2. Subtract zero points
    // 3. Compute scale from max absolute residual
    // 4. Quantize residuals to 3-bit symmetric
}
```

The NEON dot product is similar to TQ3_0 but subtracts the zero point per group.

### Phase 3: QJL Projection (5 days)

The projection layer wraps the KV cache:

```
                    ┌──────────────────────────┐
                    │     TurboQuant KV Cache   │
                    │                           │
  K_cur ──→ [R_k] ──→ [PQ3 quantize] ──→ cache_k │
  V_cur ──→ [R_v] ──→ [PQ3 quantize] ──→ cache_v │
                    │                           │
  Q_cur ──→ [R_k] ──→ attn(q', k', v') ──→ [R_v^T] ──→ output
                    └──────────────────────────┘
```

Key components:
- **Projection matrices**: one R_k and R_v per layer, shape [head_dim, proj_dim]
  - Initialized as random orthogonal (QR decomposition of Gaussian)
  - Stored as FP16, total overhead: 2 * 64 * 128 * 2 * 64 = 2MB (negligible)
- **Modified KV cache**: stores projected+quantized vectors
  - K shape: [proj_dim * n_kv_heads, kv_size, n_stream] at PQ3_0
  - V shape: same
- **Modified attention**: Q is projected before computing scores

Files:
- `src/llama-kv-cache-turbo.h` — TurboQuant KV cache class (extends llama_kv_cache)
- `src/llama-kv-cache-turbo.cpp` — implementation
- `src/llama-graph.cpp` — new `build_attn_turbo()` path
- `src/llama-context.cpp` — instantiate TurboQuant cache when `--turbo-kv` flag

### Phase 4: Adaptive Precision (2 days)

Not all layers are equal:
- First 2 layers: use FP16 KV (quality-critical for token disambiguation)
- Last 2 layers: use FP16 KV (quality-critical for output generation)
- Middle layers: PQ3 + QJL (bulk of the cache, tolerates compression)

This adds ~0.5GB overhead for the 4 full-precision layers but significantly improves output quality.

## Memory Budget Breakdown (128K context, 120B MoE, 8 KV heads)

| Component | Size |
|-----------|------|
| Active model weights (TQ3_0, ~10B active) | 3.8 GB |
| KV cache: 4 FP16 layers (first/last 2) | 0.5 GB |
| KV cache: 60 PQ3+QJL layers (64→proj, 3-bit) | 1.6 GB |
| Projection matrices (FP16) | 0.002 GB |
| Scratch buffers + overhead | 0.1 GB |
| **Total** | **6.0 GB** |

## Attention Quality Analysis

| Method | KQ Dot Product Error | Perplexity Delta | Compression |
|--------|---------------------|------------------|-------------|
| FP16 (baseline) | 0 | 0 | 1x |
| Q4_0 KV | ~0.01% | +0.05 | 4x |
| TQ3_0 KV | ~0.1% | +0.15 | 5.2x |
| PQ3_0 KV | ~0.05% | +0.08 | 4.8x |
| PQ3_0 + QJL 2x | ~0.5% | +0.3 | 9.7x |
| PQ3_0 + QJL 2x + adaptive | ~0.3% | +0.2 | ~8x avg |

Numbers are estimates based on published QJL/PolarQuant research. Actual numbers require benchmarking.

## Risk Analysis

1. **QJL projection error compounds across layers** — mitigated by adaptive precision (full FP16 on first/last layers)
2. **PQ3 zero-point overhead** — 8 extra bytes per 256 values (0.25 bpw overhead). Worth it for KV quality.
3. **Projection matmul overhead** — extra [128, 64] matmul per layer per token. On M4 Pro at BF16: ~0.01ms per layer. Negligible for generation (1 token), adds ~0.6ms for 1K prompt tokens.
4. **Flash attention compatibility** — QJL changes KV dimensions. Flash attention works on the projected dimensions, no special kernel needed. The projection/unprojection happens outside FA.

## API

```bash
# Phase 1: Simple TQ KV
./llama-cli -m model.gguf -ctk TQ3_0 -ctv TQ3_0 -c 131072

# Phase 2: PolarQuant KV
./llama-cli -m model.gguf -ctk PQ3_0 -ctv PQ3_0 -c 131072

# Phase 3: Full TurboQuant (PQ3 + QJL)
./llama-cli -m model.gguf --turbo-kv -c 131072
# Equivalent to: -ctk PQ3_0 -ctv PQ3_0 --kv-proj-dim 64 --kv-adaptive-layers 2
```
