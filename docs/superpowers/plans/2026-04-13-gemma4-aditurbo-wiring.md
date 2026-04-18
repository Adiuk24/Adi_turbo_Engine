# Gemma 4 Architecture Wiring — AdiTurbo Engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up Gemma 4 ISWA architecture so AdiTurbo can load and run the existing Gemma 4 26B-A4B GGUF at `/Users/adi/noor/models/google_gemma-4-26B-A4B-it-IQ3_M.gguf`.

**Architecture:** The model graph implementation already exists at `src/models/gemma4-iswa.cpp` (untracked). What's missing is all the registration/wiring code — enum entries, struct fields, hparams loading, tensor loading, tensor name mappings, and CMake build integration. All of this exists in upstream `origin/master` and needs to be cherry-picked into our fork.

**Tech Stack:** C/C++ (llama.cpp), CMake, GGUF format

**Test model GGUF metadata (26B-A4B):**
```
general.architecture = gemma4
gemma4.block_count = 30
gemma4.embedding_length = 2816
gemma4.feed_forward_length = 2112
gemma4.expert_count = 128, expert_used_count = 8
gemma4.expert_feed_forward_length = 704
gemma4.attention.head_count = 16
gemma4.attention.head_count_kv = [8,8,8,8,8,2, ...] (per-layer, 8 for SWA, 2 for global)
gemma4.attention.key_length = 512 (global), key_length_swa = 256
gemma4.attention.sliding_window = 1024
gemma4.attention.sliding_window_pattern = [T,T,T,T,T,F, ...] (5:1 SWA:global)
gemma4.rope.freq_base = 1000000 (global), freq_base_swa = 10000
gemma4.final_logit_softcapping = 30.0
gemma4.embedding_length_per_layer_input = 0 (PLE NOT used on 26B)
gemma4.attention.shared_kv_layers = 0 (no KV sharing on 26B)
```

**Upstream reference:** All changes exist in `origin/master` (fetched). Use `git show origin/master:<file>` to verify.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/llama-arch.h` | Modify | Add `LLM_ARCH_GEMMA4` enum + new LLM_KV/LLM_TENSOR enums |
| `src/llama-arch.cpp` | Modify | Add arch name mapping + tensor name strings |
| `src/llama-hparams.h` | Modify | Add `n_embd_per_layer`, `n_ff_exp`, `n_embd_head_k_swa`, `n_embd_head_v_swa` fields |
| `src/llama-model.h` | Modify | Add layer struct fields: `ffn_gate_inp_s`, `ffn_post_norm_1/2`, `ffn_pre_norm_2`, `out_scale`, `per_layer_tok_embd` |
| `src/llama-model.cpp` | Modify | Add 5 `case LLM_ARCH_GEMMA4:` blocks (hparams, tensors, KV reuse, graph builder, rope type) |
| `src/models/models.h` | Modify | Add `llm_build_gemma4_iswa` class declaration |
| `src/CMakeLists.txt` | Modify | Add `models/gemma4-iswa.cpp` to build |
| `src/models/gemma4-iswa.cpp` | Already exists | No changes needed (just needs to compile) |

---

### Task 1: Add Gemma 4 Enums to llama-arch.h

**Files:**
- Modify: `src/llama-arch.h`

- [ ] **Step 1: Add LLM_ARCH_GEMMA4 to arch enum**

In `src/llama-arch.h`, after line 62 (`LLM_ARCH_GEMMA3N,`), add:

```cpp
    LLM_ARCH_GEMMA4,
```

- [ ] **Step 2: Add new LLM_KV keys**

Find the `LLM_KV_` enum section. Add these entries (check exact positions by diffing with `git show origin/master:src/llama-arch.h`):

```cpp
    LLM_KV_EMBEDDING_LENGTH_PER_LAYER,
    LLM_KV_ATTENTION_SHARED_KV_LAYERS,
    LLM_KV_EXPERT_FEED_FORWARD_LENGTH,
```

Note: `LLM_KV_ATTENTION_KEY_LENGTH_SWA` and `LLM_KV_ATTENTION_VALUE_LENGTH_SWA` may already exist — verify first.

- [ ] **Step 3: Add new LLM_TENSOR entries**

Find the `LLM_TENSOR_` enum section. Add after `LLM_TENSOR_FFN_POST_NORM`:

```cpp
    LLM_TENSOR_FFN_POST_NORM_1,
    LLM_TENSOR_FFN_POST_NORM_2,
    LLM_TENSOR_FFN_PRE_NORM_2,
```

And add near the end of the enum (before the closing):

```cpp
    LLM_TENSOR_LAYER_OUT_SCALE,
```

- [ ] **Step 4: Verify against upstream**

Run: `git show origin/master:src/llama-arch.h | grep -n "GEMMA4\|FFN_POST_NORM_1\|FFN_PRE_NORM_2\|LAYER_OUT_SCALE\|EMBEDDING_LENGTH_PER_LAYER\|SHARED_KV_LAYERS\|EXPERT_FEED_FORWARD"`

Compare line numbers and positions to ensure no enum value clashes.

- [ ] **Step 5: Commit**

```bash
git add src/llama-arch.h
git commit -m "feat: add Gemma 4 arch enum + new LLM_KV/LLM_TENSOR entries"
```

---

### Task 2: Add Hparams Fields to llama-hparams.h

**Files:**
- Modify: `src/llama-hparams.h`

- [ ] **Step 1: Add new hparam fields**

The upstream version has these fields that our fork is missing. Add them:

After the existing `n_embd_head_k` / `n_embd_head_v` fields:
```cpp
    uint32_t n_embd_head_k_swa;
    uint32_t n_embd_head_v_swa;
```

After the existing `n_ff` field:
```cpp
    uint32_t n_ff_exp           = 0;
```

After the existing `n_embd_altup` field:
```cpp
    uint32_t n_embd_per_layer = 0;
```

Verify exact positions: `git show origin/master:src/llama-hparams.h | grep -n "n_embd_head_k_swa\|n_ff_exp\|n_embd_per_layer"`

- [ ] **Step 2: Verify the `n_embd_head_k(il)` and similar per-layer accessors handle SWA variant**

Check that `n_embd_head_k(uint32_t il)` already has per-layer support (it does — it uses `hparams.n_embd_head_k` array). The SWA variant needs to return `n_embd_head_k_swa` for SWA layers and `n_embd_head_k` for global layers. Verify upstream handles this:

`git show origin/master:src/llama-hparams.h | grep -A10 "n_embd_head_k("`

If the upstream accessor already dispatches on `is_swa(il)`, mirror that logic.

- [ ] **Step 3: Commit**

```bash
git add src/llama-hparams.h
git commit -m "feat: add Gemma 4 hparams — n_embd_per_layer, n_ff_exp, SWA head dims"
```

---

### Task 3: Add Layer Struct Fields to llama-model.h

**Files:**
- Modify: `src/llama-model.h`

- [ ] **Step 1: Add Gemma 4-specific layer fields**

After `ffn_post_norm` (around line 272):
```cpp
    struct ggml_tensor * ffn_post_norm_1  = nullptr; // gemma4
    struct ggml_tensor * ffn_post_norm_2  = nullptr; // gemma4
    struct ggml_tensor * ffn_pre_norm_2   = nullptr; // gemma4
```

After `ffn_gate_inp` (around line 287), before `ffn_gate_exps`:
```cpp
    struct ggml_tensor * ffn_gate_inp_s    = nullptr; // gemma4
```

In the layer struct near the end (where `rope_freqs` etc. are), add:
```cpp
    struct ggml_tensor * out_scale = nullptr;
```

- [ ] **Step 2: Add model-level per_layer_tok_embd field**

In the `llama_model` struct (NOT the layer struct), after `tok_embd`:
```cpp
    struct ggml_tensor * per_layer_tok_embd   = nullptr;
```

Verify positions against upstream:
`git show origin/master:src/llama-model.h | grep -n "ffn_post_norm_1\|ffn_gate_inp_s\|out_scale\|per_layer_tok_embd"`

- [ ] **Step 3: Commit**

```bash
git add src/llama-model.h
git commit -m "feat: add Gemma 4 layer struct fields — MoE norms, router scale, out_scale"
```

---

### Task 4: Add Tensor Name Mappings to llama-arch.cpp

**Files:**
- Modify: `src/llama-arch.cpp`

- [ ] **Step 1: Add arch name mapping**

After line 58 (`{ LLM_ARCH_GEMMA3N, "gemma3n" },`):
```cpp
    { LLM_ARCH_GEMMA4,           "gemma4"           },
```

- [ ] **Step 2: Add LLM_KV string mappings**

Find the `LLM_KV_NAMES` map. Add entries for the new keys:

```cpp
    { LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  "%s.embedding_length_per_layer_input" },
    { LLM_KV_ATTENTION_SHARED_KV_LAYERS,  "%s.attention.shared_kv_layers" },
    { LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  "%s.expert_feed_forward_length" },
```

Verify exact strings against GGUF metadata: the 26B GGUF has `gemma4.embedding_length_per_layer_input`, `gemma4.attention.shared_kv_layers`, `gemma4.expert_feed_forward_length`.

- [ ] **Step 3: Add LLM_TENSOR string mappings**

Find the `LLM_TENSOR_NAMES` map. Add:

```cpp
    { LLM_TENSOR_FFN_POST_NORM_1,                        "blk.%d.post_ffw_norm_1" },
    { LLM_TENSOR_FFN_POST_NORM_2,                        "blk.%d.post_ffw_norm_2" },
    { LLM_TENSOR_FFN_PRE_NORM_2,                         "blk.%d.pre_ffw_norm_2" },
    { LLM_TENSOR_LAYER_OUT_SCALE,                        "blk.%d.layer_output_scale" },
```

These match the GGUF tensor names exactly (verified: `blk.0.post_ffw_norm_1.weight`, `blk.0.pre_ffw_norm_2.weight`, `blk.0.layer_output_scale.weight`).

Note: `ffn_gate_inp.scale` uses the existing `LLM_TENSOR_FFN_GATE_INP` with `"scale"` suffix — the `create_tensor(tn(..., "scale", i), ...)` call handles this. No new mapping needed.

- [ ] **Step 4: Commit**

```bash
git add src/llama-arch.cpp
git commit -m "feat: add Gemma 4 arch name, KV keys, and tensor name mappings"
```

---

### Task 5: Add Class Declaration to models.h

**Files:**
- Modify: `src/models/models.h`

- [ ] **Step 1: Add llm_build_gemma4_iswa declaration**

After the `llm_build_gemma3` template (around line 245), add:

```cpp
struct llm_build_gemma4_iswa : public llm_graph_context {
    const llama_model & model;

    const int64_t n_embd_per_layer;

    llm_build_gemma4_iswa(const llama_model & model, const llm_graph_params & params);

    // TODO: refactor in common "per-layer" functionality [TAG_PER_LAYER]
    ggml_tensor * build_inp_per_layer();
    ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);
};
```

- [ ] **Step 2: Commit**

```bash
git add src/models/models.h
git commit -m "feat: declare llm_build_gemma4_iswa class in models.h"
```

---

### Task 6: Wire Up llama-model.cpp — 5 Registration Points

**Files:**
- Modify: `src/llama-model.cpp`

This is the largest task — 5 separate `case LLM_ARCH_GEMMA4:` blocks need to be added.

- [ ] **Step 1: Add hparams loading (after GEMMA3N case, ~line 1266)**

After the `case LLM_ARCH_GEMMA3N:` block's closing `break;`, add:

```cpp
        case LLM_ARCH_GEMMA4:
            {
                hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

                uint32_t n_kv_shared_layers = 0;
                ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);

                hparams.n_layer_kv_from_start = hparams.n_layer - (int32_t)n_kv_shared_layers;
                hparams.f_attention_scale     = 1.0f;

                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);
                ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
                ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

                switch (hparams.n_layer) {
                    case 35: type = LLM_TYPE_E2B; break;
                    case 42: type = LLM_TYPE_E4B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
```

**Important:** `swa_layers` is loaded as a per-layer boolean array directly from GGUF, NOT via `set_swa_pattern()`. The 26B GGUF has `attention.sliding_window_pattern` as a 30-element boolean array. Verify that `get_key_or_arr` for `swa_layers` already supports bool arrays: `git show origin/master:src/llama-model.cpp | grep -B2 -A5 "swa_layers"`

- [ ] **Step 2: Add tensor loading (after GEMMA3N tensor case, ~line 4234)**

```cpp
            case LLM_ARCH_GEMMA4:
                {
                    const uint32_t n_embd_per_layer = hparams.n_embd_per_layer;
                    const int64_t  n_ff_exp         = hparams.n_ff_exp;

                    if (n_embd_head_k != n_embd_head_v) {
                        throw std::runtime_error("Gemma 4 requires n_embd_head_k == n_embd_head_v");
                    }
                    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
                        throw std::runtime_error("Gemma 4 requires n_embd_head_k_swa == n_embd_head_v_swa");
                    }

                    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    if (n_embd_per_layer > 0) {
                        per_layer_tok_embd   = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),    {n_embd_per_layer * n_layer, n_vocab}, 0);
                        per_layer_model_proj = create_tensor(tn(LLM_TENSOR_PER_LAYER_MODEL_PROJ, "weight", 0), {n_embd, n_embd_per_layer * n_layer}, 0);
                        per_layer_proj_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ_NORM,  "weight", 0), {n_embd_per_layer}, 0);
                    }

                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

                    int rope_freqs_flag = 0;

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];
                        const int64_t n_head      = hparams.n_head(i);
                        const int64_t n_embd_head = hparams.n_embd_head_k(i);
                        const int64_t n_embd_k    = hparams.n_embd_k_gqa(i);
                        const int64_t n_embd_v    = hparams.n_embd_v_gqa(i);
                        const int     kv_flags    = hparams.has_kv(i) ? 0 : TENSOR_NOT_REQUIRED;

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k}, kv_flags);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v}, TENSOR_NOT_REQUIRED);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head * n_head, n_embd}, 0);

                        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, 0);
                        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head}, kv_flags);
                        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

                        layer.out_scale = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1u}, TENSOR_NOT_REQUIRED);

                        if (!hparams.is_swa(i)) {
                            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head/2}, rope_freqs_flag);
                            rope_freqs_flag = TENSOR_DUPLICATED;
                        }

                        int64_t n_ff_cur = hparams.n_ff(i);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff_cur}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff_cur}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_cur, n_embd}, 0);
                        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

                        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
                        bool has_expert = layer.ffn_gate_inp != nullptr;

                        if (has_expert) {
                            layer.ffn_gate_inp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "scale", i), {n_embd}, 0);

                            layer.ffn_pre_norm_2  = create_tensor(tn(LLM_TENSOR_FFN_PRE_NORM_2,  "weight", i), {n_embd}, 0);
                            layer.ffn_post_norm_1 = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_1, "weight", i), {n_embd}, 0);
                            layer.ffn_post_norm_2 = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_2, "weight", i), {n_embd}, 0);

                            layer.ffn_gate_up_exps  = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS,  "weight", i), {n_embd, n_ff_exp * 2, n_expert}, 0);
                            layer.ffn_down_exps     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,     "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
                        }

                        if (n_embd_per_layer > 0) {
                            layer.per_layer_inp_gate   = create_tensor(tn(LLM_TENSOR_PER_LAYER_INP_GATE,  "weight", i), {n_embd, n_embd_per_layer}, 0);
                            layer.per_layer_proj       = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ,      "weight", i), {n_embd_per_layer, n_embd}, 0);
                            layer.per_layer_post_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_POST_NORM, "weight", i), {n_embd}, 0);
                        }
                    }
                } break;
```

- [ ] **Step 3: Add KV reuse callback (after GEMMA3N KV reuse, ~line 8257)**

At the `if (arch == LLM_ARCH_GEMMA3N)` check (line 8249), extend it:

```cpp
                    if (arch == LLM_ARCH_GEMMA3N || arch == LLM_ARCH_GEMMA4) {
```

This enables KV cache layer reuse for layers that share KV (n_layer_kv_from_start).

- [ ] **Step 4: Add graph builder dispatch (after GEMMA3N case, ~line 8501)**

After:
```cpp
        case LLM_ARCH_GEMMA3N:
            {
                llm = std::make_unique<llm_build_gemma3n_iswa>(*this, params);
            } break;
```

Add:
```cpp
        case LLM_ARCH_GEMMA4:
            {
                llm = std::make_unique<llm_build_gemma4_iswa>(*this, params);
            } break;
```

- [ ] **Step 5: Add ROPE type (after GEMMA3N in rope_type, ~line 9021)**

After `case LLM_ARCH_GEMMA3N:` in the rope_type function, add:
```cpp
        case LLM_ARCH_GEMMA4:
```

(Falls through to the same `return LLAMA_ROPE_TYPE_NEOX;`)

- [ ] **Step 6: Verify all 5 registration points**

Run: `grep -n "LLM_ARCH_GEMMA4" src/llama-model.cpp`

Expected: 5 or 6 matches (hparams, tensors, KV reuse, graph builder, rope type, possibly more).

- [ ] **Step 7: Commit**

```bash
git add src/llama-model.cpp
git commit -m "feat: wire Gemma 4 into llama-model — hparams, tensors, KV reuse, graph, rope"
```

---

### Task 7: Add to CMake Build

**Files:**
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Add gemma4-iswa.cpp to sources**

After line 75 (`models/gemma3n-iswa.cpp`), add:
```
            models/gemma4-iswa.cpp
```

- [ ] **Step 2: Stage the untracked gemma4-iswa.cpp**

```bash
git add src/models/gemma4-iswa.cpp
```

- [ ] **Step 3: Commit**

```bash
git add src/CMakeLists.txt src/models/gemma4-iswa.cpp
git commit -m "feat: add gemma4-iswa.cpp to build"
```

---

### Task 8: Build and Fix Compilation Errors

**Files:**
- Potentially any file from Tasks 1-7 if there are issues

- [ ] **Step 1: Clean build**

```bash
cd /Users/adi/PAIA_V1/llama-cpp-turboquant/build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
```

- [ ] **Step 2: Build core targets**

```bash
make -j$(sysctl -n hw.ncpu) llama-cli llama-quantize llama-server 2>&1 | tail -30
```

Expected: successful compilation. If errors:
- Missing field errors → check Task 3 field names
- Missing enum errors → check Task 1 enum names
- Undeclared function → check Task 5 class declaration
- Linker errors → check Task 7 CMake

- [ ] **Step 3: Fix any `swa_layers` type mismatch**

The upstream `get_key_or_arr` for `swa_layers` expects `std::array<uint32_t, LLAMA_MAX_LAYERS>` but the GGUF stores booleans. Verify upstream's `swa_layers` type in hparams and match it. If needed, check: `git show origin/master:src/llama-hparams.h | grep swa_layers`

- [ ] **Step 4: Fix any `n_embd_head_k(il)` per-layer accessor issues**

Gemma 4 has different head dims per layer (256 SWA, 512 global). The accessor must return different values based on `is_swa(il)`. Check upstream implementation and port if needed.

- [ ] **Step 5: Commit fixes**

```bash
git add -u
git commit -m "fix: resolve Gemma 4 compilation issues"
```

---

### Task 9: Test Model Loading

- [ ] **Step 1: Test loading the GGUF**

```bash
./build/bin/llama-cli \
  -m /Users/adi/noor/models/google_gemma-4-26B-A4B-it-IQ3_M.gguf \
  -p "Hello" \
  -n 1 \
  --no-warmup 2>&1 | head -50
```

Expected: model loads without "unknown model architecture: 'gemma4'" error. May still fail at inference — that's Task 10.

- [ ] **Step 2: Check tensor loading output**

Look for warnings like "unused tensor" or "missing tensor" in stderr. All 658 tensors should load.

- [ ] **Step 3: If tensor shape mismatch, debug**

Compare the expected shapes (from GGUF metadata) against what the code computes. Common issues:
- `n_embd_head_k(i)` returning wrong value for SWA vs global layers
- `n_head(i)` not being per-layer
- Expert tensor dimensions wrong

---

### Task 10: Test Inference

- [ ] **Step 1: Run a short generation test**

```bash
./build/bin/llama-cli \
  -m /Users/adi/noor/models/google_gemma-4-26B-A4B-it-IQ3_M.gguf \
  -p "What is the capital of Bangladesh?" \
  -n 50 \
  --temp 0.7 2>&1
```

- [ ] **Step 2: Verify output quality**

Expected: coherent text. If garbage output, check:
- RoPE frequencies (freq_base 10000 for SWA, 1000000 for global)
- Attention scale (should be 1.0)
- V projection reuse (global layers: V = K)
- Logit softcapping (30.0)

- [ ] **Step 3: Check speed**

Look for the generation speed in the output. Compare with Ollama baseline (46 t/s generation).

- [ ] **Step 4: Test with server**

```bash
./build/bin/llama-server \
  -m /Users/adi/noor/models/google_gemma-4-26B-A4B-it-IQ3_M.gguf \
  --port 8081 2>&1 &
```

Quick curl test:
```bash
curl -s http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'
```

---

### Task 11: TQ3_0 Quantization Test (Optional — if time permits)

- [ ] **Step 1: Check if imatrix exists or generate one**

```bash
ls /Users/adi/noor/models/*imatrix* 2>/dev/null
```

If not, skip imatrix for first test.

- [ ] **Step 2: Quantize to TQ3_0**

```bash
./build/bin/llama-quantize \
  /Users/adi/noor/models/google_gemma-4-26B-A4B-it-IQ3_M.gguf \
  /Users/adi/noor/models/gemma-4-26B-A4B-it-TQ3_0.gguf \
  TQ3_0 2>&1 | tail -20
```

Note: This requantizes from IQ3_M to TQ3_0. For best quality, start from BF16/F16. But this tests the pipeline.

- [ ] **Step 3: Test TQ3_0 inference**

```bash
./build/bin/llama-cli \
  -m /Users/adi/noor/models/gemma-4-26B-A4B-it-TQ3_0.gguf \
  -p "What is the capital of Bangladesh?" \
  -n 50 \
  --temp 0.7 2>&1
```

- [ ] **Step 4: Test TQ3_S KV cache**

```bash
./build/bin/llama-cli \
  -m /Users/adi/noor/models/google_gemma-4-26B-A4B-it-IQ3_M.gguf \
  -p "What is the capital of Bangladesh?" \
  -n 50 \
  -ctk tq3_s -ctv tq3_s 2>&1
```

---

## Risk Notes

1. **Per-layer head dimensions**: Gemma 4's most complex feature — SWA layers have 256 head_dim / 8 KV heads, global layers have 512 head_dim / 2 KV heads. The upstream `n_embd_head_k(il)` accessor handles this, but our fork may not have it yet. This is the most likely source of compilation or runtime errors.

2. **swa_layers as boolean array**: Unlike Gemma 3 (which uses `set_swa_pattern(period)`), Gemma 4 loads the SWA pattern as an explicit per-layer boolean array from GGUF. The `get_key_or_arr` function needs to handle this format.

3. **KV reuse for shared layers**: The 26B model has `shared_kv_layers = 0` (no sharing), so this is not exercised initially. But E2B/E4B models DO share KV across layers.

4. **Requantization quality**: Quantizing IQ3_M → TQ3_0 degrades quality. For production, download the BF16 SafeTensors and convert fresh.
