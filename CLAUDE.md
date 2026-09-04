# AdiTurbo Engine (llama.cpp fork) — STANDALONE REPO

**Location:** `~/aditurbo-engine`   **Canonical branch:** `engine-main`

Moved out of `~/PAIA_V1` on 2026-09-04 and consolidated from five divergent
worktrees into ONE branch. PAIA_V1 is a separate project; the engine is no
longer inside it. `~/PAIA_V1/llama-cpp-turboquant` is retired and deletable.

Installed to ADIOS at `/Volumes/MODELS/ADIOS/bin/` and `bin-noor/`
(build 273, commit 9efe4786d). Previous engine preserved at
`/Volumes/MODELS/ADIOS/bin.backup-20260904/`; the legacy Noor parity ORACLE is
preserved at `/Volumes/MODELS/ADIOS/bin-noor.oracle-b8667/` (version 8667,
b784cc570) — do not delete it, it is the independent verification binary.

Validation gates + baselines: `~/engine_gates/`.

This is **our fork**, not upstream llama.cpp. Product: moe-stream (SSD expert
streaming) + KV persistence + blend. See README.md for the feature story and
measured numbers.

## Git rules (hard)

- Push ONLY to `aditurbo` (Adiuk24/Adi_turbo_Engine), branch `engine-main`.
- NEVER push to `origin`/`upstream` (ggml-org) or `jjj`. Never open upstream
  PRs, issues, or comments — no external communication of any kind.
- Commit agent/subagent work IMMEDIATELY before touching their files
  (`git checkout <file>` once wiped 529 uncommitted lines).
- Commit messages carry MEASURED results, including negative ones.

## Build & verify

```bash
cmake --build build --target llama-server llama-cli -j8
```

- Error check: `grep -cE "error:"` — NOT `grep -c " error "` (that pattern
  gave a false "0 errors" on a broken build once).
- Never trust a builder agent's "syntax-only clean" — build yourself.
- Metal + CPU both matter; CPU-only paths (Android, Windows) are shipping lanes.

## Ownership map

- `ggml/src/ggml-cpu/moe-stream.c` — the streaming engine (slot pool, pread,
  LFU/pinlru, prefetch). All knobs in `ggml/include/ggml-moe-stream.h`.
- `tools/server/server-context.cpp` — blend (`try_blend_prefill`, ~line 2746;
  `POST /blend/adopt`) and slot save/restore.
- Everything else is upstream code — keep diffs against it minimal to keep
  rebases cheap.

## Canonical launch (streamed giant)

```bash
GGML_MOE_STREAM=1 GGML_MOE_STREAM_ALLOC_ORDER=1 \
GGML_MOE_STREAM_SLOTS=16 GGML_MOE_STREAM_CHUNKS=4 \
./build/bin/llama-server -m <single-file>.gguf \
  -ngl 99 -cmoe -t 8 -np 1 -c 1024 -ub 256 -b 1024 \
  --poll 0 --no-warmup --jinja --reasoning off \
  [--override-kv <arch>.expert_used_count=int:3]
```

- GGUF MUST be a single file (`llama-gguf-split --merge` first).
- Blend needs `--kv-unified -np 1`. KV persist: `--slot-save-path` +
  `/slots/0?action=save|restore`.
- Models live on the SSD (`/Volumes/MODELS/...`), never /tmp. One giant model
  at a time on the 24 GB Mac — two at once OOM-killed the box before.
- WARN the user before killing a running server or starting a heavy local run.

## Validation gates (run these, don't eyeball)

1. **Parity** (all knobs default-off): temp-0, identical seed, knob-off vs
   baseline must be byte-identical. Strip `[moe-stream]` stats lines AND the
   blank lines they leave, or you get false failures (happened twice).
   Raw llama-cli parity is INVALID for Noor — use `llamacpp_safe_prompt` /
   `test_engine_parity.py`.
2. **Quality**: plant verifiable facts in the prompt, check they survive.
3. **Perf**: adjacent A/B, ABBA order, note swap/cache state. First run after
   big file I/O is a cold-cache outlier. USB decode is bus-bound — most knobs
   measure neutral there; report neutral as neutral.
4. **First-divergence regression** (configs where exact parity isn't expected,
   e.g. k-dial or mixed-precision): record WHERE output first diverges from
   the reference (battle harness prints the byte offset). The divergence
   point moving EARLIER after a change is a regression needing explanation,
   even if the output "still looks fine".
5. **Keep the logs**: save gate/bench output to /tmp or the run dir and put
   the path (plus the numbers) in the commit message — a claim without its
   log is a builder report, not a result.

## Upstream drift policy (two-lane, adopted from CKE's practice)

- The fork's merge-base with upstream is the PINNED numerical oracle. A
  periodic check may fetch upstream HEAD, report commit delta, and test-apply
  our patch set — but a drift failure is EVIDENCE, never authorization to
  widen a tolerance or move the pin. Moving the pin is its own reviewed
  change, gated on identical-hardware parity runs.

## Backlog (validated ideas, not yet built — from the 2026-08-31 CKE study)

- Logit-level blend gate: token-identity parity passed in a case where logit
  cosine was 0.866 (CKE's batched-prefill blocker) — add cosine/RMSE on
  final-token logits at >=1k ctx to the blend gate.
- Sparse cb_eval layer hook (public ggml API, no fork patch): per-layer
  wall-time attribution separating bus stall / Metal / CPU expert GEMM, and a
  first-divergent-LAYER oracle. Sparse l_out-N boundaries only — hooking
  every node splits the graph and corrupts the measurement. Medians over
  many tokens, never single-token totals.
- Slot-pool telemetry: machine-readable eviction reason per victim +
  exposed-wait-at-first-use vs hidden-behind-compute I/O accounting.
- NVMe rig only: raise effective queue depth by batching all k routed-expert
  reads per token (USB measured link-limited — QD is neutral there).
- GGUF EOF-trailer sidecar (advisory-only) for slab index / routing priors /
  k-dial defaults — appends without rewriting a 200GB file.
- Gate hardening (from CKE's verification notes, all cheap): (a) two-control
  repeatability — run the baseline TWICE before trusting any parity red
  (their 27B Q4_K_M was nondeterministic at 24 threads, first visible at
  token 76); (b) canary neutrality — canaries-on vs off must be
  byte-identical (fold into the pending fault-injection session); (c) lengthen
  one greedy replay gate to 128+ tokens ("diverges coherently at token 40-50"
  at cosine 0.9989 passes every shorter check); (d) thread-sweep the parity
  assertion (t=1 vs t=8), not just the default; (e) replay gates assert the
  RENDERED PROMPT STRING too — a numerically perfect model + wrong chat
  template = garbage that byte-parity can't see; (f) three-way perf verdict:
  keep / reject-if-wrong / research-only-if-no-model-level-gain, and keep a
  permanent rejected-experiments list with reasons.
- Noor multimodal (when OCR/vision lands): encoder/tokenizer PREFIX parity
  before decode parity; a deliberately tiny/degenerate input in the corpus
  (their 3x3-grid case scored 0.799 cosine where full-size scored 0.999);
  preprocessing (resize/normalize) as its own pinned unit test; hash-pinned
  5-sample gate per change, full corpus nightly; prefill-vs-decode
  single-token equivalence gate (GDN has two code paths for the same math).
- Checked and already correct (do NOT redo): last-only prefill logits (stock
  + blend), hot-path getenv caching, dynamic chunk-claiming in mul_mat/GDN,
  F_NOCACHE knob, contiguous slabs, head-parallel GDN, alternating-order A/B.

## Known state / traps

- `qwen4exp` (Qwen3.8-Flash-Next 180B): WORKS as of 1d57fca084 (needed upstream
  cherry-picks 925e117994 token-tracking + fac889fb38 TENSOR_READ_LAZY on top
  of the 6c84c7d5d8 model commit). Smoke-verified 1.3 tok/s streamed off USB.
  Model: `/Volumes/MODELS/qwen38flash/Qwen3.8-Flash-Next-UD-Q3_K_XL-merged.gguf` (84G).
- Noor arch is NOT in this fork (separate noor-arch tree) — "unknown model
  architecture: 'noor'" here is expected, not a regression.
- Noor GGUFs: `o_proj` + shexp MUST stay F16; `tokenizer.ggml.pre` must be
  `deepseek-v3` for Bangla; shexp f32 in the converter for CPU.
- Fused Metal GDN kernel scale at `ggml-metal.metal:2750,2869` is dormant and
  wrong (1/sqrt(S_v), should be S_k) — don't enable without fixing.
- Work style: use the `/deploy-loop` skill — cheap agents build, you validate.
