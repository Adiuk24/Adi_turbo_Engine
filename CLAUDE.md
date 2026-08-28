# AdiTurbo Engine (llama.cpp fork)

This is **our fork**, not upstream llama.cpp. Product: moe-stream (SSD expert
streaming) + KV persistence + blend. See README.md for the feature story and
measured numbers.

## Git rules (hard)

- Push ONLY to `aditurbo` (Adiuk24/Adi_turbo_Engine), branch `moe-stream-upstream`.
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

## Known state / traps

- `qwen4exp` (Qwen3.8-Flash-Next 180B): upstream support needs newer kv-cache
  internals (`llama_kv_cell_ext.tok`, `for_each_token_in`) — a straight
  cherry-pick does NOT build on this branch. Model is ready at
  `/Volumes/MODELS/qwen38flash/Qwen3.8-Flash-Next-UD-Q3_K_XL-merged.gguf` (84G).
- Noor GGUFs: `o_proj` + shexp MUST stay F16; `tokenizer.ggml.pre` must be
  `deepseek-v3` for Bangla; shexp f32 in the converter for CPU.
- Fused Metal GDN kernel scale at `ggml-metal.metal:2750,2869` is dormant and
  wrong (1/sqrt(S_v), should be S_k) — don't enable without fixing.
- Work style: use the `/deploy-loop` skill — cheap agents build, you validate.
