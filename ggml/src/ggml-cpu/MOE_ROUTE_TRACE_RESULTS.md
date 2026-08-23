# MoE expert-routing predictability probe — results

Instrumentation: `GGML_MOE_STREAM_TRACE=<path>` in `moe-stream.c` /
`ggml-cpu.c` (see `ggml-moe-stream.h` for the call contract). Logs one CSV
row per (token, layer): the top-8 selected expert IDs.

Model: Qwen3-30B-A3B-Q3_K_M, CPU-only, 48 layers, 128 experts/layer, top-8.

## Methodology correction (2026-08-23, same day as the first pass)

The first version of this analysis built the layer→layer and per-layer
frequency tables from the **same rows they were scored against** — the
co-occurrence table was fit on all 4 prompts' data, then evaluated on that
same data. That is train-on-test, and it made the numbers optimistic. This
file replaces those numbers with a proper held-out evaluation:

1. **Leave-one-prompt-out (LOO)** over the original 4 prompts: fit the
   table on 3, score on the 4th, rotate, average.
2. **Fresh prompts**: fit the table on all 4 original prompts, score on 3
   brand-new prompts of different character (arithmetic, dialogue, list
   formatting) the table never saw, ~41-46 decode steps each.

Same free per-layer frequency baseline, computed on the held-out split each
time (never the eval split).

## Results

Recall@N = fraction of the true top-8 experts covered by the top-N predicted
candidates, averaged over all (token, layer) pairs in the eval set.

| split | method | recall@8 | recall@16 | recall@32 |
|---|---|---|---|---|
| train-on-test (biased, for reference) | baseline | 0.27 | 0.43 | 0.64 |
| train-on-test (biased, for reference) | prev-layer | 0.56 | 0.78 | 0.93 |
| **LOO mean, 4 folds** | **baseline** | **0.18** | **0.30** | **0.48** |
| **LOO mean, 4 folds** | **prev-layer** | **0.37** | **0.55** | **0.70** |
| LOO mean, 4 folds | prev-token | 0.44 | 0.53 | 0.66 |
| **fresh prompts (3, unseen)** | **baseline** | **0.19** | **0.31** | **0.51** |
| **fresh prompts (3, unseen)** | **prev-layer** | **0.41** | **0.59** | **0.74** |
| fresh prompts (3, unseen) | prev-token | 0.44 | 0.52 | 0.65 |

The held-out numbers (LOO and fresh-prompt) agree closely with each other
and both sit well below the train-on-test numbers. That agreement is the
important part: it means the LOO estimate is not itself an artifact — the
signal really does transfer to prompts the table never saw.

## Per-fold spread (stability across prompt type)

LOO fold-to-fold spread (max − min recall across the 4 held-out folds):

| method | @8 | @16 | @32 |
|---|---|---|---|
| baseline | 0.07 | 0.11 | 0.15 |
| prev-layer | 0.11 | 0.19 | 0.23 |
| prev-token | 0.17 | 0.19 | 0.14 |

Not stable. Bangla is the outlier: prev-layer is its *worst* fold
(0.32/0.45/0.58) while prev-token is its *best* (0.56/0.65/0.73) — the
opposite ranking from English/code/factual text, where prev-layer beats
prev-token. A single static table trained on English-heavy data would
under-serve non-Latin-script workloads; which signal (cross-layer vs.
cross-token) dominates depends on prompt/tokenization character, not just
on the model.

## Decision (per the stated rule: pays only if held-out recall@16-32 > ~80%
## and clearly beats baseline)

Margin over baseline is real and held-out (prev-layer ≈ 2x baseline at
every N, both in LOO and on fresh prompts) — this is **not** prompt-specific
memorization, the cross-layer pattern genuinely generalizes.

But recall@16 (0.55-0.59) and recall@32 (0.70-0.74) held-out fall well short
of the 80% bar needed to hide I/O. At n_slots=16-32, a prefetcher built on
this signal would still miss roughly a quarter to a third of the real next
layer's experts and fall back to blocking pread on the miss — most of the
latency win evaporates.

**TRANSFERS** (the layer→layer signal is real, not memorized) **but does
not clear the prefetch-worthiness bar** at the required recall level.
Verdict: do not build the prefetcher on this signal as-is. If revisited,
the Bangla-vs-Latin split above is the first thing to chase — a
signal-selection or per-workload table might close some of the gap that a
single static table can't.

## Generalization test (2026-08-23): does this hold on other MoE architectures?

Same instrumentation (`GGML_MOE_STREAM_TRACE`), same methodology (LOO
held-out over 7 mixed prompts: EN question, Bangla, code, arithmetic,
dialogue, factual, list), applied to two more models to test whether
predictability depends on expert granularity.

### Qwen3-Next-80B-A3B-Instruct-Q4_K_M

Config (from GGUF metadata): `qwen3next` arch, 48 layers, **512 experts/layer,
top-10**, plus a shared expert (`expert_shared_feed_forward_length=512`).
CPU-only via `GGML_MOE_STREAM=1 GGML_MOE_STREAM_SLOTS=64`, `-ngl 0`, on
branch `moe-route-trace` (instrumentation already present, no port needed —
same engine, different GGUF). ~1.5 tok/s decode; 35 decode steps/prompt.

Entropy 7.58 bits/layer-token, mean **361/512 experts active per layer
(70%)** — even more diffuse than Qwen3-30B in absolute expert count, similar
in fraction-of-total.

| split | method | recall@10 | recall@20 | recall@40 |
|---|---|---|---|---|
| train-on-test (biased) | baseline | 0.200 | 0.310 | 0.457 |
| train-on-test (biased) | prev-layer | 0.542 | 0.740 | 0.899 |
| **LOO, 7 folds** | **baseline** | **0.073** | **0.127** | **0.201** |
| **LOO, 7 folds** | **prev-layer** | **0.187** | **0.280** | **0.387** |
| LOO, 7 folds | prev-token | 0.174 | 0.254 | 0.355 |

Held-out recall is markedly *worse* than Qwen3-30B's (recall@2k: 0.28 vs
0.55-0.59; recall@4k: 0.39 vs 0.70-0.74) despite prev-layer still beating
baseline by ~2x at every N — same qualitative signal, weaker in absolute
terms. Bangla anomaly did **not** reproduce here: prev-layer beat prev-token
on every single fold, including Bangla (@20: prev-layer=0.234 vs
prev-token=0.229), unlike Qwen3-30B where Bangla flipped the ranking.

**Verdict: further below the bar than Qwen3-30B.** More experts (512 vs
128), same top-k regime, more diffuse routing, less predictable.

### Noor-edge-v2-f16 (Bangla-native, GDN-hybrid MoE)

Full detail + config + build notes: `llama-cpp-turboquant` repo, branch
`moe-route-trace-noor` (cherry-picked the 3 trace-instrumentation commits
from this branch onto `noor-arch`; required one small filter fix — Noor has
no `ffn_gate_exps` tensor, MoE FFN is up/down only, no gate — see that
branch's commit log). Config: **8 experts/layer, top-2**, plus a shared
expert, 24 layers, tiny (2.7GB f16).

Entropy only 2.28 bits/layer-token, mean **7.7/8 experts active per layer
(96%)** — with only 8 total experts almost everything gets touched anyway.

| split (LOO, 7 folds) | method | recall@2 | recall@4 | recall@8 |
|---|---|---|---|---|
| baseline | | 0.527 | 0.834 | 0.999 |
| prev-layer | | 0.666 | 0.875 | 0.996 |
| prev-token | | 0.698 | 0.873 | 0.997 |

Caveat: 3 of 7 prompts (en_q, code, factual — raw completion mode, no chat
template) collapsed into a degenerate repeated-token loop (Noor is
Bangla-first; English/code raw completion is weak and falls into an
attractor state even with `--repeat-penalty 1.3`). Those folds trivially
inflate recall (same token repeated -> same experts repeated) and are not
clean signal. The 4 clean folds (bangla, arith, dialogue, list) are the
trustworthy ones; Bangla anomaly **partially reproduces** there: prev-token
beats prev-layer on bangla (0.416 vs 0.376 @2) and dialogue (0.455 vs 0.385
@2), same direction as Qwen3-30B, though the margin is smaller and doesn't
hold at every N.

recall@8 = ~0.996-0.999 is **not a real win** — with only 8 total experts,
@8 means "predict the entire expert set," which is vacuous. The honest
metric is @4 (2x top-k): 0.834-0.875, close to the ~85% bar, but baseline
alone already gets 0.834 — the apparent "predictability" is mostly a
small-total-space ceiling effect, not strong structural signal. And
practically moot: the whole model is 2.7GB and already fits trivially in
RAM, so there is no I/O to hide — prefetch-by-prediction has no target here.

## Cross-model comparison

| Model | n_experts | top_k | shared exp | layers | experts active/layer | entropy (bits) | baseline recall@k/2k/4k (LOO) | prev-layer recall@k/2k/4k (LOO) |
|---|---|---|---|---|---|---|---|---|
| Noor-edge-v2 | 8 | 2 | yes | 24 | 7.7 (96%) | 2.28 | 0.527 / 0.834 / 0.999 | 0.666 / 0.875 / 0.996 |
| Qwen3-30B-A3B | 128 | 8 | no | 48 | ~97-120 (76-94%) | ~6-7 | 0.18 / 0.30 / 0.48 | 0.37 / 0.55 / 0.70 |
| Qwen3-Next-80B-A3B | 512 | 10 | yes | 48 | 361 (70%) | 7.58 | 0.073 / 0.127 / 0.201 | 0.187 / 0.280 / 0.387 |

**Predictability correlates with expert granularity, monotonically, in the
expected direction: fewer total experts -> higher recall.** Going from 128
to 512 experts (Qwen3 family, same top-k order of magnitude) roughly halves
held-out prev-layer recall@2k/4k. Going down to 8 experts (Noor) pushes
recall@4k near-ceiling — but that's substantially a small-total-space
artifact (baseline is already near-ceiling too), not proof of strong
cross-layer structure, and the model is too small to need prefetching in the
first place.

**Overall verdict across all 3 models: prefetch-by-prediction does not clear
the bar anywhere it would matter.** It's unnecessary where the numbers look
best (Noor — too small to need it) and insufficient where it would actually
save I/O (Qwen3-30B, Qwen3-Next-80B — both diffuse, both well below 80%
held-out recall@2k-4k, and Next is worse than 30B, not better). Do not build
the prefetcher on this signal for any tested model class.
