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
