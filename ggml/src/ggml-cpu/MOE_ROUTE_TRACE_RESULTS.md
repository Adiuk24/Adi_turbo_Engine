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

## Noor-edge-v2-f16 generalization run (2026-08-23, this repo/branch)

Cherry-picked the 3 trace commits from `llama-cpp-moestream:moe-route-trace`
onto `noor-arch` (clean, no conflicts, ~15 min). One fix needed: Noor's MoE
FFN has no `ffn_gate_exps` tensor (up/down only, no gate projection) — the
trace hook's name filter was `ffn_gate_exps`-only and never fired. Added
`ffn_up_exps` as a fallback match (see moe-stream.c, same commit range).

Command: `-ot "exps=CPU" -ngl 999` (keeps the GDN op on Metal, where it
works; forcing everything to `-ngl 0` breaks Noor's GDN CPU kernel and
produces degenerate output — a known issue, not something this probe should
paper over). `--repeat-penalty 1.3` did not fix it: 3 of 7 raw-completion
prompts (en_q, code, factual) still collapsed into a repeated-token
attractor state (Noor is Bangla-first; English/code without a chat template
is weak). Those 3 folds are excluded from the headline read.

Config: 8 experts/layer, top-2, shared expert, 24 layers, entropy 2.28
bits/layer-token, 7.7/8 experts active per layer (96% — small space, almost
everything gets touched regardless of prompt).

| split (LOO, 7 folds, all prompts) | method | recall@2 | recall@4 | recall@8 |
|---|---|---|---|---|
| baseline | | 0.527 | 0.834 | 0.999 |
| prev-layer | | 0.666 | 0.875 | 0.996 |
| prev-token | | 0.698 | 0.873 | 0.997 |

recall@8 is vacuous here (8 = total expert count, "predicting" the whole
set). The real number is recall@4 (2x top-k): 0.83-0.88, close to the ~85%
bar, but baseline alone is already at 0.834 — mostly a small-total-space
ceiling effect, not strong evidence of cross-layer/cross-token structure.
And moot in practice: the whole model is 2.7GB, already fits in RAM, there
is no I/O latency to hide with a prefetcher.

Bangla-vs-Latin anomaly (prev-token beating prev-layer, opposite of the
usual ranking) from the Qwen3-30B run partially reproduces on the 4 clean
folds: bangla (prev-token 0.416 vs prev-layer 0.376 @2) and dialogue
(0.455 vs 0.385 @2) both flip the usual ranking; arith and list do not
consistently. Weaker and less clean than the Qwen3-30B result, but the
same-direction flip on Bangla, on a completely different architecture (GDN
hybrid, 8 experts vs. 128), is suggestive that it's a genuine
script/tokenization effect and not an artifact of one model family.

Full cross-model table (Noor + Qwen3-30B + Qwen3-Next-80B) and final
verdict: see `llama-cpp-moestream:moe-route-trace`,
`ggml/src/ggml-cpu/MOE_ROUTE_TRACE_RESULTS.md`, "Cross-model comparison"
section (same day). Short version: predictability correlates monotonically
with expert count, in the expected direction, but no tested model clears
the prefetch-worthiness bar in a way that matters — do not build the
prefetcher on this signal.
