#pragma once

// moe-stream: universal SSD expert-streaming for MoE models.
//
// Slot-pool expert streamer for the CPU backend. When enabled (GGML_MOE_STREAM=1),
// a fixed number of expert "slots" per MoE weight tensor are kept resident in
// page-aligned host memory; experts are pread() from the GGUF file on demand and
// evicted LFU-with-recency-tiebreak. The mmap'd pages backing the tensor's
// mapped experts are never touched on the streamed path -- that is the memory win.
//
// A single mul_mat_id call can route to more distinct experts than the slot
// pool holds (a multi-token prefill ubatch, or a warmup pass touching every
// expert). The caller (ggml-cpu.c) handles this by splitting the call's
// active experts into consecutive groups of at most n_slots and running
// plan+fetch+compute once per group, with barriers between groups so a slot
// is never reused while another thread is still computing from it. This
// keeps every call memory-bounded to n_slots resident experts, decode or
// prefill, warmup or not.
//
// Env vars (no CLI plumbing by design):
//   GGML_MOE_STREAM=1            enable streaming
//   GGML_MOE_STREAM_SLOTS=<n>    slots per tensor (default 16, clamped to n_expert)
//   GGML_MOE_STREAM_STATS=1      print one hits/misses/bytes_read line per tensor at exit
//   GGML_MOE_STREAM_HITFIRST=1   overlap hit-expert compute with the miss-expert fetch
//                                fan-out on the single-group decode fast path (default off)
//   GGML_MOE_STREAM_FETCH_THREADS=<n>  threads dedicated to the fetch fan-out under
//                                HITFIRST; the rest compute hits concurrently (default 2)
//   GGML_MOE_STREAM_PREFETCH=1   enable a background IO thread that speculatively
//                                pre-reads likely-next experts into extra landing slots
//                                during the idle bus time between fetch phases (default off,
//                                zero behavior change when unset -- byte-identical output)
//   GGML_MOE_STREAM_PREFETCH_SLOTS=<n>  extra landing slots reserved per tensor for
//                                prefetch, on top of GGML_MOE_STREAM_SLOTS (default 4)
//   GGML_MOE_STREAM_PREFETCH_DEPTH=<n>  how many tensors ahead (registration order,
//                                which tracks layer execution order) to predict likely
//                                experts for on every plan() call (default 2)
//   GGML_MOE_STREAM_PREFETCH_MB=<n>     total prefetch-arena RAM budget across ALL
//                                tensors, in MiB (default 2048). Tensors are granted
//                                their full GGML_MOE_STREAM_PREFETCH_SLOTS allotment,
//                                all-or-nothing, until the budget runs out; later
//                                tensors then get zero prefetch slots.

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Cached getenv("GGML_MOE_STREAM") == "1".
GGML_API bool ggml_moe_stream_enabled(void);

// Register a 3-D expert tensor (ne[2] == n_expert) for streaming. Opens `path`
// O_RDONLY (fds are deduped by path across tensors from the same file).
// `file_offs` is the byte offset of expert 0's data within `path`; experts are
// laid out contiguously, each `t->nb[2]` bytes wide (matches the loader's
// llama_tensor_weight::offs convention). No-op if `t` or `path` is NULL, or if
// the registry is full.
GGML_API void ggml_moe_stream_register(struct ggml_tensor * t, const char * path, size_t file_offs);

GGML_API bool ggml_moe_stream_is_registered(const struct ggml_tensor * t);

// Single-threaded planning step: call from thread 0 only, after building the
// per-expert row-routing counts for one mul_mat_id invocation. Marks hits,
// assigns eviction slots for misses (reserving them so two misses in the same
// call never collide), and bumps LFU/recency bookkeeping.
GGML_API void ggml_moe_stream_plan(const struct ggml_tensor * t, const int64_t * row_counts, int n_expert);

// Slot count for this tensor (<= n_expert, see GGML_MOE_STREAM_SLOTS). Callers
// use this to split a call's active experts into groups of at most n_slots
// when the call needs more distinct experts than the pool holds.
GGML_API int ggml_moe_stream_n_slots(const struct ggml_tensor * t);

// Records that this call needed more than one group (stats only, call once
// from thread 0 before the group loop starts).
GGML_API void ggml_moe_stream_mark_chunked(const struct ggml_tensor * t);

// Critical-path phase profiler. GGML_MOE_STREAM_PROF=1 dumps, at exit, the
// wall-clock split of the streamed mul_mat_id path:
//
//     phase 0 = plan      (choose which experts miss)
//     phase 1 = fetch     (parallel pread fan-out, disk busy / CPU idle)
//     phase 2 = compute   (expert GEMMs, CPU busy / disk idle)
//
// Only thread 0 records, and only at ggml_barrier boundaries, so these are true
// wall-clock phase durations -- NOT a sum over the 8 workers, which would
// overcount blocked time by ~nth and make every phase look enormous.
GGML_API void ggml_moe_stream_prof_add(int phase, int64_t us);

// Fetch one CHUNK of a miss's slab. n_chunks>1 splits a single expert slab across
// several callers so every compute thread keeps a read in flight for the whole
// fetch phase -- with only ~6 misses and 8 threads the queue otherwise drains early.
GGML_API void ggml_moe_stream_fetch_chunk(const struct ggml_tensor * t, int miss_idx,
                                          int chunk, int n_chunks);

// Number of misses planned by the most recent ggml_moe_stream_plan() call.
GGML_API int ggml_moe_stream_n_misses(const struct ggml_tensor * t);

// Fetch miss #miss_idx (of the most recent plan) via a pread loop into its
// assigned slot. Safe to call from multiple threads concurrently with
// disjoint miss_idx values (e.g. miss_idx = ith, ith+nth, ...). Aborts with a
// clear stderr message on unrecoverable I/O failure.
GGML_API void ggml_moe_stream_fetch(const struct ggml_tensor * t, int miss_idx);

// Pointer to the resident slab for `expert_id`. Must be called only after the
// fetch phase (and its barrier) for the current plan; aborts if the expert is
// not resident.
GGML_API const char * ggml_moe_stream_slab(const struct ggml_tensor * t, int expert_id);

// Was `expert_id` a HIT (already resident, no pread needed) in the most
// recent ggml_moe_stream_plan() call for this tensor? Only meaningful for an
// expert that was active in that call (row_counts[expert_id] > 0) -- undefined
// for anything else. Same validity window as ggml_moe_stream_n_misses() /
// ggml_moe_stream_slab(): after that plan()'s barrier, before the next plan().
GGML_API bool ggml_moe_stream_expert_is_hit(const struct ggml_tensor * t, int expert_id);

#ifdef __cplusplus
}
#endif
