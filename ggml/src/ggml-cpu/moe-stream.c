// moe-stream: universal SSD expert-streaming for MoE models (Phase 1).
// See ggml/include/ggml-moe-stream.h for the public API and env vars.
//
// ponytail: POSIX-only (open/pread/posix_memalign/pthread) -- this fork's
// deploy targets are ARM macOS and Linux (see PAIA_V1/CLAUDE.md), never
// Windows, so there is no Win32 fallback here. Upgrade path if that changes:
// swap pread() for ReadFile+OVERLAPPED and posix_memalign() for
// _aligned_malloc() behind the existing function boundaries.

#include "ggml-moe-stream.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MOE_STREAM_MAX_TENSORS 512
#define MOE_STREAM_ALIGN       (2 * 1024 * 1024) // 2 MiB, per harvested reference designs

// GGML_MOE_STREAM_STATS=1 latency sampling (Step 1 diagnosis instrumentation).
// One sample = one ggml_moe_stream_fetch() call, i.e. one expert slab pulled
// off disk (usually exactly one pread(), occasionally more on a short read).
// Fixed-size lock-free ring via atomic fetch-add index: samples past the cap
// are dropped (counted, not stored) rather than reallocating on the hot path.
#define MOE_STREAM_MAX_LAT_SAMPLES (1 << 18) // 262144 * 4B = 1 MiB, plenty for a CLI run
static _Atomic uint32_t g_lat_us[MOE_STREAM_MAX_LAT_SAMPLES]; // microseconds per fetch
static _Atomic uint64_t g_lat_count;   // total fetch() calls (may exceed array cap)
static _Atomic uint64_t g_lat_sum_us;  // sum of ALL samples (even dropped ones)

static int moe_stream_cmp_u32(const void * a, const void * b) {
    uint32_t va = *(const uint32_t *) a, vb = *(const uint32_t *) b;
    return (va > vb) - (va < vb);
}

// GGML_MOE_STREAM_TRACE=<path> per-(token,layer) selected-expert-id logging.
// One-shot Phase-1 instrumentation for the expert-routing predictability
// probe: does the router pick the same experts across layers/tokens often
// enough to prefetch? See ggml-moe-stream.h for the call contract.
#define MOE_STREAM_TRACE_MAX_LAYERS 256
static FILE *          g_trace_fp = NULL;
static pthread_mutex_t g_trace_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t        g_trace_token_offset[MOE_STREAM_TRACE_MAX_LAYERS];
static uint64_t        g_trace_rows = 0;

static const char * moe_stream_trace_path(void) {
    static const char * cached = NULL;
    static int checked = 0;
    if (!checked) {
        cached  = getenv("GGML_MOE_STREAM_TRACE");
        checked = 1;
    }
    return cached;
}

// Tags rows from this process so multiple llama-cli invocations (one per
// prompt) can append to one shared trace file without their token_index
// counters (which restart at 0 per process) colliding across prompts.
static int moe_stream_trace_run_id(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_MOE_STREAM_TRACE_RUN");
        cached = v ? atoi(v) : 0;
    }
    return cached;
}

static void moe_stream_trace_atexit(void) {
    if (g_trace_fp) {
        fclose(g_trace_fp);
        g_trace_fp = NULL;
    }
    fprintf(stderr, "[moe-stream-trace] wrote %llu rows\n", (unsigned long long) g_trace_rows);
}

void ggml_moe_stream_trace_experts(const struct ggml_tensor * src0, const struct ggml_tensor * ids) {
    const char * path = moe_stream_trace_path();
    if (!path) {
        return;
    }

    const char * name = ggml_get_name(src0);
    // gate/up/down mul_mat_id calls all reuse the same `ids` (one routing
    // decision per layer per token) -- log only once. Prefer the gate call
    // (standard SwiGLU archs: Qwen3/Qwen3-Next); Noor's MoE FFN has no
    // separate gate projection (up/down only), so fall back to the up call
    // there -- either way this fires exactly once per layer per token.
    if (!strstr(name, "ffn_gate_exps") && !strstr(name, "ffn_up_exps")) {
        return;
    }

    int layer = -1;
    sscanf(name, "blk.%d.", &layer);
    if (layer < 0 || layer >= MOE_STREAM_TRACE_MAX_LAYERS) {
        return;
    }

    const int     n_ids = (int) ids->ne[0];
    const int64_t n_tok = ids->ne[1];

    pthread_mutex_lock(&g_trace_lock);

    if (!g_trace_fp) {
        // Append mode: separate llama-cli invocations (one per prompt, in the
        // Phase-1 sweep) share one trace file. Header only on first creation.
        const bool is_new = access(path, F_OK) != 0;
        g_trace_fp = fopen(path, "a");
        if (!g_trace_fp) {
            fprintf(stderr, "ggml_moe_stream_trace: failed to open '%s': %s\n", path, strerror(errno));
            pthread_mutex_unlock(&g_trace_lock);
            return;
        }
        if (is_new) {
            fprintf(g_trace_fp, "run_id,token_index,layer_index,expert_ids\n");
        }
        atexit(moe_stream_trace_atexit);
    }

    const int      run_id = moe_stream_trace_run_id();
    const uint64_t base   = g_trace_token_offset[layer];
    char buf[512];
    for (int64_t iid1 = 0; iid1 < n_tok; iid1++) {
        int off = 0;
        for (int id = 0; id < n_ids && off < (int) sizeof(buf) - 16; id++) {
            const int32_t eid = *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);
            off += snprintf(buf + off, sizeof(buf) - off, id == 0 ? "%d" : "|%d", eid);
        }
        fprintf(g_trace_fp, "%d,%llu,%d,%s\n", run_id, (unsigned long long) (base + (uint64_t) iid1), layer, buf);
        g_trace_rows++;
    }
    g_trace_token_offset[layer] = base + (uint64_t) n_tok;

    pthread_mutex_unlock(&g_trace_lock);
}

// Prints the aggregate pread-latency table across all tensors. Called once
// from moe_stream_atexit, guarded by moe_stream_stats_enabled().
static void moe_stream_print_lat_stats(void) {
    uint64_t count = atomic_load_explicit(&g_lat_count, memory_order_relaxed);
    if (count == 0) {
        return;
    }
    uint64_t sum_us   = atomic_load_explicit(&g_lat_sum_us, memory_order_relaxed);
    uint64_t stored   = count < MOE_STREAM_MAX_LAT_SAMPLES ? count : MOE_STREAM_MAX_LAT_SAMPLES;
    uint32_t * copy = malloc(stored * sizeof(uint32_t));
    if (!copy) {
        fprintf(stderr, "[moe-stream] STATS fetch_count=%llu mean_us=%.1f (percentiles skipped: OOM)\n",
                (unsigned long long) count, (double) sum_us / (double) count);
        return;
    }
    for (uint64_t i = 0; i < stored; i++) {
        copy[i] = atomic_load_explicit(&g_lat_us[i], memory_order_relaxed);
    }
    qsort(copy, stored, sizeof(uint32_t), moe_stream_cmp_u32);
    uint32_t p50 = copy[(size_t) (stored * 50 / 100)];
    uint32_t p99 = copy[(size_t) (stored * 99 / 100 < stored ? stored * 99 / 100 : stored - 1)];
    fprintf(stderr,
            "[moe-stream] STATS fetch_count=%llu mean_us=%.1f p50_us=%u p99_us=%u (sampled %llu of %llu)\n",
            (unsigned long long) count, (double) sum_us / (double) count, p50, p99,
            (unsigned long long) stored, (unsigned long long) count);
    free(copy);
}

struct moe_slot {
    int      expert_id; // -1 = empty
    uint64_t last_use;
    void   * data;
};

struct moe_stream_entry {
    struct ggml_tensor * tensor;
    char name[GGML_MAX_NAME]; // snapshot of ggml_get_name(tensor) at register time --
                               // the tensor/context can be torn down before our
                               // atexit stats handler runs, so it must not
                               // dereference `tensor` for its name.

    int    fd;
    size_t file_offs;
    size_t slab_bytes;       // bytes per expert (= tensor->nb[2])
    size_t slab_alloc_bytes; // slab_bytes rounded up to a page

    int n_expert;
    int n_slots;

    struct moe_slot * slots;         // [n_slots]
    uint64_t        * use_count;     // [n_expert], LFU frequency
    int              * expert_to_slot; // [n_expert], -1 if not resident
    uint64_t           clock;

    // plan state for the in-flight mul_mat_id call (single-threaded producer
    // in ggml_moe_stream_plan, read-only fan-out consumers in ...fetch)
    int  * miss_expert_id;   // [n_slots] max possible misses per call
    int  * miss_slot_idx;    // [n_slots]
    int    n_misses;
    bool * active_this_call; // [n_expert] -- protects slots this call still
                              // needs from being evicted by a later miss in
                              // the SAME plan() call (a group's experts, or --
                              // pre-chunking -- a whole call's experts)

    pthread_mutex_t lock;

    // stats
    uint64_t          stat_hits;
    uint64_t          stat_misses;
    uint64_t          stat_chunked_calls;
    _Atomic uint64_t stat_bytes_read;
};

static struct moe_stream_entry g_entries[MOE_STREAM_MAX_TENSORS];
static int                     g_n_entries = 0;
static pthread_mutex_t         g_registry_lock = PTHREAD_MUTEX_INITIALIZER;
static int                     g_atexit_registered = 0;

struct moe_fd_entry {
    char * path;
    int    fd;
};
static struct moe_fd_entry g_fds[MOE_STREAM_MAX_TENSORS];
static int                 g_n_fds = 0;

bool ggml_moe_stream_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_MOE_STREAM");
        cached = (v && strcmp(v, "1") == 0) ? 1 : 0;
    }
    return cached != 0;
}

static int moe_stream_n_slots_env(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_MOE_STREAM_SLOTS");
        int n = v ? atoi(v) : 16;
        cached = (n > 0) ? n : 16;
    }
    return cached;
}

static bool moe_stream_stats_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_MOE_STREAM_STATS");
        cached = (v && strcmp(v, "1") == 0) ? 1 : 0;
    }
    return cached != 0;
}

static size_t moe_stream_page_size(void) {
    static long page = 0;
    if (page <= 0) {
        page = sysconf(_SC_PAGESIZE);
        if (page <= 0) {
            page = 4096;
        }
    }
    return (size_t) page;
}

// caller must hold g_registry_lock
static int moe_stream_find_or_open_fd(const char * path) {
    for (int i = 0; i < g_n_fds; i++) {
        if (strcmp(g_fds[i].path, path) == 0) {
            return g_fds[i].fd;
        }
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }
    if (g_n_fds < MOE_STREAM_MAX_TENSORS) {
        g_fds[g_n_fds].path = strdup(path);
        g_fds[g_n_fds].fd   = fd;
        g_n_fds++;
    }
    return fd;
}

static void moe_stream_atexit(void) {
    for (int i = 0; i < g_n_entries; i++) {
        struct moe_stream_entry * e = &g_entries[i];
        fprintf(stderr, "[moe-stream] tensor=%s slots=%d hits=%llu misses=%llu bytes_read=%llu chunked_calls=%llu\n",
                e->name, e->n_slots,
                (unsigned long long) e->stat_hits,
                (unsigned long long) e->stat_misses,
                (unsigned long long) e->stat_bytes_read,
                (unsigned long long) e->stat_chunked_calls);
    }
    moe_stream_print_lat_stats();
}

// find_entry does a bounded linear scan (<=512), lock-free. No hash map: the
// registry is populated once, single-threaded, entirely during model load
// (ggml_moe_stream_register is only ever called from llama_model::load_tensors,
// before any compute thread exists), so g_entries/g_n_entries are read-only
// for the rest of the process's life. This runs on the hot path -- it is
// called from is_registered/plan/n_slots/n_misses/fetch/slab on every
// thread for every mul_mat_id call -- so it must not pay for a lock that
// protects a mutation window which is already over by the time inference starts.
static struct moe_stream_entry * moe_stream_find_entry(const struct ggml_tensor * t) {
    const int n = g_n_entries; // single read; registry is append-only and frozen post-load
    for (int i = 0; i < n; i++) {
        if (g_entries[i].tensor == t) {
            return &g_entries[i];
        }
    }
    return NULL;
}

void ggml_moe_stream_register(struct ggml_tensor * t, const char * path, size_t file_offs) {
    if (!t || !path) {
        return;
    }

    if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
        fprintf(stderr, "ggml_moe_stream: skipping tensor '%s' -- not a plain CPU buffer\n", ggml_get_name(t));
        return;
    }

    pthread_mutex_lock(&g_registry_lock);

    if (g_n_entries >= MOE_STREAM_MAX_TENSORS) {
        pthread_mutex_unlock(&g_registry_lock);
        fprintf(stderr, "ggml_moe_stream: registry full (max %d), skipping tensor '%s'\n",
                MOE_STREAM_MAX_TENSORS, ggml_get_name(t));
        return;
    }

    int fd = moe_stream_find_or_open_fd(path);
    if (fd < 0) {
        pthread_mutex_unlock(&g_registry_lock);
        fprintf(stderr, "ggml_moe_stream: failed to open '%s': %s\n", path, strerror(errno));
        return;
    }

    struct moe_stream_entry * e = &g_entries[g_n_entries++];
    memset(e, 0, sizeof(*e));

    e->tensor    = t;
    snprintf(e->name, sizeof(e->name), "%s", ggml_get_name(t));
    e->fd        = fd;
    e->file_offs = file_offs;
    e->slab_bytes = t->nb[2];
    e->n_expert   = (int) t->ne[2];

    e->n_slots = moe_stream_n_slots_env();
    if (e->n_slots > e->n_expert) e->n_slots = e->n_expert;
    if (e->n_slots < 1)           e->n_slots = 1;

    pthread_mutex_init(&e->lock, NULL);

    const size_t page = moe_stream_page_size();
    e->slab_alloc_bytes = (e->slab_bytes + page - 1) / page * page;

    e->slots          = calloc((size_t) e->n_slots, sizeof(struct moe_slot));
    e->use_count       = calloc((size_t) e->n_expert, sizeof(uint64_t));
    e->expert_to_slot  = malloc((size_t) e->n_expert * sizeof(int));
    e->miss_expert_id  = malloc((size_t) e->n_slots * sizeof(int));
    e->miss_slot_idx   = malloc((size_t) e->n_slots * sizeof(int));
    e->active_this_call = calloc((size_t) e->n_expert, sizeof(bool));
    if (!e->slots || !e->use_count || !e->expert_to_slot || !e->miss_expert_id || !e->miss_slot_idx || !e->active_this_call) {
        fprintf(stderr, "ggml_moe_stream: out of memory registering tensor '%s'\n", ggml_get_name(t));
        abort();
    }

    for (int i = 0; i < e->n_expert; i++) {
        e->expert_to_slot[i] = -1;
    }

    for (int s = 0; s < e->n_slots; s++) {
        e->slots[s].expert_id = -1;
        void * mem = NULL;
        int rc = posix_memalign(&mem, MOE_STREAM_ALIGN, e->slab_alloc_bytes);
        if (rc != 0 || !mem) {
            fprintf(stderr, "ggml_moe_stream: posix_memalign failed for tensor '%s' slot %d\n",
                    ggml_get_name(t), s);
            abort();
        }
        e->slots[s].data = mem;
    }

    if (moe_stream_stats_enabled() && !g_atexit_registered) {
        atexit(moe_stream_atexit);
        g_atexit_registered = 1;
    }

    pthread_mutex_unlock(&g_registry_lock);
}

bool ggml_moe_stream_is_registered(const struct ggml_tensor * t) {
    return moe_stream_find_entry(t) != NULL;
}

// caller must hold e->lock. Never returns a slot whose occupant is marked
// active_this_call -- that expert is still needed later in the SAME
// mul_mat_id call (e.g. a later cur_a in this ubatch), so evicting it would
// leave the compute loop unable to find it a few lines below.
static int moe_stream_pick_victim(struct moe_stream_entry * e) {
    for (int s = 0; s < e->n_slots; s++) {
        if (e->slots[s].expert_id == -1) {
            return s; // empty slots evict first
        }
    }
    int best = -1;
    for (int s = 0; s < e->n_slots; s++) {
        const int occupant = e->slots[s].expert_id;
        if (e->active_this_call[occupant]) {
            continue;
        }
        if (best == -1) {
            best = s;
            continue;
        }
        const uint64_t best_freq = e->use_count[e->slots[best].expert_id];
        const uint64_t cur_freq  = e->use_count[occupant];
        if (cur_freq < best_freq ||
            (cur_freq == best_freq && e->slots[s].last_use < e->slots[best].last_use)) {
            best = s;
        }
    }
    // never -1: at most n_slots-1 OTHER slots can be pinned by the time a
    // miss needs one (this plan() call has at most n_slots active experts,
    // by the caller's group-size contract), so a non-pinned slot always exists.
    return best;
}

void ggml_moe_stream_plan(const struct ggml_tensor * t, const int64_t * row_counts, int n_expert) {
    struct moe_stream_entry * e = moe_stream_find_entry(t);
    if (!e) {
        return;
    }

    pthread_mutex_lock(&e->lock);

    e->clock++;
    e->n_misses = 0;

    const int n = n_expert < e->n_expert ? n_expert : e->n_expert;

    // The caller (ggml-cpu.c) guarantees the number of experts with
    // row_counts[i] > 0 never exceeds n_slots: it splits any call routing to
    // more distinct experts than the pool holds into consecutive groups of at
    // most n_slots and calls plan() once per group. That's what makes the
    // active_this_call protection below sufficient -- pick_victim never needs
    // to evict a slot this same plan() call still needs.
    memset(e->active_this_call, 0, (size_t) e->n_expert * sizeof(bool));
    for (int i = 0; i < n; i++) {
        e->active_this_call[i] = row_counts[i] > 0;
    }

    for (int i = 0; i < n; i++) {
        if (row_counts[i] <= 0) {
            continue;
        }

        e->use_count[i]++;

        int slot = e->expert_to_slot[i];
        if (slot >= 0) {
            e->slots[slot].last_use = e->clock;
            e->stat_hits++;
            continue;
        }

        const int victim = moe_stream_pick_victim(e);
        // n_active <= e->n_slots (checked above) guarantees a non-pinned
        // slot always exists here.
        const int old_expert = e->slots[victim].expert_id;
        if (old_expert >= 0) {
            e->expert_to_slot[old_expert] = -1;
        }

        e->slots[victim].expert_id = i;
        e->slots[victim].last_use  = e->clock;
        e->expert_to_slot[i] = victim;

        e->miss_expert_id[e->n_misses] = i;
        e->miss_slot_idx[e->n_misses]  = victim;
        e->n_misses++;
        e->stat_misses++;
    }

    pthread_mutex_unlock(&e->lock);
}

int ggml_moe_stream_n_slots(const struct ggml_tensor * t) {
    struct moe_stream_entry * e = moe_stream_find_entry(t);
    return e ? e->n_slots : 0;
}

void ggml_moe_stream_mark_chunked(const struct ggml_tensor * t) {
    struct moe_stream_entry * e = moe_stream_find_entry(t);
    if (!e) {
        return;
    }
    pthread_mutex_lock(&e->lock);
    e->stat_chunked_calls++;
    pthread_mutex_unlock(&e->lock);
}

int ggml_moe_stream_n_misses(const struct ggml_tensor * t) {
    struct moe_stream_entry * e = moe_stream_find_entry(t);
    return e ? e->n_misses : 0;
}

void ggml_moe_stream_fetch(const struct ggml_tensor * t, int miss_idx) {
    struct moe_stream_entry * e = moe_stream_find_entry(t);
    if (!e || miss_idx < 0 || miss_idx >= e->n_misses) {
        return;
    }

    const int    slot      = e->miss_slot_idx[miss_idx];
    const int    expert_id = e->miss_expert_id[miss_idx];
    const size_t off       = e->file_offs + (size_t) expert_id * e->slab_bytes;
    char       * dst       = (char *) e->slots[slot].data;
    const size_t to_read   = e->slab_bytes;

    const bool stats = moe_stream_stats_enabled();
    struct timespec t0;
    if (stats) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    size_t done = 0;
    while (done < to_read) {
        ssize_t r = pread(e->fd, dst + done, to_read - done, (off_t) (off + done));
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            fprintf(stderr, "ggml_moe_stream: FATAL pread failed for tensor '%s' expert %d: %s\n",
                    e->name, expert_id, strerror(errno));
            abort();
        }
        if (r == 0) {
            fprintf(stderr, "ggml_moe_stream: FATAL unexpected EOF reading tensor '%s' expert %d\n",
                    e->name, expert_id);
            abort();
        }
        done += (size_t) r;
    }

    atomic_fetch_add_explicit(&e->stat_bytes_read, (uint64_t) to_read, memory_order_relaxed);

    if (stats) {
        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        int64_t sec  = (int64_t) t1.tv_sec  - (int64_t) t0.tv_sec;
        int64_t nsec = (int64_t) t1.tv_nsec - (int64_t) t0.tv_nsec;
        if (nsec < 0) { // borrow -- tv_nsec alone can go negative across a second boundary
            nsec += 1000000000LL;
            sec  -= 1;
        }
        uint64_t us = (uint64_t) sec * 1000000ULL + (uint64_t) nsec / 1000ULL;
        uint64_t idx = atomic_fetch_add_explicit(&g_lat_count, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_lat_sum_us, us, memory_order_relaxed);
        if (idx < MOE_STREAM_MAX_LAT_SAMPLES) {
            atomic_store_explicit(&g_lat_us[idx], (uint32_t) (us > UINT32_MAX ? UINT32_MAX : us), memory_order_relaxed);
        }
    }
}

const char * ggml_moe_stream_slab(const struct ggml_tensor * t, int expert_id) {
    struct moe_stream_entry * e = moe_stream_find_entry(t);
    if (!e || expert_id < 0 || expert_id >= e->n_expert || e->expert_to_slot[expert_id] < 0) {
        fprintf(stderr, "ggml_moe_stream: FATAL slab requested for non-resident expert %d of tensor '%s'\n",
                expert_id, e ? e->name : "?");
        abort();
    }
    return (const char *) e->slots[e->expert_to_slot[expert_id]].data;
}

int ggml_moe_stream_parallel_n(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_MOE_STREAM_PARALLEL");
        int n = v ? atoi(v) : 0;
        cached = (n > 0) ? n : 0;
    }
    return cached;
}

struct moe_stream_fetch_task {
    const struct ggml_tensor * t;
    int                        start;
    int                        stride;
    int                        n_misses;
};

static void * moe_stream_fetch_worker(void * arg) {
    struct moe_stream_fetch_task * task = (struct moe_stream_fetch_task *) arg;
    for (int j = task->start; j < task->n_misses; j += task->stride) {
        ggml_moe_stream_fetch(task->t, j);
    }
    return NULL;
}

// ponytail: ephemeral spawn-join per call, not a persistent pool -- one
// mul_mat_id group fetches at most n_slots (~64) experts, so thread-create
// overhead (a few us) is noise next to a multi-ms pread, and there is no
// pool lifecycle to manage (shutdown, model reload, multi-model processes).
// Upgrade to a persistent worker pool with a condvar queue if profiling ever
// shows thread-spawn cost mattering at this call frequency.
void ggml_moe_stream_fetch_all(const struct ggml_tensor * t, int n_misses) {
    if (n_misses <= 0) {
        return;
    }
    const int requested = ggml_moe_stream_parallel_n();
    const int n = requested < n_misses ? requested : n_misses;
    if (n <= 1) {
        for (int j = 0; j < n_misses; j++) {
            ggml_moe_stream_fetch(t, j);
        }
        return;
    }

    pthread_t threads[n];
    struct moe_stream_fetch_task tasks[n];
    for (int i = 0; i < n; i++) {
        tasks[i].t        = t;
        tasks[i].start    = i;
        tasks[i].stride   = n;
        tasks[i].n_misses = n_misses;
        pthread_create(&threads[i], NULL, moe_stream_fetch_worker, &tasks[i]);
    }
    for (int i = 0; i < n; i++) {
        pthread_join(threads[i], NULL);
    }
}
