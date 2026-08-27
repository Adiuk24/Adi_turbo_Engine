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

#include <assert.h>
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

// GGML_MOE_STREAM_PREFETCH=1: state machine for the P extra landing slots
// reserved per tensor (see ggml-moe-stream.h). NORMAL slots (the original
// n_slots) never carry this state -- only the P prefetch slots do.
enum pf_state {
    PF_FREE     = 0, // available for a new speculative claim
    PF_INFLIGHT = 1, // claimed by plan(), pread in progress on the IO thread
    PF_READY    = 2, // pread complete, buffer valid, awaiting promotion or supersession
};

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
    uint64_t        * use_count;     // [n_expert], LFU frequency (also the cumulative
                                      // route-frequency counter pinlru pins from)
    int              * expert_to_slot; // [n_expert], -1 if not resident
    uint64_t           clock;
    uint64_t           plan_calls; // for periodic LFU decay / pinlru re-pin
    int                pinned_expert; // pinlru only: sticky top-1 hottest expert, -1 = none yet

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

    // ---- prefetch (GGML_MOE_STREAM_PREFETCH=1), see ggml-moe-stream.h ----
    int                 pf_n_slots;    // P granted to this tensor by the global RAM budget; 0 = no prefetch here
    int               * pf_expert_id;  // [pf_n_slots], expert id claimed/held by this pf slot, -1 = none
    _Atomic int       * pf_state;      // [pf_n_slots], enum pf_state -- IO thread release-stores PF_READY,
                                        // plan() acquire-loads it; no lock needed (invariant 4, see plan())
    struct moe_slot   * pf_slots;      // [pf_n_slots], landing buffers; .data is swapped into a NORMAL slot
                                        // on promotion (two pointer writes, never a memcpy)
    _Atomic int       * miss_chunks_done; // [n_slots], per-miss count of completed fetch_chunk() calls this
                                           // plan() cycle -- drives the g_demand_pending bus-yield signal

    uint64_t          stat_prefetch_issued;
    uint64_t          stat_prefetch_hits;   // promotions
    uint64_t          stat_prefetch_wasted; // superseded by a demand fetch before promotion
    uint64_t          stat_prefetch_ring_full;
    _Atomic uint64_t stat_prefetch_bytes;   // speculative bytes, kept separate from stat_bytes_read
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

// ---- prefetch (GGML_MOE_STREAM_PREFETCH=1) ----

// Global RAM budget for prefetch arenas, spent (never refunded) as tensors
// register. Caller must hold g_registry_lock, same as everything else
// touched during ggml_moe_stream_register.
static size_t g_pf_budget_remaining = 0;
static int    g_pf_budget_inited    = 0;
static int    g_pf_tensors_granted  = 0;

struct moe_prefetch_req {
    struct moe_stream_entry * e;
    int                       expert_id;
    int                       pf_slot_idx; // index into e->pf_expert_id / e->pf_state / e->pf_slots
};

#define MOE_PREFETCH_RING_CAP 64

// Single-producer/single-consumer ring: thread 0 inside plan() is the sole
// producer, the one global IO thread is the sole consumer. head/tail are
// monotonic counters, indexed mod capacity.
static struct moe_prefetch_req g_pf_ring[MOE_PREFETCH_RING_CAP];
static _Atomic size_t          g_pf_ring_head;
static _Atomic size_t          g_pf_ring_tail;

static pthread_t      g_pf_io_thread;
static _Atomic int    g_pf_thread_live;
static pthread_once_t g_pf_once = PTHREAD_ONCE_INIT;

static pthread_mutex_t g_pf_io_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_pf_io_cond = PTHREAD_COND_INITIALIZER;
static _Atomic int     g_pf_shutdown;

// Incremented in plan() by the number of misses just planned, decremented in
// fetch_chunk() when a miss's last chunk completes. The prefetch IO thread
// polls this between its own 4 MiB sub-reads and backs off while it's > 0,
// so speculative reads never contend the bus against demand fetches.
static _Atomic int g_demand_pending;

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

// GGML_MOE_STREAM_PREFETCH=1: enable the background speculative-prefetch IO
// thread. Default off -- zero behavior change, byte-identical to before.
static bool moe_stream_prefetch_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_MOE_STREAM_PREFETCH");
        cached = (v && strcmp(v, "1") == 0) ? 1 : 0;
    }
    return cached != 0;
}

// GGML_MOE_STREAM_PREFETCH_SLOTS=<n>: extra landing slots per tensor (default 4).
static int moe_stream_prefetch_slots_env(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_MOE_STREAM_PREFETCH_SLOTS");
        int n = v ? atoi(v) : 4;
        cached = (n > 0) ? n : 4;
    }
    return cached;
}

// GGML_MOE_STREAM_PREFETCH_DEPTH=<n>: tensors ahead (registration order) to predict for (default 2).
static int moe_stream_prefetch_depth_env(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_MOE_STREAM_PREFETCH_DEPTH");
        int n = v ? atoi(v) : 2;
        cached = (n > 0) ? n : 2;
    }
    return cached;
}

// GGML_MOE_STREAM_PREFETCH_MB=<n>: total prefetch arena budget across ALL
// tensors, in MiB (default 2048). Prevents P x slab_alloc_bytes x n_tensors
// from becoming catastrophic (e.g. 4 x 90 MB x 183 tensors).
static size_t moe_stream_prefetch_budget_bytes(void) {
    static long long cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_MOE_STREAM_PREFETCH_MB");
        long long mb = v ? atoll(v) : 2048;
        if (mb <= 0) {
            mb = 2048;
        }
        cached = mb * 1024 * 1024;
    }
    return (size_t) cached;
}

enum moe_evict_policy {
    MOE_EVICT_LFU    = 0, // default: LFU-with-recency-tiebreak, periodic decay
    MOE_EVICT_PINLRU = 1, // sticky top-1 pin per tensor + plain LRU for the rest
};

// GGML_MOE_STREAM_EVICT=pinlru switches eviction policy; anything else (unset
// included) keeps the original LFU behavior.
static enum moe_evict_policy moe_stream_evict_policy(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("GGML_MOE_STREAM_EVICT");
        cached = (v && strcmp(v, "pinlru") == 0) ? MOE_EVICT_PINLRU : MOE_EVICT_LFU;
    }
    return (enum moe_evict_policy) cached;
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
    // GGML_MOE_STREAM_NOCACHE=1: bypass the page cache on the expert fetch path.
    // At low hit rates the stream sweeps far more data than RAM, so caching it
    // only double-copies every byte and evicts whatever mmap'd dense weights were
    // resident (kimi-k3-in-c measured direct reads FASTER for the same regime).
    const char * nc = getenv("GGML_MOE_STREAM_NOCACHE");
    if (nc && atoi(nc) != 0) {
#if defined(__APPLE__)
        if (fcntl(fd, F_NOCACHE, 1) != 0) {
            fprintf(stderr, "moe-stream: F_NOCACHE failed on %s (continuing buffered)\n", path);
        } else {
            fprintf(stderr, "moe-stream: page cache BYPASSED (F_NOCACHE) for %s\n", path);
        }
#elif defined(POSIX_FADV_DONTNEED)
        // Linux: O_DIRECT needs aligned buffers we don't guarantee; advise-away instead.
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        fprintf(stderr, "moe-stream: POSIX_FADV_DONTNEED advised for %s\n", path);
#endif
    }
    if (g_n_fds < MOE_STREAM_MAX_TENSORS) {
        g_fds[g_n_fds].path = strdup(path);
        g_fds[g_n_fds].fd   = fd;
        g_n_fds++;
    }
    return fd;
}

// ---------------------------------------------------------------- phase profiler
// See ggml-moe-stream.h. Written only by thread 0 at barrier boundaries, so plain
// (non-atomic) accumulation is safe and costs nothing on the hot path.
static _Atomic int g_inflight;
static _Atomic int g_inflight_max;
static int64_t g_prof_us[3];
static int64_t g_prof_n;

void ggml_moe_stream_prof_add(int phase, int64_t us) {
    if (phase >= 0 && phase < 3) {
        g_prof_us[phase] += us;
        if (phase == 0) {
            g_prof_n++;
        }
    }
}

static void moe_stream_dump_prof(void) {
    const int64_t tot = g_prof_us[0] + g_prof_us[1] + g_prof_us[2];
    if (tot <= 0) {
        return;
    }
    static const char * names[3] = { "plan", "fetch (disk)", "compute (cpu)" };
    fprintf(stderr, "[moe-stream] critical-path phases over %lld streamed mul_mat_id call(s):\n",
            (long long) g_prof_n);
    for (int i = 0; i < 3; i++) {
        fprintf(stderr, "[moe-stream]   %-14s %8.2f s  %5.1f%%\n",
                names[i], g_prof_us[i] / 1e6, 100.0 * (double) g_prof_us[i] / (double) tot);
    }
    fprintf(stderr, "[moe-stream]   %-14s %8.2f s\n", "TOTAL", tot / 1e6);
    fprintf(stderr, "[moe-stream]   peak in-flight reads: %d\n",
            atomic_load_explicit(&g_inflight_max, memory_order_relaxed));
}

static int moe_stream_cmp_u64_desc(const void * a, const void * b) {
    const uint64_t x = *(const uint64_t *) a, y = *(const uint64_t *) b;
    return (x < y) - (x > y);
}

// GGML_MOE_STREAM_HIST=1 additionally dumps the routing-concentration curve:
// the share of all expert selections captured by the top-K hottest experts.
//
// This is the number that decides whether STATIC PINNING is worth building. With
// n_slots=8 of 256 experts, uniform routing predicts a 3.1% hit rate; a measured
// rate far above that means routing is concentrated and a pinned working set of the
// top-K experts would cut bytes-read-per-token directly. `use_count` is already
// maintained for LFU eviction, so this costs nothing on the hot path.
static void moe_stream_dump_hist(void) {
    const int KS[] = { 1, 2, 4, 8, 16, 32, 64 };
    const int NK = (int) (sizeof(KS) / sizeof(KS[0]));

    // aggregate across every registered tensor: per-tensor curves are noisy,
    // and a pinning policy has to be decided per tensor anyway, so report the
    // mean concentration over all 129 of them.
    double cum[7] = { 0 };
    int    n_used = 0;

    for (int i = 0; i < g_n_entries; i++) {
        struct moe_stream_entry * e = &g_entries[i];
        if (e->n_expert <= 0) {
            continue;
        }
        uint64_t * v = malloc((size_t) e->n_expert * sizeof(uint64_t));
        if (!v) {
            continue;
        }
        uint64_t total = 0;
        for (int j = 0; j < e->n_expert; j++) {
            v[j] = e->use_count[j];
            total += v[j];
        }
        if (total == 0) {
            free(v);
            continue;
        }
        qsort(v, (size_t) e->n_expert, sizeof(uint64_t), moe_stream_cmp_u64_desc);
        for (int k = 0; k < NK; k++) {
            uint64_t s = 0;
            for (int j = 0; j < KS[k] && j < e->n_expert; j++) {
                s += v[j];
            }
            cum[k] += (double) s / (double) total;
        }
        n_used++;
        free(v);
    }
    if (n_used == 0) {
        return;
    }
    fprintf(stderr, "[moe-stream] routing concentration, mean over %d tensor(s) of %d experts:\n",
            n_used, g_n_entries ? g_entries[0].n_expert : 0);
    for (int k = 0; k < NK; k++) {
        fprintf(stderr, "[moe-stream]   top-%-3d captures %5.1f%% of selections\n",
                KS[k], 100.0 * cum[k] / n_used);
    }
}

static void moe_stream_atexit(void) {
    for (int i = 0; i < g_n_entries; i++) {
        struct moe_stream_entry * e = &g_entries[i];
        // pinned: 1 if this tensor's pinlru pin is currently resident, else 0
        // (always 0 under plain LFU, since pinned_expert stays -1).
        const int pinned = (e->pinned_expert >= 0 && e->expert_to_slot[e->pinned_expert] >= 0) ? 1 : 0;
        fprintf(stderr, "[moe-stream] tensor=%s slots=%d hits=%llu misses=%llu bytes_read=%llu chunked_calls=%llu pinned=%d\n",
                e->name, e->n_slots,
                (unsigned long long) e->stat_hits,
                (unsigned long long) e->stat_misses,
                (unsigned long long) e->stat_bytes_read,
                (unsigned long long) e->stat_chunked_calls,
                pinned);
        if (moe_stream_prefetch_enabled() && e->pf_n_slots > 0) {
            fprintf(stderr, "[moe-stream]   prefetch: pf_slots=%d issued=%llu hits=%llu wasted=%llu ring_full=%llu bytes=%llu\n",
                    e->pf_n_slots,
                    (unsigned long long) e->stat_prefetch_issued,
                    (unsigned long long) e->stat_prefetch_hits,
                    (unsigned long long) e->stat_prefetch_wasted,
                    (unsigned long long) e->stat_prefetch_ring_full,
                    (unsigned long long) e->stat_prefetch_bytes);
        }
    }
    if (moe_stream_prefetch_enabled()) {
        uint64_t tot_issued = 0, tot_hits = 0, tot_wasted = 0, tot_ring_full = 0, tot_bytes = 0;
        for (int i = 0; i < g_n_entries; i++) {
            struct moe_stream_entry * e = &g_entries[i];
            tot_issued    += e->stat_prefetch_issued;
            tot_hits      += e->stat_prefetch_hits;
            tot_wasted    += e->stat_prefetch_wasted;
            tot_ring_full += e->stat_prefetch_ring_full;
            tot_bytes     += e->stat_prefetch_bytes;
        }
        fprintf(stderr, "[moe-stream] prefetch totals: issued=%llu hits=%llu wasted=%llu ring_full=%llu bytes=%llu\n",
                (unsigned long long) tot_issued, (unsigned long long) tot_hits, (unsigned long long) tot_wasted,
                (unsigned long long) tot_ring_full, (unsigned long long) tot_bytes);
    }
    if (getenv("GGML_MOE_STREAM_HIST")) {
        moe_stream_dump_hist();
    }
    if (getenv("GGML_MOE_STREAM_PROF")) {
        moe_stream_dump_prof();
    }
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
    e->pinned_expert = -1;

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

    // prefetch: grant this tensor P extra landing slots only if the global
    // RAM budget still has room for all of them (all-or-nothing per tensor,
    // never partial -- see ggml-moe-stream.h). e->pf_n_slots stays 0 (fields
    // NULL) otherwise, so every prefetch code path is a no-op for this tensor.
    e->pf_n_slots       = 0;
    e->pf_expert_id     = NULL;
    e->pf_state         = NULL;
    e->pf_slots         = NULL;
    e->miss_chunks_done = NULL;

    if (moe_stream_prefetch_enabled()) {
        if (!g_pf_budget_inited) {
            g_pf_budget_remaining = moe_stream_prefetch_budget_bytes();
            g_pf_budget_inited    = 1;
        }
        const int    want_p     = moe_stream_prefetch_slots_env();
        const size_t want_bytes = (size_t) want_p * e->slab_alloc_bytes;
        if (want_bytes <= g_pf_budget_remaining) {
            e->pf_n_slots = want_p;
            g_pf_budget_remaining -= want_bytes;
            g_pf_tensors_granted++;
        }

        if (e->pf_n_slots > 0) {
            e->pf_expert_id     = malloc((size_t) e->pf_n_slots * sizeof(int));
            e->pf_state         = malloc((size_t) e->pf_n_slots * sizeof(_Atomic int));
            e->pf_slots         = calloc((size_t) e->pf_n_slots, sizeof(struct moe_slot));
            e->miss_chunks_done = malloc((size_t) e->n_slots * sizeof(_Atomic int));
            if (!e->pf_expert_id || !e->pf_state || !e->pf_slots || !e->miss_chunks_done) {
                fprintf(stderr, "ggml_moe_stream: out of memory allocating prefetch state for tensor '%s'\n",
                        ggml_get_name(t));
                abort();
            }
            for (int p = 0; p < e->pf_n_slots; p++) {
                e->pf_expert_id[p] = -1;
                atomic_init(&e->pf_state[p], PF_FREE);
                void * mem = NULL;
                int rc = posix_memalign(&mem, MOE_STREAM_ALIGN, e->slab_alloc_bytes);
                if (rc != 0 || !mem) {
                    fprintf(stderr, "ggml_moe_stream: posix_memalign failed for tensor '%s' prefetch slot %d\n",
                            ggml_get_name(t), p);
                    abort();
                }
                e->pf_slots[p].data       = mem;
                e->pf_slots[p].expert_id  = -1;
            }
            for (int i = 0; i < e->n_slots; i++) {
                atomic_init(&e->miss_chunks_done[i], 0);
            }
        }
    }

    if (moe_stream_stats_enabled() && !g_atexit_registered) {
        fprintf(stderr, "[moe-stream] eviction policy: %s\n",
                moe_stream_evict_policy() == MOE_EVICT_PINLRU ? "pinlru" : "lfu");
        atexit(moe_stream_atexit);
        g_atexit_registered = 1;
    }

    pthread_mutex_unlock(&g_registry_lock);
}

bool ggml_moe_stream_is_registered(const struct ggml_tensor * t) {
    return moe_stream_find_entry(t) != NULL;
}

// true if slot `s` is a better (more evictable) victim than the current
// candidate `cur`, under the active policy. LFU: lowest use_count, recency
// tiebreak. pinlru: lowest last_use (plain LRU) -- last_use is stamped with
// e->clock on every hit and every fill, so it already doubles as an LRU stamp.
static bool moe_stream_evicts_before(const struct moe_stream_entry * e, enum moe_evict_policy policy,
                                      int s, int cur) {
    if (policy == MOE_EVICT_PINLRU) {
        return e->slots[s].last_use < e->slots[cur].last_use;
    }
    const uint64_t cur_freq = e->use_count[e->slots[cur].expert_id];
    const uint64_t s_freq   = e->use_count[e->slots[s].expert_id];
    return s_freq < cur_freq || (s_freq == cur_freq && e->slots[s].last_use < e->slots[cur].last_use);
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
    const enum moe_evict_policy policy = moe_stream_evict_policy();
    int best     = -1; // best candidate that also honors the pinlru pin
    int fallback = -1; // best candidate ignoring the pin, used only if the pin leaves nothing else
    for (int s = 0; s < e->n_slots; s++) {
        const int occupant = e->slots[s].expert_id;
        if (e->active_this_call[occupant]) {
            continue;
        }
        if (fallback == -1 || moe_stream_evicts_before(e, policy, s, fallback)) {
            fallback = s;
        }
        if (policy == MOE_EVICT_PINLRU && occupant == e->pinned_expert) {
            continue; // pinned expert's slot is never chosen as a victim, except via fallback below
        }
        if (best == -1 || moe_stream_evicts_before(e, policy, s, best)) {
            best = s;
        }
    }
    // never -1: at most n_slots-1 OTHER slots can be active_this_call (this
    // plan() call has at most n_slots active experts, by the caller's
    // group-size contract), so a non-active slot always exists as `fallback`.
    // `best` alone can be -1 if the pin occupies the only non-active slot
    // (e.g. n_slots==1) -- fall back to evicting the pin rather than stalling.
    return best != -1 ? best : fallback;
}

// ---- prefetch (GGML_MOE_STREAM_PREFETCH=1): ring, IO thread, predictor ----

static bool moe_stream_pf_ring_push(const struct moe_prefetch_req * req) {
    const size_t head = atomic_load_explicit(&g_pf_ring_head, memory_order_relaxed);
    const size_t tail = atomic_load_explicit(&g_pf_ring_tail, memory_order_acquire);
    if (head - tail >= MOE_PREFETCH_RING_CAP) {
        return false; // full
    }
    g_pf_ring[head % MOE_PREFETCH_RING_CAP] = *req;
    atomic_store_explicit(&g_pf_ring_head, head + 1, memory_order_release);
    pthread_mutex_lock(&g_pf_io_lock);
    pthread_cond_signal(&g_pf_io_cond);
    pthread_mutex_unlock(&g_pf_io_lock);
    return true;
}

static bool moe_stream_pf_ring_pop(struct moe_prefetch_req * out) {
    const size_t tail = atomic_load_explicit(&g_pf_ring_tail, memory_order_relaxed);
    const size_t head = atomic_load_explicit(&g_pf_ring_head, memory_order_acquire);
    if (tail == head) {
        return false; // empty
    }
    *out = g_pf_ring[tail % MOE_PREFETCH_RING_CAP];
    atomic_store_explicit(&g_pf_ring_tail, tail + 1, memory_order_release);
    return true;
}

// Runs on the IO thread only. Reads one expert's slab into its claimed pf
// slot's buffer in 4 MiB sub-reads, backing off while a demand fetch is
// outstanding anywhere (g_demand_pending). Never touches e->lock: the slot
// was claimed (PF_INFLIGHT) by plan() before this request was enqueued, and
// nothing else writes this buffer or reads it until PF_READY is visible
// (invariant 4: state alone is the synchronization, no lock).
static void moe_stream_pf_do_fetch(const struct moe_prefetch_req * req) {
    struct moe_stream_entry * e         = req->e;
    const int                 expert_id = req->expert_id;
    const int                 p         = req->pf_slot_idx;

    const size_t off     = e->file_offs + (size_t) expert_id * e->slab_bytes;
    char       * dst      = (char *) e->pf_slots[p].data;
    const size_t to_read  = e->slab_bytes;
    const size_t sub      = 4 * 1024 * 1024;
    size_t       done     = 0;

    while (done < to_read) {
        while (atomic_load_explicit(&g_demand_pending, memory_order_acquire) > 0) {
            if (atomic_load_explicit(&g_pf_shutdown, memory_order_relaxed)) {
                break;
            }
            usleep(500);
        }

        size_t chunk = to_read - done;
        if (chunk > sub) {
            chunk = sub;
        }
        size_t chunk_done = 0;
        while (chunk_done < chunk) {
            ssize_t r = pread(e->fd, dst + done + chunk_done, chunk - chunk_done, (off_t) (off + done + chunk_done));
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                // speculative, not on the critical path -- abandon the slot instead of
                // abort()ing like the demand-fetch path does (ggml_moe_stream_fetch_chunk)
                fprintf(stderr, "ggml_moe_stream: prefetch pread failed for tensor '%s' expert %d: %s (abandoning prefetch)\n",
                        e->name, expert_id, strerror(errno));
                e->pf_expert_id[p] = -1;
                atomic_store_explicit(&e->pf_state[p], PF_FREE, memory_order_release);
                return;
            }
            if (r == 0) {
                fprintf(stderr, "ggml_moe_stream: prefetch EOF for tensor '%s' expert %d (abandoning prefetch)\n",
                        e->name, expert_id);
                e->pf_expert_id[p] = -1;
                atomic_store_explicit(&e->pf_state[p], PF_FREE, memory_order_release);
                return;
            }
            chunk_done += (size_t) r;
        }
        done += chunk;
    }

    atomic_fetch_add_explicit(&e->stat_prefetch_bytes, (uint64_t) to_read, memory_order_relaxed);
    // release-store: publishes the completed buffer to plan()'s acquire-load (invariant 4)
    atomic_store_explicit(&e->pf_state[p], PF_READY, memory_order_release);
}

static void * moe_stream_pf_io_main(void * arg) {
    (void) arg;
    for (;;) {
        struct moe_prefetch_req req;
        if (moe_stream_pf_ring_pop(&req)) {
            moe_stream_pf_do_fetch(&req);
            continue;
        }
        if (atomic_load_explicit(&g_pf_shutdown, memory_order_acquire)) {
            break;
        }
        // kimi-k3 pattern: 1ms-timeout condvar wait instead of a busy spin
        pthread_mutex_lock(&g_pf_io_lock);
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 1000000;
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec  += 1;
            ts.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&g_pf_io_cond, &g_pf_io_lock, &ts);
        pthread_mutex_unlock(&g_pf_io_lock);
    }
    return NULL;
}

// moe-stream has no per-entry teardown path today (atexit only, see the file
// header). This only shuts down the global IO thread cleanly. Registered via
// atexit() at first-spawn time (inside a plan() call, i.e. after model load),
// which is always AFTER moe_stream_atexit's own atexit() registration (that
// happens at model-load time, see ggml_moe_stream_register). atexit() runs
// handlers LIFO, so this join always completes before moe_stream_atexit
// prints the final prefetch stats below.
static void moe_stream_pf_atexit_join(void) {
    atomic_store_explicit(&g_pf_shutdown, 1, memory_order_release);
    pthread_mutex_lock(&g_pf_io_lock);
    pthread_cond_signal(&g_pf_io_cond);
    pthread_mutex_unlock(&g_pf_io_lock);
    pthread_join(g_pf_io_thread, NULL);
}

static void moe_stream_pf_thread_init(void) {
    if (moe_stream_stats_enabled()) {
        pthread_mutex_lock(&g_registry_lock);
        const size_t budget  = moe_stream_prefetch_budget_bytes();
        const size_t used    = budget - g_pf_budget_remaining;
        const int    granted = g_pf_tensors_granted;
        const int    total   = g_n_entries;
        pthread_mutex_unlock(&g_registry_lock);
        fprintf(stderr, "[moe-stream] prefetch: granted %d/%d tensor(s) %zu MiB of %zu MiB budget (slots=%d depth=%d)\n",
                granted, total, used / (1024 * 1024), budget / (1024 * 1024),
                moe_stream_prefetch_slots_env(), moe_stream_prefetch_depth_env());
    }
    atomic_store_explicit(&g_pf_shutdown, 0, memory_order_relaxed);
    int rc = pthread_create(&g_pf_io_thread, NULL, moe_stream_pf_io_main, NULL);
    if (rc != 0) {
        fprintf(stderr, "ggml_moe_stream: failed to spawn prefetch IO thread: %s (prefetch disabled for this run)\n",
                strerror(rc));
        return;
    }
    atomic_store_explicit(&g_pf_thread_live, 1, memory_order_release);
    atexit(moe_stream_pf_atexit_join);
}

// Lazy-spawns the one global IO thread on the first plan() call with
// prefetch on. pthread_once makes this safe even if two threadpools from
// separate llama_context instances both call plan() concurrently.
static void moe_stream_pf_ensure_thread(void) {
    pthread_once(&g_pf_once, moe_stream_pf_thread_init);
}

// caller must NOT hold e->lock (called after ggml_moe_stream_plan releases
// it -- see the deadlock-avoidance note there). Locks each target tensor's
// own lock with trylock, one at a time, never nesting two entry locks.
static void moe_stream_prefetch_enqueue_predictions(struct moe_stream_entry * e) {
    const int e_idx = (int) (e - g_entries);
    const int depth = moe_stream_prefetch_depth_env();

    for (int d = 1; d <= depth; d++) {
        const int f_idx = e_idx + d;
        if (f_idx >= g_n_entries) {
            break;
        }
        struct moe_stream_entry * f = &g_entries[f_idx];
        if (f->pf_n_slots <= 0) {
            continue; // this tensor got no prefetch slots from the RAM budget
        }
        if (pthread_mutex_trylock(&f->lock) != 0) {
            continue; // f's own plan() owns the lock right now -- skip this round
        }

        int n_free = 0;
        for (int p = 0; p < f->pf_n_slots; p++) {
            if (atomic_load_explicit(&f->pf_state[p], memory_order_relaxed) == PF_FREE) {
                n_free++;
            }
        }

        // top-M (M = however many PF_FREE slots f has right now) experts by
        // use_count that are neither resident nor already claimed by an
        // earlier prediction for f
        for (int pick = 0; pick < n_free; pick++) {
            int      best      = -1;
            uint64_t best_freq = 0;
            for (int i = 0; i < f->n_expert; i++) {
                if (f->expert_to_slot[i] >= 0 || f->use_count[i] == 0) {
                    continue; // resident, or never routed -- not worth predicting
                }
                bool already_pf = false;
                for (int p = 0; p < f->pf_n_slots; p++) {
                    // state check MUST come first: it's the acquire-load that
                    // establishes happens-before with the IO thread's writes,
                    // via the ring's release/acquire pair (see
                    // moe_stream_pf_do_fetch). Reading pf_expert_id before an
                    // acquire that observes it is a data race even though f->lock
                    // is held here -- the IO thread's abandon-on-error path
                    // writes pf_expert_id without taking any entry lock.
                    if (atomic_load_explicit(&f->pf_state[p], memory_order_acquire) != PF_FREE &&
                        f->pf_expert_id[p] == i) {
                        already_pf = true;
                        break;
                    }
                }
                if (already_pf) {
                    continue;
                }
                if (best < 0 || f->use_count[i] > best_freq) {
                    best      = i;
                    best_freq = f->use_count[i];
                }
            }
            if (best < 0) {
                break; // no more candidates worth predicting
            }

            int slot_p = -1;
            for (int p = 0; p < f->pf_n_slots; p++) {
                if (atomic_load_explicit(&f->pf_state[p], memory_order_relaxed) == PF_FREE) {
                    slot_p = p;
                    break;
                }
            }
            if (slot_p < 0) {
                break; // n_free was stale (shouldn't happen under a single trylock owner, but be safe)
            }

            // claim first (PF_INFLIGHT), publish second (ring push) -- the IO
            // thread must never see a request for a slot that isn't claimed yet
            f->pf_expert_id[slot_p] = best;
            atomic_store_explicit(&f->pf_state[slot_p], PF_INFLIGHT, memory_order_release);

            const struct moe_prefetch_req req = { .e = f, .expert_id = best, .pf_slot_idx = slot_p };
            if (!moe_stream_pf_ring_push(&req)) {
                // ring full: undo the claim so the slot stays available
                f->pf_expert_id[slot_p] = -1;
                atomic_store_explicit(&f->pf_state[slot_p], PF_FREE, memory_order_relaxed);
                f->stat_prefetch_ring_full++;
                break; // ring is globally full -- further attempts this round will fail too
            }
            f->stat_prefetch_issued++;
        }

        pthread_mutex_unlock(&f->lock);
    }
}

void ggml_moe_stream_plan(const struct ggml_tensor * t, const int64_t * row_counts, int n_expert) {
    struct moe_stream_entry * e = moe_stream_find_entry(t);
    if (!e) {
        return;
    }

    pthread_mutex_lock(&e->lock);

    e->clock++;
    e->n_misses = 0;

    // reset this cycle's per-miss chunk-completion counters (bounded by
    // n_slots, the max possible misses); thread 0 only, so plain writes
    // under the release below are fine -- fetch_chunk() only reads/writes
    // indices [0, n_misses) which are (re)assigned again a few lines down
    if (e->miss_chunks_done) {
        for (int i = 0; i < e->n_slots; i++) {
            atomic_store_explicit(&e->miss_chunks_done[i], 0, memory_order_relaxed);
        }
    }

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

    e->plan_calls++;
    const enum moe_evict_policy policy = moe_stream_evict_policy();
    if (policy == MOE_EVICT_PINLRU) {
        // Sticky top-1 pin: every 64 plan() calls, re-derive the hottest
        // expert from the cumulative (never decayed, see the `else` below)
        // route frequency and pin it. 64 (not 1024): plan() runs once per
        // token per tensor, so a longer period would leave the pin dormant
        // for the first ~1k decoded tokens; the top-1 cumulative count is
        // stable enough that re-deriving often doesn't flap it.
        if ((e->plan_calls & 63) == 0) {
            int      top      = -1;
            uint64_t top_freq = 0;
            for (int i = 0; i < e->n_expert; i++) {
                if (e->use_count[i] > top_freq) {
                    top_freq = e->use_count[i];
                    top      = i;
                }
            }
            e->pinned_expert = top;
        }
    } else {
        // Audit adoption (PR #25294's hotness decay): plain LFU never forgets, so an
        // early-hot expert squats in a slot forever even after routing drifts with the
        // topic. Halve all counts every 256 plan() calls (~256 decoded tokens per
        // tensor); recency (last_use) still breaks ties.
        if ((e->plan_calls & 255) == 0) {
            for (int i = 0; i < e->n_expert; i++) {
                e->use_count[i] >>= 1;
            }
        }
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
            // superseded-prefetch cleanup: `i` may ALSO be sitting PF_READY
            // in a prefetch slot (a prediction that lost the race to this
            // demand fetch, or to an earlier promotion). Keep the normal
            // slot, free the pf slot back to PF_FREE (invalidation subtlety).
            if (e->pf_n_slots > 0) {
                for (int p = 0; p < e->pf_n_slots; p++) {
                    // acquire-load first: see the comment in
                    // moe_stream_prefetch_enqueue_predictions -- this is the
                    // edge that makes reading pf_expert_id[p] safe.
                    if (atomic_load_explicit(&e->pf_state[p], memory_order_acquire) == PF_READY &&
                        e->pf_expert_id[p] == i) {
                        e->pf_expert_id[p] = -1;
                        atomic_store_explicit(&e->pf_state[p], PF_FREE, memory_order_release);
                        e->stat_prefetch_wasted++;
                        break;
                    }
                }
            }
            continue;
        }

        // promotion: is `i` sitting PF_READY in a prefetch slot? A
        // PF_INFLIGHT match is treated as a plain miss below and re-derived
        // as superseded on a later plan() once it turns PF_READY (invariant 3
        // -- plan() never blocks on the IO thread).
        int promoted_from = -1;
        if (e->pf_n_slots > 0) {
            for (int p = 0; p < e->pf_n_slots; p++) {
                if (atomic_load_explicit(&e->pf_state[p], memory_order_acquire) == PF_READY &&
                    e->pf_expert_id[p] == i) {
                    promoted_from = p;
                    break;
                }
            }
        }

        const int victim = moe_stream_pick_victim(e);
        // n_active <= e->n_slots (checked above) guarantees a non-pinned
        // slot always exists here.
        const int old_expert = e->slots[victim].expert_id;
        // safety invariant: a slot this call still needs (active_this_call)
        // must never be picked as a victim -- see moe_stream_pick_victim.
        assert(old_expert < 0 || !e->active_this_call[old_expert]);
        if (old_expert >= 0) {
            e->expert_to_slot[old_expert] = -1;
        }

        if (promoted_from >= 0) {
            // promote by swap: two pointer writes, never a memcpy (invariant
            // 1/2 -- this runs under e->lock in plan(), before any barrier
            // releases compute threads, so compute never reads a pf buffer)
            void * tmp = e->slots[victim].data;
            e->slots[victim].data           = e->pf_slots[promoted_from].data;
            e->pf_slots[promoted_from].data = tmp;

            e->pf_expert_id[promoted_from] = -1;
            atomic_store_explicit(&e->pf_state[promoted_from], PF_FREE, memory_order_release);

            e->slots[victim].expert_id = i;
            e->slots[victim].last_use  = e->clock;
            e->expert_to_slot[i] = victim;

            e->stat_hits++;
            e->stat_prefetch_hits++;
            continue;
        }

        e->slots[victim].expert_id = i;
        e->slots[victim].last_use  = e->clock;
        e->expert_to_slot[i] = victim;

        e->miss_expert_id[e->n_misses] = i;
        e->miss_slot_idx[e->n_misses]  = victim;
        e->n_misses++;
        e->stat_misses++;
    }

    if (moe_stream_prefetch_enabled() && e->n_misses > 0) {
        atomic_fetch_add_explicit(&g_demand_pending, e->n_misses, memory_order_relaxed);
    }

    pthread_mutex_unlock(&e->lock);

    // Prediction enqueue happens AFTER releasing e->lock, one target lock at
    // a time via trylock -- never hold two entry locks at once (deadlock
    // avoidance), see moe_stream_prefetch_enqueue_predictions.
    if (moe_stream_prefetch_enabled()) {
        moe_stream_pf_ensure_thread();
        if (atomic_load_explicit(&g_pf_thread_live, memory_order_acquire)) {
            moe_stream_prefetch_enqueue_predictions(e);
        }
    }
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
    ggml_moe_stream_fetch_chunk(t, miss_idx, 0, 1);
}

void ggml_moe_stream_fetch_chunk(const struct ggml_tensor * t, int miss_idx, int chunk, int n_chunks) {
    struct moe_stream_entry * e = moe_stream_find_entry(t);
    if (!e || miss_idx < 0 || miss_idx >= e->n_misses) {
        return;
    }

    const int    slot      = e->miss_slot_idx[miss_idx];
    const int    expert_id = e->miss_expert_id[miss_idx];
    size_t       off       = e->file_offs + (size_t) expert_id * e->slab_bytes;
    char       * dst       = (char *) e->slots[slot].data;
    size_t       to_read   = e->slab_bytes;

    // Split one expert slab across n_chunks callers. The fetch phase has only
    // n_misses (~6 at top-6 routing) units of work but nth (8) threads, so two
    // threads idle immediately and the rest finish staggered -- the SSD queue
    // drains long before the barrier. Measured: 900 MB/s achieved against 1386
    // MB/s the device sustains at QD=6. Chunking keeps every thread issuing for
    // the whole phase, which is what actually raises sustained queue depth.
    if (n_chunks > 1) {
        // Audit F5: this used MOE_STREAM_ALIGN (2 MiB) as the split floor, which
        // degenerates CHUNKS=4 on a 3.58 MiB slab into 2 real chunks + 2 no-ops.
        // The actual page is 16 KiB; align to that so requested parallelism is real.
        const size_t page  = moe_stream_page_size();
        // page-align the split so each pread starts on a page boundary
        size_t per = (e->slab_bytes + n_chunks - 1) / n_chunks;
        per = ((per + page - 1) / page) * page;
        const size_t start = (size_t) chunk * per;
        if (start >= e->slab_bytes) {
            return;                       // this chunk is past the end; nothing to do
        }
        size_t len = e->slab_bytes - start;
        if (len > per) {
            len = per;
        }
        off     += start;
        dst     += start;
        to_read  = len;
    }

    atomic_fetch_add_explicit(&g_inflight, 1, memory_order_relaxed);
    {
        int cur = atomic_load_explicit(&g_inflight, memory_order_relaxed);
        int mx  = atomic_load_explicit(&g_inflight_max, memory_order_relaxed);
        while (cur > mx &&
               !atomic_compare_exchange_weak_explicit(&g_inflight_max, &mx, cur,
                                                      memory_order_relaxed, memory_order_relaxed)) {
        }
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
    atomic_fetch_sub_explicit(&g_inflight, 1, memory_order_relaxed);

    // g_demand_pending tracks outstanding demand-fetch chunks so the prefetch
    // IO thread can yield the bus to them; decrement once per miss, on its
    // LAST chunk (n_chunks==1 -- the common case -- always fires this).
    if (e->miss_chunks_done) {
        const int chunks_done = atomic_fetch_add_explicit(&e->miss_chunks_done[miss_idx], 1, memory_order_acq_rel) + 1;
        if (chunks_done == n_chunks) {
            atomic_fetch_sub_explicit(&g_demand_pending, 1, memory_order_release);
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

// GGML_MOE_STREAM_HITFIRST support (ggml-cpu.c): was `expert_id` a hit in the
// plan() call this call's miss_expert_id[]/n_misses describe? Read-only, no
// lock -- called only in the window between that plan()'s barrier and the
// next plan() (same contract as ggml_moe_stream_n_misses/slab above), by
// which time miss_expert_id/n_misses are frozen until the next plan() call.
// n_misses is bounded by n_slots (typically <=16), so the linear scan is cheap.
bool ggml_moe_stream_expert_is_hit(const struct ggml_tensor * t, int expert_id) {
    struct moe_stream_entry * e = moe_stream_find_entry(t);
    if (!e) {
        return false;
    }
    for (int i = 0; i < e->n_misses; i++) {
        if (e->miss_expert_id[i] == expert_id) {
            return false;
        }
    }
    return true;
}
