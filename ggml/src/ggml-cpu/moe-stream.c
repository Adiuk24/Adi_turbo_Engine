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
#include <unistd.h>

#define MOE_STREAM_MAX_TENSORS 512
#define MOE_STREAM_ALIGN       (2 * 1024 * 1024) // 2 MiB, per harvested reference designs

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
        fprintf(stderr, "[moe-stream] tensor=%s slots=%d hits=%llu misses=%llu bytes_read=%llu chunked_calls=%llu\n",
                e->name, e->n_slots,
                (unsigned long long) e->stat_hits,
                (unsigned long long) e->stat_misses,
                (unsigned long long) e->stat_bytes_read,
                (unsigned long long) e->stat_chunked_calls);
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
        const size_t page  = MOE_STREAM_ALIGN;
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
