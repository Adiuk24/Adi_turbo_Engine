// server-moe-pager.cpp — demand-paged MoE expert memory advisor
//
// See server-moe-pager.h for protocol documentation.
//
// How it works
// ------------
// With -cmoe, all MoE FFN expert tensors are placed in the CPU memory backend.
// For mmap'd models (default), this means tensor->data points into the process's
// virtual address space backed by the GGUF file on disk.
//
// Expert e in a fused tensor of shape [ne0, ne1, 128]:
//   ptr  = (char*)tensor->data + e * tensor->nb[2]
//   size = tensor->nb[2]          (= nb[1] * ne[1])
//
// nb[2] is the byte stride of the expert dimension (verified against Python splitter:
//   down_exps: 1,081,344 bytes/expert  gate_exps/up_exps: 675,840 bytes/expert)
//
// madvise calls are page-rounded and issued on the three fused tensors per expert.

#include "server-moe-pager.h"

#include "llama.h"
#include "../../src/llama-model.h"    // internal: llama_model, llama_layer structs
#include "log.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#  include <sys/mman.h>
#  include <sys/socket.h>
#  include <sys/un.h>
#  include <unistd.h>
#  define MOE_PAGER_SUPPORTED 1
#else
#  define MOE_PAGER_SUPPORTED 0
#endif

// ── Layout ───────────────────────────────────────────────────────────────────

struct moe_layout {
    int n_layer   = 0;
    int n_experts = 0;
    std::vector<std::vector<moe_expert_region>> regions;  // [layer][expert]
};

static moe_layout g_layout;
static std::atomic<bool>  g_running{false};
static std::thread        g_thread;
static std::string        g_sock_path;

// ── Stats ─────────────────────────────────────────────────────────────────────

static std::atomic<uint64_t> g_evict_calls{0};
static std::atomic<uint64_t> g_prefetch_calls{0};

// ── Hint mode (set via PAGER_HINT_MODE env var before server start) ───────────
//   A0 = 0 : no madvise hints (pure demand-fault baseline)
//   A1 = 1 : MADV_WILLNEED only  (prefetch, no eviction)
//   A2 = 2 : MADV_DONTNEED only  (eviction, no prefetch)
//   A3 = 3 : both WILLNEED + DONTNEED (default, production setting)
static int g_hint_mode = 3;

// ── Residency snapshot (SNAPSHOT / DIFF commands) ─────────────────────────────
//
// Each layer's gate tensor covers all n_experts contiguously:
//   expert e starts at: regions[l][0].gate_ptr + e * gate_stride
// One mincore() call per layer samples the whole layer's gate pages.
// Snapshot stores one bit per (layer, expert) pair.

#if MOE_PAGER_SUPPORTED

static std::vector<uint8_t>  g_snap_bits;      // packed bitmask: bit[l*N+e] = resident at snapshot time
static std::vector<std::vector<unsigned char>> g_mc_buf;   // mincore page buffers, one per layer

static void snap_init() {
    if (!g_snap_bits.empty()) return;
    const int N = g_layout.n_experts;
    const int L = g_layout.n_layer;
    g_snap_bits.assign(((size_t)L * N + 7) / 8, 0);
    g_mc_buf.resize(L);
    static const long PS = sysconf(_SC_PAGESIZE);
    for (int l = 0; l < L; l++) {
        if ((int)g_layout.regions[l].size() == 0 || !g_layout.regions[l][0].gate_ptr) continue;
        size_t total = (size_t)N * g_layout.regions[l][0].gate_size;
        uintptr_t base = (uintptr_t)g_layout.regions[l][0].gate_ptr & ~((uintptr_t)(PS - 1));
        size_t align   = (uintptr_t)g_layout.regions[l][0].gate_ptr - base;
        g_mc_buf[l].resize((total + align + PS - 1) / PS, 0);
    }
}

static void snap_sample_layer(int l) {
    if (l >= (int)g_mc_buf.size() || g_mc_buf[l].empty()) return;
    static const long PS = sysconf(_SC_PAGESIZE);
    void * base = (void *)((uintptr_t)g_layout.regions[l][0].gate_ptr & ~((uintptr_t)(PS - 1)));
    ::mincore(base, g_mc_buf[l].size() * (size_t)PS,
              reinterpret_cast<char *>(g_mc_buf[l].data()));
}

static bool snap_expert_resident(int l, int e) {
    if (l >= (int)g_mc_buf.size() || g_mc_buf[l].empty()) return false;
    static const long PS = sysconf(_SC_PAGESIZE);
    size_t stride = g_layout.regions[l][0].gate_size;
    uintptr_t base = (uintptr_t)g_layout.regions[l][0].gate_ptr & ~((uintptr_t)(PS - 1));
    size_t align   = (uintptr_t)g_layout.regions[l][0].gate_ptr - base;
    size_t page_idx = (align + (size_t)e * stride) / (size_t)PS;
    if (page_idx >= g_mc_buf[l].size()) return false;
    return (g_mc_buf[l][page_idx] & 1) != 0;
}

static inline bool snap_get(int l, int e) {
    size_t i = (size_t)l * g_layout.n_experts + e;
    return (g_snap_bits[i / 8] >> (i % 8)) & 1;
}
static inline void snap_set(int l, int e, bool v) {
    size_t i = (size_t)l * g_layout.n_experts + e;
    if (v) g_snap_bits[i / 8] |=  (uint8_t)(1 << (i % 8));
    else   g_snap_bits[i / 8] &= ~(uint8_t)(1 << (i % 8));
}

// SNAPSHOT — save current page residency as baseline for next DIFF
static std::string cmd_snapshot() {
    snap_init();
    for (int l = 0; l < g_layout.n_layer; l++) {
        snap_sample_layer(l);
        for (int e = 0; e < g_layout.n_experts; e++)
            snap_set(l, e, snap_expert_resident(l, e));
    }
    return "OK\n";
}

// DIFF — report experts that became newly resident since last SNAPSHOT
// Returns: "DIFF L E L E ...\n" (space-separated layer/expert pairs)
// Side effect: updates snapshot to current state
static std::string cmd_diff() {
    snap_init();
    std::string out = "DIFF";
    for (int l = 0; l < g_layout.n_layer; l++) {
        snap_sample_layer(l);
        for (int e = 0; e < g_layout.n_experts; e++) {
            bool was = snap_get(l, e);
            bool now = snap_expert_resident(l, e);
            if (now && !was) {
                out += ' '; out += std::to_string(l);
                out += ' '; out += std::to_string(e);
            }
            snap_set(l, e, now);
        }
    }
    out += '\n';
    return out;
}

#else

static std::string cmd_snapshot() { return "OK\n"; }
static std::string cmd_diff()     { return "DIFF\n"; }

#endif

// ── madvise helpers ───────────────────────────────────────────────────────────

#if MOE_PAGER_SUPPORTED

static void advise_region(void * ptr, size_t size, int advice) {
    if (!ptr || size == 0) return;
    // madvise requires page-aligned pointer + length
    static const size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    uintptr_t start = (uintptr_t)ptr & ~(page_size - 1);
    uintptr_t end   = ((uintptr_t)ptr + size + page_size - 1) & ~(page_size - 1);
    madvise(reinterpret_cast<void*>(start), end - start, advice);
}

static void evict_expert(int layer, int expert) {
    if (layer < 0 || layer >= g_layout.n_layer)   return;
    if (expert < 0 || expert >= g_layout.n_experts) return;
    g_evict_calls++;
    // A0 or A1: DONTNEED suppressed
    if (g_hint_mode != 2 && g_hint_mode != 3) return;
    const auto & r = g_layout.regions[layer][expert];
    advise_region(r.down_ptr, r.down_size, MADV_DONTNEED);
    advise_region(r.gate_ptr, r.gate_size, MADV_DONTNEED);
    advise_region(r.up_ptr,   r.up_size,   MADV_DONTNEED);
}

static void prefetch_expert(int layer, int expert) {
    if (layer < 0 || layer >= g_layout.n_layer)   return;
    if (expert < 0 || expert >= g_layout.n_experts) return;
    g_prefetch_calls++;
    // A0 or A2: WILLNEED suppressed
    if (g_hint_mode != 1 && g_hint_mode != 3) return;
    const auto & r = g_layout.regions[layer][expert];
    advise_region(r.down_ptr, r.down_size, MADV_WILLNEED);
    advise_region(r.gate_ptr, r.gate_size, MADV_WILLNEED);
    advise_region(r.up_ptr,   r.up_size,   MADV_WILLNEED);
}

#else  // !MOE_PAGER_SUPPORTED

static void evict_expert(int, int)   {}
static void prefetch_expert(int, int) {}

#endif

// ── Request handler ───────────────────────────────────────────────────────────

static std::string handle_command(const std::string & line) {
    if (line == "STATS") {
        char buf[256];
        snprintf(buf, sizeof(buf),
            "{\"evict_calls\":%lu,\"prefetch_calls\":%lu,\"n_layer\":%d,\"n_experts\":%d}\n",
            (unsigned long)g_evict_calls.load(),
            (unsigned long)g_prefetch_calls.load(),
            g_layout.n_layer, g_layout.n_experts);
        return buf;
    }

    if (line == "LAYOUT") {
        std::string out = "[";
        for (int l = 0; l < g_layout.n_layer; l++) {
            for (int e = 0; e < g_layout.n_experts; e++) {
                if (l > 0 || e > 0) out += ",";
                const auto & r = g_layout.regions[l][e];
                char buf[256];
                snprintf(buf, sizeof(buf),
                    "{\"l\":%d,\"e\":%d,\"ds\":%zu,\"gs\":%zu,\"us\":%zu}",
                    l, e, r.down_size, r.gate_size, r.up_size);
                out += buf;
            }
        }
        out += "]\n";
        return out;
    }

    if (line.substr(0, 6) == "EVICT ") {
        int l = 0, e = 0;
        if (sscanf(line.c_str() + 6, "%d %d", &l, &e) == 2) {
            evict_expert(l, e);
            return "OK\n";
        }
        return "ERR bad EVICT args\n";
    }

    if (line.substr(0, 9) == "PREFETCH ") {
        int l = 0, e = 0;
        if (sscanf(line.c_str() + 9, "%d %d", &l, &e) == 2) {
            prefetch_expert(l, e);
            return "OK\n";
        }
        return "ERR bad PREFETCH args\n";
    }

    if (line == "SNAPSHOT") return cmd_snapshot();
    if (line == "DIFF")     return cmd_diff();

    return "ERR unknown command\n";
}

// ── Socket server ─────────────────────────────────────────────────────────────

#if MOE_PAGER_SUPPORTED

static void pager_thread_fn(const std::string & sock_path) {
    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        LOG_WRN("%s: socket() failed: %s\n", __func__, strerror(errno));
        return;
    }

    unlink(sock_path.c_str());

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        LOG_WRN("%s: bind() failed: %s\n", __func__, strerror(errno));
        close(server_fd);
        return;
    }

    listen(server_fd, 4);
    LOG_INF("%s: MoE pager listening on %s\n", __func__, sock_path.c_str());

    while (g_running.load()) {
        // Use select() so we can check g_running periodically
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(server_fd, &rfds);
        struct timeval tv{1, 0};  // 1s timeout
        if (select(server_fd + 1, &rfds, nullptr, nullptr, &tv) <= 0) continue;

        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) continue;

        // Handle one client synchronously (scheduler is single-threaded in practice)
        char buf[256];
        std::string line;
        while (true) {
            ssize_t n = read(client_fd, buf, sizeof(buf));
            if (n <= 0) break;
            for (ssize_t i = 0; i < n; i++) {
                if (buf[i] == '\n') {
                    if (line == "QUIT") goto client_done;
                    std::string resp = handle_command(line);
                    write(client_fd, resp.c_str(), resp.size());
                    line.clear();
                } else {
                    line += buf[i];
                }
            }
        }
        client_done:
        close(client_fd);
    }

    close(server_fd);
    unlink(sock_path.c_str());
    LOG_INF("%s: MoE pager stopped\n", __func__);
}

#endif  // MOE_PAGER_SUPPORTED

// ── Public API ────────────────────────────────────────────────────────────────

bool moe_pager_init(const struct llama_context * ctx) {
    if (!ctx) return false;

    const llama_model * model = llama_get_model(ctx);
    if (!model) return false;

    const int n_layer = (int)model->layers.size();
    if (n_layer == 0) return false;

    // Check if this model has fused MoE expert tensors
    // Use layer 0 as a probe
    const auto & l0 = model->layers[0];
    if (!l0.ffn_down_exps || !l0.ffn_gate_exps || !l0.ffn_up_exps) {
        LOG_INF("%s: no MoE expert tensors found (dense model or wrong arch)\n", __func__);
        return false;
    }

    // Verify tensors are CPU-accessible (required for madvise)
    if (!l0.ffn_down_exps->data) {
        LOG_WRN("%s: ffn_down_exps->data is null — did you forget -cmoe?\n", __func__);
        return false;
    }

    // Qwen3-30B-A3B: ne[2] = 128 experts
    const int n_experts = (int)l0.ffn_down_exps->ne[2];
    if (n_experts <= 0 || n_experts > 4096) {
        LOG_WRN("%s: unexpected n_experts=%d\n", __func__, n_experts);
        return false;
    }

    g_layout.n_layer   = n_layer;
    g_layout.n_experts = n_experts;
    g_layout.regions.resize(n_layer, std::vector<moe_expert_region>(n_experts));

    for (int l = 0; l < n_layer; l++) {
        const auto & layer = model->layers[l];

        // Only layers with fused expert tensors (some models mix dense + MoE)
        if (!layer.ffn_down_exps || !layer.ffn_gate_exps || !layer.ffn_up_exps) continue;
        if (!layer.ffn_down_exps->data) continue;

        // nb[2] = per-expert byte stride (expert is ne[2], the outermost dimension)
        const size_t down_stride = (size_t)layer.ffn_down_exps->nb[2];
        const size_t gate_stride = (size_t)layer.ffn_gate_exps->nb[2];
        const size_t up_stride   = (size_t)layer.ffn_up_exps->nb[2];

        for (int e = 0; e < n_experts; e++) {
            auto & r = g_layout.regions[l][e];
            r.down_ptr  = (char*)layer.ffn_down_exps->data + (size_t)e * down_stride;
            r.down_size = down_stride;
            r.gate_ptr  = (char*)layer.ffn_gate_exps->data + (size_t)e * gate_stride;
            r.gate_size = gate_stride;
            r.up_ptr    = (char*)layer.ffn_up_exps->data   + (size_t)e * up_stride;
            r.up_size   = up_stride;
        }
    }

    const auto & e0 = g_layout.regions[0][0];
    LOG_INF("%s: MoE layout built: %d layers × %d experts, "
            "~%.3f MB/expert (down=%zu B gate=%zu B up=%zu B)\n",
            __func__, n_layer, n_experts,
            (double)(e0.down_size + e0.gate_size + e0.up_size) / 1024.0 / 1024.0,
            e0.down_size, e0.gate_size, e0.up_size);

    // Read ablation mode (E3 experiment: PAGER_HINT_MODE=0..3)
    if (const char * m = getenv("PAGER_HINT_MODE")) {
        int v = atoi(m);
        g_hint_mode = (v >= 0 && v <= 3) ? v : 3;
    }
    static const char * mode_names[] = {"no hints", "WILLNEED only", "DONTNEED only", "WILLNEED+DONTNEED"};
    LOG_INF("%s: hint mode A%d (%s)\n", __func__, g_hint_mode, mode_names[g_hint_mode]);

#if MOE_PAGER_SUPPORTED
    // Pre-allocate snapshot / mincore buffers so SNAPSHOT/DIFF are ready immediately
    snap_init();
#endif

    return true;
}

void moe_pager_start(const char * sock_path) {
#if MOE_PAGER_SUPPORTED
    if (g_layout.n_layer == 0) {
        LOG_WRN("%s: pager not started — call moe_pager_init() first\n", __func__);
        return;
    }
    g_running.store(true);
    g_sock_path = sock_path;
    g_thread = std::thread(pager_thread_fn, g_sock_path);
#else
    (void)sock_path;
    LOG_WRN("%s: MoE pager not supported on this platform\n", __func__);
#endif
}

void moe_pager_stop() {
    g_running.store(false);
#if MOE_PAGER_SUPPORTED
    if (g_thread.joinable()) {
        g_thread.join();
    }
#endif
}
