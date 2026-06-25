#pragma once
// server-moe-pager.h — demand-paged MoE expert memory advisor
//
// Phase C of the Eyla demand-paged inference engine.
// Exposes a UNIX socket that the TypeScript scheduler connects to in order to
// call posix_madvise on specific expert weight pages in the mmap'd model file.
//
// Socket path:  /tmp/aditurbo-pager.sock
// Protocol:     newline-terminated ASCII commands
//   LAYOUT\n          → JSON list of {layer, expert, addrs, sizes}
//   EVICT L E\n       → MADV_DONTNEED on expert E of layer L (3 fused tensors)
//   PREFETCH L E\n    → MADV_WILLNEED on expert E of layer L
//   SNAPSHOT\n        → record current OS page residency as baseline; returns "OK\n"
//   DIFF\n            → return experts newly resident since last SNAPSHOT;
//                       format: "DIFF L E L E ...\n" (pairs); updates baseline
//   STATS\n           → JSON hit/miss stats
//   QUIT\n            → close this connection
//
// SNAPSHOT/DIFF usage (real activation detection, replaces synthetic activations):
//   Send SNAPSHOT before each decode step, DIFF after → real expert access data.
//
// Requires: model loaded with -cmoe (MoE FFN tensors in CPU/mmap backend)

#include "llama.h"
#include <cstdint>

struct moe_expert_region {
    void *   down_ptr;   size_t down_size;
    void *   gate_ptr;   size_t gate_size;
    void *   up_ptr;     size_t up_size;
};

// Call once after model load. Builds the expert layout map from tensor data pointers.
// Returns false if the model has no MoE FFN tensors (dense model) or tensors are
// not in CPU-accessible memory (forgot -cmoe).
bool moe_pager_init(const struct llama_context * ctx);

// Start the control socket in a background thread.
// sock_path defaults to /tmp/aditurbo-pager.sock.
void moe_pager_start(const char * sock_path = "/tmp/aditurbo-pager.sock");

// Stop the control socket thread (called on server shutdown).
void moe_pager_stop();
