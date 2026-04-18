#include "ggml.h"
#include "ggml-alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // [K, N, E] - experts
    struct ggml_tensor * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, 64, 64, 4);
    // [K, S, T] - input
    struct ggml_tensor * input   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 8, 1);
    // [S, T] - indices
    struct ggml_tensor * ids     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 1);

    ggml_set_param(ctx, experts);
    ggml_set_param(ctx, input);

    // [N, S, T]
    struct ggml_tensor * output = ggml_mul_mat_id(ctx, experts, input, ids);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    // In modern ggml, we use ggml_graph_compute_backward
    // But let's check the ops in the graph directly if we can.
    // We can use ggml_graph_backward to build the backward graph.
    
    struct ggml_cgraph * gb = ggml_graph_view(ctx, gf, 0, gf->n_nodes); // This is not quite right
    // Actually, let's just use the build_backward we have in ggml.h
    ggml_build_backward_expand(ctx, gf, NULL);

    bool found = false;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->op == GGML_OP_MUL_MAT_ID_BACK_SRC1) {
            printf("Found GGML_OP_MUL_MAT_ID_BACK_SRC1 in graph at node %d\n", i);
            found = true;
        }
    }

    if (!found) {
        printf("FAILED: GGML_OP_MUL_MAT_ID_BACK_SRC1 not found in graph\n");
    }

    ggml_free(ctx);
    return found ? 0 : 1;
}
