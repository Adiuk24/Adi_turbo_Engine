#include "models.h"

#include "llama-memory-recurrent.h"

// Noor-Edge: 24-layer Gated-DeltaNet(18)/GQA(6) 3:1 hybrid + MoE.
// Ground truth: /Users/adi/noor/noor_hybrid/mac_infer.py (oracle-verified against the
// H200 training run). Converter: noor_hybrid/convert_noor_gguf.py.
//
// The checkpoint's GDN layers are 12 heads, head_dim=128 for q/k but head_dim=256 for v
// (expand_v=2.0, so the per-head recurrent state is the RECTANGULAR [128,256] matrix
// naive_recurrent_gated_delta_rule builds). The in-tree delta-net kernel
// (delta-net-base.cpp's build_delta_net_{chunking,autoregressive,fused}) hard-asserts
// S_k==S_v (a SQUARE per-head state) -- true for every arch that currently reuses it
// (qwen3next/kimi-linear/qwen35 all have head_dim_k==head_dim_v). Rather than touch that
// shared kernel, the converter repacks the checkpoint into 24 heads of width 128: each
// original head's low/high 128-wide v-halves become two square-state sub-heads that
// share the SAME q/k/beta/gate as the original head (mathematically exact -- the delta
// rule's state update is independent per value-column, so splitting the value axis
// changes nothing as long as q/k/beta/gate are duplicated, not split). See
// convert_noor_gguf.py's `repack_gdn_heads` for the exact byte-for-byte derivation.

// causal_conv1d -- one of Q/K/V's [proj -> causal depthwise conv1d -> silu], with its own
// slice of the layer's shared recurrent-state conv buffer. Adapted from
// kimi-linear.cpp's helper of the same name/purpose; Noor's three convs are uniformly
// d_inner=n_head_gdn*head_dim_gdn wide (unlike Kimi's KDA), so this is called identically
// for qkv_idx 0/1/2 with the SAME d_inner/head_dim/n_head every time.
static ggml_tensor * causal_conv1d(
        ggml_cgraph * gf, ggml_context * ctx0,
        ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
        int64_t qkv_idx, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
        int64_t d_conv, int64_t head_dim, int64_t n_head,
        int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total = 3 * conv_state_size; // Q + K + V

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1) * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv_idx * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);

    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_3d(ctx0, conv_states_all,
                d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_states_all),
                n_embd_r_total * ggml_element_size(conv_states_all),
                (kv_head * n_embd_r_total + qkv_idx * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

llm_build_noor::llm_build_noor(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_embd", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        // sandwich pre-norm
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            cur = build_layer_delta_net(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // sandwich post-norm (depth_scale folded into this weight by the converter), then residual add
        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        cur = build_norm(sa_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, sa_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// GQA full-attn layer (6 of 24): plain bias-qkv, no bias on o, no qk-norm, causal SDPA,
// RoPE over the FULL head_dim (NeoX/rotate-half; n_rot defaults to n_embd_head_k since
// the converter never writes a partial rope_dimension_count). Mirrors qwen2.cpp/
// gemma2-iswa.cpp's plain attention block exactly -- Noor's full-attn layer has none of
// qwen3next's extras (no fused q+gate, no q/k RMSNorm).
ggml_tensor * llm_build_noor::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
    if (model.layers[il].bq) {
        Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
    }
    cb(Qcur, "Qcur", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    if (model.layers[il].bk) {
        Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
    }
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    if (model.layers[il].bv) {
        Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
    }
    cb(Vcur, "Vcur", il);

    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = ggml_rope_ext(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    Kcur = ggml_rope_ext(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    cur = build_attn(inp,
            model.layers[il].wo, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);

    return cur;
}

// Gated DeltaNet layer (18 of 24): separate depthwise Q/K/V causal convs (kernel 4, no
// bias) + SiLU; q/k L2-normalized (eps sits inside the sqrt in the oracle -- ggml_l2_norm
// uses max(norm,eps) instead, immaterial for any non-degenerate activation); beta =
// sigmoid(b_proj); gate = -exp(A_log) * softplus(a_proj + dt_bias) (ssm_a already holds
// -exp(A_log), precomputed by the converter); decay-first delta rule via the shared
// build_delta_net kernel (scale 1/sqrt(head_dim) is applied inside it); output =
// RMSNormGated(o, eps 1e-5) * silu(g_proj) per (repacked) head, then o_proj. All 24
// repacked heads carry head_dim=128 uniformly (q/k duplicated, v/gate reinterpreted, see
// the file header comment) so the delta-net kernel's S_k==S_v assumption always holds.
ggml_tensor * llm_build_noor::build_layer_delta_net(
        llm_graph_input_rs * inp,
        ggml_tensor *         cur,
        int                   il) {
    const auto * mctx_cur = inp->mctx;
    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t head_dim = hparams.ssm_d_state;  // 128, repacked
    const int64_t n_head_g = hparams.ssm_n_group;  // 24,  repacked
    const int64_t d_conv   = hparams.ssm_d_conv;   // 4

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_state_all  = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);

    ggml_tensor * Qcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur,
            model.layers[il].wq, model.layers[il].ssm_q_conv, d_conv, head_dim, n_head_g,
            n_seq_tokens, n_seqs, n_tokens, kv_head);
    ggml_tensor * Kcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur,
            model.layers[il].wk, model.layers[il].ssm_k_conv, d_conv, head_dim, n_head_g,
            n_seq_tokens, n_seqs, n_tokens, kv_head);
    ggml_tensor * Vcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur,
            model.layers[il].wv, model.layers[il].ssm_v_conv, d_conv, head_dim, n_head_g,
            n_seq_tokens, n_seqs, n_tokens, kv_head);

    const float eps_norm = hparams.f_norm_rms_eps;
    Qcur = ggml_l2_norm(ctx0, Qcur, eps_norm);
    Kcur = ggml_l2_norm(ctx0, Kcur, eps_norm);
    cb(Qcur, "Qcur_normed", il);
    cb(Kcur, "Kcur_normed", il);

    // beta = sigmoid(b_proj(cur))
    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur); // [n_head_g, n_tokens]
    beta = ggml_sigmoid(ctx0, beta);
    beta = ggml_reshape_4d(ctx0, beta, 1, n_head_g, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    // gate = softplus(a_proj(cur) + dt_bias) * ssm_a, where ssm_a == -exp(A_log) (precomputed)
    ggml_tensor * gate = build_lora_mm(model.layers[il].ssm_alpha, cur); // [n_head_g, n_tokens]
    gate = ggml_add(ctx0, gate, model.layers[il].ssm_dt);
    gate = ggml_softplus(ctx0, gate);
    gate = ggml_mul(ctx0, gate, model.layers[il].ssm_a);
    gate = ggml_reshape_4d(ctx0, gate, 1, n_head_g, n_seq_tokens, n_seqs);
    cb(gate, "gate", il);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head_g, n_seqs);

    auto attn_out = build_delta_net(Qcur, Kcur, Vcur, gate, beta, state, il);
    ggml_tensor * output    = ggml_cont(ctx0, attn_out.first); // [head_dim, n_head_g, n_tokens, n_seqs]
    ggml_tensor * new_state = attn_out.second;
    cb(output, "delta_net_out", il);

    ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, new_state,
                ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                    kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    // output gate: RMSNormGated(o, eps=1e-5) * silu(g_proj(cur)) -- a direct ggml_rms_norm
    // call (NOT build_norm) because this eps (1e-5) differs from the shared
    // hparams.f_norm_rms_eps (1e-6) used by every other norm in this model. ssm_norm is
    // pre-tiled [head_dim, n_head_g] by the converter (the oracle's single 256-wide
    // weight, alternating low/high 128-halves across the 24 repacked heads) so this is a
    // plain elementwise multiply, no broadcast ambiguity.
    ggml_tensor * gproj = build_lora_mm(model.layers[il].wqkv_gate, cur); // [head_dim*n_head_g, n_tokens]
    gproj = ggml_reshape_4d(ctx0, gproj, head_dim, n_head_g, n_seq_tokens, n_seqs);

    ggml_tensor * normed = ggml_rms_norm(ctx0, output, 1e-5f);
    normed = ggml_mul(ctx0, normed, model.layers[il].ssm_norm);
    ggml_tensor * gated  = ggml_mul(ctx0, normed, ggml_silu(ctx0, gproj));
    cb(gated, "o_norm_gated", il);

    gated = ggml_cont_2d(ctx0, gated, head_dim * n_head_g, n_tokens);
    cur = build_lora_mm(model.layers[il].wo, gated);
    cb(cur, "delta_net_out_proj", il);

    return cur;
}

// MoE FFN (every layer): softmax-then-top2 router with RAW (unrenormalized) weights,
// non-gated GELU-tanh experts (down_exps written pre-transposed by the converter so the
// standard build_lora_mm_id convention reproduces megablocks' "no transpose on w2"
// exactly), plus an always-on ungated shared SwiGLU added unconditionally (deepseek2-style
// plain add -- Noor's shared expert has no external gate at all).
ggml_tensor * llm_build_noor::build_layer_ffn(ggml_tensor * cur, const int il) {
    ggml_tensor * moe_out = build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            nullptr, // no gate_exps: experts are non-gated (plain 2-matrix GELU-tanh FFN)
            model.layers[il].ffn_down_exps,
            nullptr, // no expert-selection bias
            n_expert, n_expert_used,
            LLM_FFN_GELU, /* norm_w */ false, /* w_scale */ 1.0f,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
    cb(moe_out, "ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            model.layers[il].ffn_up_shexp,   nullptr, nullptr,
            model.layers[il].ffn_gate_shexp, nullptr, nullptr,
            model.layers[il].ffn_down_shexp, nullptr, nullptr,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "ffn_shexp", il);

    return ggml_add(ctx0, moe_out, ffn_shexp);
}
