#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"

#include <cmath>

namespace qwen3_tts {

struct ggml_cgraph * transformer_internal::ops::build_prefill_forward_graph(TTSTransformer & self, int32_t n_tokens, int32_t n_past) {
    auto & impl = self.impl_;
    const auto & cfg = impl->model.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_layers;

    struct ggml_init_params params = {
        /*.mem_size   =*/ impl->state.compute_meta.size(),
        /*.mem_buffer =*/ impl->state.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_prefill_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(inp_prefill_embd, "inp_prefill_embd");
    ggml_set_input(inp_prefill_embd);

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    struct ggml_tensor * inp_mrope_pos = nullptr;
    if (cfg.use_mrope) {
        inp_mrope_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4 * n_tokens);
        ggml_set_name(inp_mrope_pos, "inp_mrope_pos");
        ggml_set_input(inp_mrope_pos);
    }

    struct ggml_tensor * cur = inp_prefill_embd;
    struct ggml_tensor * inpL = cur;

    const float KQscale = 1.0f / sqrtf((float) head_dim);
    int mrope_sections[GGML_MROPE_SECTIONS] = { cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2], 0 };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = impl->model.layers[il];

        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);

        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);

        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }

        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }

        if (cfg.use_mrope) {
            Qcur = ggml_rope_multi(ctx0, Qcur, inp_mrope_pos, nullptr,
                                   head_dim, mrope_sections, GGML_ROPE_TYPE_MROPE, 0,
                                   rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            Kcur = ggml_rope_multi(ctx0, Kcur, inp_mrope_pos, nullptr,
                                   head_dim, mrope_sections, GGML_ROPE_TYPE_MROPE, 0,
                                   rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        } else {
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                                 head_dim, GGML_ROPE_TYPE_NEOX, 0,
                                 rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                                 head_dim, GGML_ROPE_TYPE_NEOX, 0,
                                 rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }

        struct ggml_tensor * k_cache = impl->state.cache.k_cache[il];
        struct ggml_tensor * v_cache = impl->state.cache.v_cache[il];

        struct ggml_tensor * k_cache_2d = ggml_view_2d(ctx0, k_cache, head_dim * n_kv_head, impl->state.cache.n_ctx, k_cache->nb[2], 0);
        struct ggml_tensor * v_cache_2d = ggml_view_2d(ctx0, v_cache, head_dim * n_kv_head, impl->state.cache.n_ctx, v_cache->nb[2], 0);

        struct ggml_tensor * Kcur_2d = ggml_view_2d(ctx0, Kcur, head_dim * n_kv_head, n_tokens, Kcur->nb[2], 0);
        struct ggml_tensor * Vcur_2d = ggml_view_2d(ctx0, Vcur, head_dim * n_kv_head, n_tokens, Vcur->nb[2], 0);

        struct ggml_tensor * k_updated = ggml_set_rows(ctx0, k_cache_2d, Kcur_2d, inp_pos);
        struct ggml_tensor * v_updated = ggml_set_rows(ctx0, v_cache_2d, Vcur_2d, inp_pos);

        ggml_build_forward_expand(gf, k_updated);
        ggml_build_forward_expand(gf, v_updated);

        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, impl->state.cache.n_ctx,
            k_cache->nb[1], k_cache->nb[2], 0);

        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, impl->state.cache.n_ctx,
            v_cache->nb[1], v_cache->nb[2], 0);

        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);

        if (n_tokens == 1) {
            struct ggml_tensor * KQ_mask = ggml_get_tensor(ctx0, "inp_mask");
            struct ggml_tensor * KQV_fa = ggml_flash_attn_ext(ctx0, Q, K, V, KQ_mask,
                                                              1.0f / sqrtf((float) head_dim), 0.0f, 0.0f);
            cur = ggml_cont_2d(ctx0, KQV_fa, n_head * head_dim, n_tokens);
        } else {
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_scale(ctx0, KQ, KQscale);
            KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
            KQ = ggml_soft_max(ctx0, KQ);

            V = ggml_cont(ctx0, ggml_transpose(ctx0, V));

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);
        }

        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;

        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);

        gate = ggml_silu(ctx0, gate);
        cur = ggml_mul(ctx0, gate, up);

        struct ggml_tensor * ffn_down_f32 = ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32);
        cur = ggml_mul_mat(ctx0, ffn_down_f32, cur);

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, impl->model.output_norm);
    ggml_set_name(cur, "hidden_states");
    ggml_set_output(cur);

    struct ggml_tensor * logits = ggml_mul_mat(ctx0, impl->model.codec_head, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph * transformer_internal::ops::build_step_graph_impl(TTSTransformer & self, int32_t n_past, struct ggml_context ** graph_ctx_out) {
    auto & impl = self.impl_;
    const auto & cfg = impl->model.config;
    const int n_head = cfg.n_attention_heads;
    const int n_kv_head = cfg.n_key_value_heads;
    const int head_dim = cfg.head_dim;
    const int hidden_size = cfg.hidden_size;
    const float eps = cfg.rms_norm_eps;
    const float rope_theta = cfg.rope_theta;
    const int n_layer = cfg.n_layers;
    const int n_tokens = 1;

    struct ggml_init_params params = {
        /*.mem_size   =*/ impl->state.compute_meta.size(),
        /*.mem_buffer =*/ impl->state.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_step_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, 1);
    ggml_set_name(inp_step_embd, "inp_step_embd");
    ggml_set_input(inp_step_embd);

    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    struct ggml_tensor * inp_mrope_pos = nullptr;
    if (cfg.use_mrope) {
        inp_mrope_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4);
        ggml_set_name(inp_mrope_pos, "inp_mrope_pos");
        ggml_set_input(inp_mrope_pos);
    }

    struct ggml_tensor * inp_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, impl->state.cache.n_ctx, 1);
    ggml_set_name(inp_mask, "inp_mask");
    ggml_set_input(inp_mask);

    struct ggml_tensor * cur = inp_step_embd;
    struct ggml_tensor * inpL = cur;

    const float KQscale = 1.0f / sqrtf((float) head_dim);
    int mrope_sections[GGML_MROPE_SECTIONS] = { cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2], 0 };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = impl->model.layers[il];

        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);

        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);

        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }

        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }

        if (cfg.use_mrope) {
            Qcur = ggml_rope_multi(ctx0, Qcur, inp_mrope_pos, nullptr,
                                   head_dim, mrope_sections, GGML_ROPE_TYPE_MROPE, 0,
                                   rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            Kcur = ggml_rope_multi(ctx0, Kcur, inp_mrope_pos, nullptr,
                                   head_dim, mrope_sections, GGML_ROPE_TYPE_MROPE, 0,
                                   rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        } else {
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                                 head_dim, GGML_ROPE_TYPE_NEOX, 0,
                                 rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                                 head_dim, GGML_ROPE_TYPE_NEOX, 0,
                                 rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }

        struct ggml_tensor * k_cache = impl->state.cache.k_cache[il];
        struct ggml_tensor * v_cache = impl->state.cache.v_cache[il];

        struct ggml_tensor * k_cache_2d = ggml_view_2d(ctx0, k_cache, head_dim * n_kv_head, impl->state.cache.n_ctx, k_cache->nb[2], 0);
        struct ggml_tensor * v_cache_2d = ggml_view_2d(ctx0, v_cache, head_dim * n_kv_head, impl->state.cache.n_ctx, v_cache->nb[2], 0);

        struct ggml_tensor * Kcur_2d = ggml_view_2d(ctx0, Kcur, head_dim * n_kv_head, n_tokens, Kcur->nb[2], 0);
        struct ggml_tensor * Vcur_2d = ggml_view_2d(ctx0, Vcur, head_dim * n_kv_head, n_tokens, Vcur->nb[2], 0);

        struct ggml_tensor * k_updated = ggml_set_rows(ctx0, k_cache_2d, Kcur_2d, inp_pos);
        struct ggml_tensor * v_updated = ggml_set_rows(ctx0, v_cache_2d, Vcur_2d, inp_pos);

        ggml_build_forward_expand(gf, k_updated);
        ggml_build_forward_expand(gf, v_updated);

        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, impl->state.cache.n_ctx,
            k_cache->nb[1], k_cache->nb[2], 0);

        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, impl->state.cache.n_ctx,
            v_cache->nb[1], v_cache->nb[2], 0);

        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);

        if (n_tokens == 1) {
            struct ggml_tensor * KQ_mask = ggml_get_tensor(ctx0, "inp_mask");
            struct ggml_tensor * KQV_fa = ggml_flash_attn_ext(ctx0, Q, K, V, KQ_mask,
                                                              1.0f / sqrtf((float) head_dim), 0.0f, 0.0f);
            cur = ggml_cont_2d(ctx0, KQV_fa, n_head * head_dim, n_tokens);
        } else {
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_scale(ctx0, KQ, KQscale);
            KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
            KQ = ggml_soft_max(ctx0, KQ);

            V = ggml_cont(ctx0, ggml_transpose(ctx0, V));

            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            cur = ggml_cont_2d(ctx0, KQV, n_head * head_dim, n_tokens);
        }

        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;

        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);

        gate = ggml_silu(ctx0, gate);
        cur = ggml_mul(ctx0, gate, up);

        struct ggml_tensor * ffn_down_f32 = ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32);
        cur = ggml_mul_mat(ctx0, ffn_down_f32, cur);

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, impl->model.output_norm);
    ggml_set_name(cur, "hidden_states");
    ggml_set_output(cur);

    struct ggml_tensor * logits = ggml_mul_mat(ctx0, impl->model.codec_head, cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);

    if (graph_ctx_out) {
        *graph_ctx_out = ctx0;
    } else {
        ggml_free(ctx0);
    }
    return gf;
}

struct ggml_cgraph * transformer_internal::ops::build_step_graph(TTSTransformer & self, int32_t n_past) {
    return build_step_graph_impl(self, n_past, nullptr);
}

} // namespace qwen3_tts
