#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace qwen3_tts {

namespace {

bool packed_code_pred_qkv_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("QWEN3_TTS_CODE_PRED_PACKED_QKV");
        return !(value && value[0] == '0');
    }();
    return enabled;
}

bool is_cuda_backend(ggml_backend_t backend) {
    ggml_backend_dev_t device = backend ? ggml_backend_get_device(backend) : nullptr;
    const char * name = device ? ggml_backend_dev_name(device) : nullptr;
    return name && std::strstr(name, "CUDA") != nullptr;
}

} // namespace

struct ggml_cgraph * transformer_internal::ops::build_code_pred_graph(TTSTransformer & self, int32_t n_prev_codes) {
    auto & impl = self.impl_;
    const auto & cfg = impl->model.config;
    const int n_head = cfg.code_pred_n_attention_heads;
    const int n_kv_head = cfg.code_pred_n_key_value_heads;
    const int head_dim = cfg.code_pred_head_dim;
    const int hidden_size = cfg.code_pred_hidden_size;
    const int in_hidden_size = cfg.hidden_size;
    const float eps = cfg.code_pred_rms_norm_eps;
    const int n_layer = cfg.code_pred_layers;
    const int n_codebooks = cfg.n_codebooks;

    struct ggml_init_params params = {
        /*.mem_size   =*/ impl->state.compute_meta.size(),
        /*.mem_buffer =*/ impl->state.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, in_hidden_size);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);

    struct ggml_tensor * inp_prev_codes = nullptr;
    if (n_prev_codes > 0) {
        inp_prev_codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_prev_codes);
        ggml_set_name(inp_prev_codes, "inp_prev_codes");
        ggml_set_input(inp_prev_codes);
    }

    struct ggml_tensor * cur = ggml_reshape_2d(ctx0, inp_hidden, in_hidden_size, 1);

    if (n_prev_codes > 0 && inp_prev_codes) {
        for (int cb = 0; cb < n_prev_codes && cb < n_codebooks - 1; ++cb) {
            struct ggml_tensor * code_idx = ggml_view_1d(ctx0, inp_prev_codes, 1, cb * sizeof(int32_t));
            struct ggml_tensor * code_embd = ggml_get_rows(ctx0, impl->model.code_pred_embd[cb], code_idx);
            cur = ggml_add(ctx0, cur, code_embd);
        }
    }

    if (impl->model.code_pred_small_to_mtp_weight) {
        cur = ggml_mul_mat(ctx0, impl->model.code_pred_small_to_mtp_weight, cur);
        if (impl->model.code_pred_small_to_mtp_bias) {
            struct ggml_tensor * bias = ggml_repeat(
                ctx0, ggml_reshape_2d(ctx0, impl->model.code_pred_small_to_mtp_bias, hidden_size, 1), cur);
            cur = ggml_add(ctx0, cur, bias);
        }
    }

    struct ggml_tensor * inpL = cur;
    const float KQscale = 1.0f / sqrtf((float) head_dim);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = impl->model.code_pred_layers[il];

        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);

        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, 1);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, 1);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, 1);

        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }

        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }

        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
        struct ggml_tensor * V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);

        struct ggml_tensor * KQV_fa = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, KQscale, 0.0f, 0.0f);
        cur = ggml_cont_2d(ctx0, KQV_fa, n_head * head_dim, 1);
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        cur = ggml_clamp(ctx0, cur, -65504.0f, 65504.0f);
        struct ggml_tensor * inpFF = cur;

        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);

        cur = ggml_swiglu_split(ctx0, gate, up);

        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);

        inpL = ggml_add(ctx0, cur, inpFF);
        inpL = ggml_clamp(ctx0, inpL, -65504.0f, 65504.0f);
    }

    cur = inpL;
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, impl->model.code_pred_output_norm);

    std::vector<struct ggml_tensor *> all_logits;
    for (int cb = 0; cb < n_codebooks - 1; ++cb) {
        struct ggml_tensor * cb_logits = ggml_mul_mat(ctx0, impl->model.code_pred_head[cb], cur);
        ggml_format_name(cb_logits, "logits_cb%d", cb + 1);
        ggml_set_output(cb_logits);
        all_logits.push_back(cb_logits);
    }

    for (auto * logits : all_logits) {
        ggml_build_forward_expand(gf, logits);
    }

    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph * transformer_internal::ops::build_code_pred_prefill_graph(TTSTransformer & self,
                                                                              bool use_hidden_bridge) {
    auto & impl = self.impl_;
    const auto & cfg = impl->model.config;
    const int n_head = cfg.code_pred_n_attention_heads;
    const int n_kv_head = cfg.code_pred_n_key_value_heads;
    const int head_dim = cfg.code_pred_head_dim;
    const int hidden_size = cfg.code_pred_hidden_size;
    const int in_hidden_size = cfg.hidden_size;
    const float eps = cfg.code_pred_rms_norm_eps;
    const float rope_theta = cfg.code_pred_rope_theta;
    const int n_layer = cfg.code_pred_layers;
    const int n_tokens = 2;
    const int n_kv_ctx = impl->state.code_pred_cache.n_ctx;
    const bool use_code_pred_mrope = false;

    struct ggml_init_params params = {
        /*.mem_size   =*/ impl->state.code_pred_compute_meta[0].size(),
        /*.mem_buffer =*/ impl->state.code_pred_compute_meta[0].data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_CODE_PRED_MAX_NODES, false);

    const bool bridge_available = use_hidden_bridge && impl->state.hidden_bridge;
    struct ggml_tensor * inp_hidden = nullptr;
    if (!bridge_available) {
        inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, in_hidden_size);
        ggml_set_name(inp_hidden, "inp_hidden");
        ggml_set_input(inp_hidden);
    }

    struct ggml_tensor * inp_cb0_code = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_cb0_code, "inp_cb0_code");
    ggml_set_input(inp_cb0_code);

    struct ggml_tensor * inp_pos = impl->state.code_pred_prefill_pos;
    if (!inp_pos) {
        inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_name(inp_pos, "inp_pos");
        ggml_set_input(inp_pos);
    }

    struct ggml_tensor * inp_mask = impl->state.code_pred_prefill_mask_tensor;
    if (!inp_mask) {
        inp_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv_ctx, n_tokens);
        ggml_set_name(inp_mask, "inp_mask");
        ggml_set_input(inp_mask);
    }

    struct ggml_tensor * inp_mrope_pos = nullptr;
    if (use_code_pred_mrope) {
        inp_mrope_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4 * n_tokens);
        ggml_set_name(inp_mrope_pos, "inp_mrope_pos");
        ggml_set_input(inp_mrope_pos);
    }

    struct ggml_tensor * hidden_src = bridge_available ? impl->state.hidden_bridge : inp_hidden;
    struct ggml_tensor * hidden_2d = ggml_reshape_2d(ctx0, hidden_src, in_hidden_size, 1);
    struct ggml_tensor * cb0_2d = ggml_get_rows(ctx0, impl->model.codec_embd, inp_cb0_code);
    cb0_2d = ggml_reshape_2d(ctx0, cb0_2d, in_hidden_size, 1);
    struct ggml_tensor * cur = ggml_concat(ctx0, hidden_2d, cb0_2d, 1);

    if (impl->model.code_pred_small_to_mtp_weight) {
        cur = ggml_mul_mat(ctx0, impl->model.code_pred_small_to_mtp_weight, cur);
        if (impl->model.code_pred_small_to_mtp_bias) {
            struct ggml_tensor * bias = ggml_repeat(
                ctx0, ggml_reshape_2d(ctx0, impl->model.code_pred_small_to_mtp_bias, hidden_size, 1), cur);
            cur = ggml_add(ctx0, cur, bias);
        }
    }

    struct ggml_tensor * inpL = cur;
    const float KQscale = 1.0f / sqrtf((float) head_dim);
    int mrope_sections[GGML_MROPE_SECTIONS] = { cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2], 0 };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = impl->model.code_pred_layers[il];

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

        if (use_code_pred_mrope) {
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

        struct ggml_tensor * k_cache = impl->state.code_pred_cache.k_cache[il];
        struct ggml_tensor * v_cache = impl->state.code_pred_cache.v_cache[il];

        struct ggml_tensor * Kcache = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));
        struct ggml_tensor * Vcache = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 0, 2, 1, 3));

        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, k_cache, Kcache, inp_pos));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, v_cache, Vcache, inp_pos));

        struct ggml_tensor * Kfull = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_ctx, n_kv_head,
            k_cache->nb[1], k_cache->nb[2], 0);

        struct ggml_tensor * Vfull = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_ctx, n_kv_head,
            v_cache->nb[1], v_cache->nb[2], 0);

        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);

        struct ggml_tensor * KQV_fa = ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, inp_mask, KQscale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(KQV_fa, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx0, KQV_fa, n_head * head_dim, n_tokens);
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);
        struct ggml_tensor * inpFF = cur;

        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);

        cur = ggml_swiglu_split(ctx0, gate, up);

        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, impl->model.code_pred_output_norm);

    struct ggml_tensor * last_hidden = ggml_view_2d(ctx0, cur, hidden_size, 1,
                                                    cur->nb[1], hidden_size * sizeof(float));
    struct ggml_tensor * logits = ggml_mul_mat(ctx0, impl->model.code_pred_head[0], last_hidden);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph * transformer_internal::ops::build_code_pred_step_graph(TTSTransformer & self, int32_t n_past, int32_t generation_step) {
    auto & impl = self.impl_;
    const auto & cfg = impl->model.config;
    const int n_head = cfg.code_pred_n_attention_heads;
    const int n_kv_head = cfg.code_pred_n_key_value_heads;
    const int head_dim = cfg.code_pred_head_dim;
    const int hidden_size = cfg.code_pred_hidden_size;
    const int in_hidden_size = cfg.hidden_size;
    const float eps = cfg.code_pred_rms_norm_eps;
    const float rope_theta = cfg.code_pred_rope_theta;
    const int n_layer = cfg.code_pred_layers;
    const int n_tokens = 1;
    const int n_kv_ctx = impl->state.code_pred_cache.n_ctx;
    const bool use_code_pred_mrope = false;

    struct ggml_init_params params = {
        /*.mem_size   =*/ impl->state.code_pred_compute_meta[generation_step].size(),
        /*.mem_buffer =*/ impl->state.code_pred_compute_meta[generation_step].data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_CODE_PRED_MAX_NODES, false);

    struct ggml_tensor * inp_hidden = nullptr;
    if (generation_step == 0) {
        inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, in_hidden_size);
        ggml_set_name(inp_hidden, "inp_hidden");
        ggml_set_input(inp_hidden);
    }

    struct ggml_tensor * inp_code = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_code, "inp_code");
    ggml_set_input(inp_code);

    struct ggml_tensor * inp_pos = nullptr;
    if (generation_step >= 0 && generation_step < (int) impl->state.code_pred_step_pos.size()) {
        inp_pos = impl->state.code_pred_step_pos[generation_step];
    }
    if (!inp_pos) {
        inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_name(inp_pos, "inp_pos");
        ggml_set_input(inp_pos);
    }

    struct ggml_tensor * inp_mrope_pos = nullptr;
    if (use_code_pred_mrope) {
        inp_mrope_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4);
        ggml_set_name(inp_mrope_pos, "inp_mrope_pos");
        ggml_set_input(inp_mrope_pos);
    }

    struct ggml_tensor * inp_mask = nullptr;
    if (generation_step >= 0 && generation_step < (int) impl->state.code_pred_step_mask_tensors.size()) {
        inp_mask = impl->state.code_pred_step_mask_tensors[generation_step];
    }
    if (!inp_mask) {
        inp_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_kv_ctx, n_tokens);
        ggml_set_name(inp_mask, "inp_mask");
        ggml_set_input(inp_mask);
    }

    struct ggml_tensor * cur;
    if (generation_step == 0) {
        cur = ggml_reshape_2d(ctx0, inp_hidden, in_hidden_size, 1);
    } else {
        cur = ggml_get_rows(ctx0, impl->model.code_pred_embd[generation_step - 1], inp_code);
        cur = ggml_reshape_2d(ctx0, cur, in_hidden_size, 1);
    }

    if (impl->model.code_pred_small_to_mtp_weight) {
        cur = ggml_mul_mat(ctx0, impl->model.code_pred_small_to_mtp_weight, cur);
        if (impl->model.code_pred_small_to_mtp_bias) {
            struct ggml_tensor * bias = ggml_repeat(
                ctx0, ggml_reshape_2d(ctx0, impl->model.code_pred_small_to_mtp_bias, hidden_size, 1), cur);
            cur = ggml_add(ctx0, cur, bias);
        }
    }

    // Packed QKV is intentionally limited to recurrent
    // Code Predictor steps; prefill remains on the established path.
    struct ggml_tensor * inpL = cur;
    const float KQscale = 1.0f / sqrtf((float) head_dim);
    int mrope_sections[GGML_MROPE_SECTIONS] = { cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2], 0 };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = impl->model.code_pred_layers[il];

        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        struct ggml_tensor * Qcur = nullptr;
        struct ggml_tensor * Kcur = nullptr;
        struct ggml_tensor * Vcur = nullptr;
        if (packed_code_pred_qkv_enabled() &&
            is_cuda_backend(impl->state.backend) && layer.attn_qkv) {
            struct ggml_tensor * qkv = ggml_mul_mat(ctx0, layer.attn_qkv, cur);
            const size_t q_bytes = (size_t) head_dim * (size_t) n_head * qkv->nb[0];
            const size_t kv_bytes = (size_t) head_dim * (size_t) n_kv_head * qkv->nb[0];
            Qcur = ggml_view_3d(ctx0, qkv, head_dim, n_head, n_tokens,
                                (size_t) head_dim * qkv->nb[0], qkv->nb[1], 0);
            Kcur = ggml_view_3d(ctx0, qkv, head_dim, n_kv_head, n_tokens,
                                (size_t) head_dim * qkv->nb[0], qkv->nb[1], q_bytes);
            Vcur = ggml_view_3d(ctx0, qkv, head_dim, n_kv_head, n_tokens,
                                (size_t) head_dim * qkv->nb[0], qkv->nb[1], q_bytes + kv_bytes);
        } else {
            Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
            Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
            Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);

            Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens);
        }

        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }

        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }

        if (use_code_pred_mrope) {
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

        struct ggml_tensor * k_cache = impl->state.code_pred_cache.k_cache[il];
        struct ggml_tensor * v_cache = impl->state.code_pred_cache.v_cache[il];

        struct ggml_tensor * Kcache = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));
        struct ggml_tensor * Vcache = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 0, 2, 1, 3));

        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, k_cache, Kcache, inp_pos));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, v_cache, Vcache, inp_pos));

        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_ctx, n_kv_head,
            k_cache->nb[1], k_cache->nb[2], 0);

        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_ctx, n_kv_head,
            v_cache->nb[1], v_cache->nb[2], 0);

        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);

        if (n_tokens == 1) {
            struct ggml_tensor * KQV_fa = ggml_flash_attn_ext(ctx0, Q, K, V, inp_mask,
                                                              1.0f / sqrtf((float) head_dim), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(KQV_fa, GGML_PREC_F32);
            cur = ggml_reshape_2d(ctx0, KQV_fa, n_head * head_dim, n_tokens);
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

        cur = ggml_swiglu_split(ctx0, gate, up);

        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, impl->model.code_pred_output_norm);

    struct ggml_tensor * logits = ggml_mul_mat(ctx0, impl->model.code_pred_head[generation_step], cur);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);
    return gf;
}

} // namespace qwen3_tts
