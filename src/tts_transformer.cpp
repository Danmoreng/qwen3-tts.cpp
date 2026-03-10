#include "tts_transformer.h"
#include "gguf_loader.h"
#include "transformer/transformer_internal.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>
#include <cstdlib>
#include <sys/stat.h>

namespace qwen3_tts {

TTSTransformer::TTSTransformer() = default;

TTSTransformer::~TTSTransformer() {
    unload_model();
}

struct ggml_cgraph * TTSTransformer::build_code_pred_graph(int32_t n_prev_codes) {
    const auto & cfg = model_.config;
    const int n_head = cfg.code_pred_n_attention_heads;
    const int n_kv_head = cfg.code_pred_n_key_value_heads;
    const int head_dim = cfg.code_pred_head_dim;
    const int hidden_size = cfg.code_pred_hidden_size;
    const int in_hidden_size = cfg.hidden_size;
    const float eps = cfg.code_pred_rms_norm_eps;
    const int n_layer = cfg.code_pred_layers;
    const int n_codebooks = cfg.n_codebooks;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
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
            struct ggml_tensor * code_embd = ggml_get_rows(ctx0, model_.code_pred_embd[cb], code_idx);
            cur = ggml_add(ctx0, cur, code_embd);
        }
    }

    if (model_.code_pred_small_to_mtp_weight) {
        cur = ggml_mul_mat(ctx0, model_.code_pred_small_to_mtp_weight, cur);
        if (model_.code_pred_small_to_mtp_bias) {
            struct ggml_tensor * bias = ggml_repeat(ctx0,
                ggml_reshape_2d(ctx0, model_.code_pred_small_to_mtp_bias, hidden_size, 1), cur);
            cur = ggml_add(ctx0, cur, bias);
        }
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.code_pred_layers[il];
        
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
        struct ggml_tensor * inpFF = cur;
        
        cur = ggml_rms_norm(ctx0, inpFF, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        
        cur = ggml_mul(ctx0, gate, up);
        
        struct ggml_tensor * old_ffn_down_f32 = ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32);
        cur = ggml_mul_mat(ctx0, old_ffn_down_f32, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
    cur = inpL;
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.code_pred_output_norm);
    
    std::vector<struct ggml_tensor *> all_logits;
    for (int cb = 0; cb < n_codebooks - 1; ++cb) {
        struct ggml_tensor * cb_logits = ggml_mul_mat(ctx0, model_.code_pred_head[cb], cur);
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

struct ggml_cgraph * TTSTransformer::build_code_pred_prefill_graph() {
    const auto & cfg = model_.config;
    const int n_head = cfg.code_pred_n_attention_heads;
    const int n_kv_head = cfg.code_pred_n_key_value_heads;
    const int head_dim = cfg.code_pred_head_dim;
    const int hidden_size = cfg.code_pred_hidden_size;
    const int in_hidden_size = cfg.hidden_size;
    const float eps = cfg.code_pred_rms_norm_eps;
    const float rope_theta = cfg.code_pred_rope_theta;
    const int n_layer = cfg.code_pred_layers;
    const int n_tokens = 2;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.code_pred_compute_meta[0].size(),
        /*.mem_buffer =*/ state_.code_pred_compute_meta[0].data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
    
    // Input: past_hidden from talker [hidden_size]
    struct ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, in_hidden_size);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);
    
    // Input: codebook 0 token embedding [talker hidden] (pre-computed using talker's codec_embd)
    struct ggml_tensor * inp_cb0_embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, in_hidden_size);
    ggml_set_name(inp_cb0_embd, "inp_cb0_embd");
    ggml_set_input(inp_cb0_embd);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    struct ggml_tensor * inp_mrope_pos = nullptr;
    if (cfg.use_mrope) {
        inp_mrope_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4 * n_tokens);
        ggml_set_name(inp_mrope_pos, "inp_mrope_pos");
        ggml_set_input(inp_mrope_pos);
    }
    
    // Concatenate [past_hidden, cb0_embd] -> [2, talker_hidden]
    struct ggml_tensor * hidden_2d = ggml_reshape_2d(ctx0, inp_hidden, in_hidden_size, 1);
    struct ggml_tensor * cb0_2d = ggml_reshape_2d(ctx0, inp_cb0_embd, in_hidden_size, 1);
    struct ggml_tensor * cur = ggml_concat(ctx0, hidden_2d, cb0_2d, 1);

    // 1.7B uses a learned projection from talker hidden -> code predictor hidden.
    if (model_.code_pred_small_to_mtp_weight) {
        cur = ggml_mul_mat(ctx0, model_.code_pred_small_to_mtp_weight, cur);
        if (model_.code_pred_small_to_mtp_bias) {
            struct ggml_tensor * bias = ggml_repeat(ctx0,
                ggml_reshape_2d(ctx0, model_.code_pred_small_to_mtp_bias, hidden_size, 1), cur);
            cur = ggml_add(ctx0, cur, bias);
        }
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    int mrope_sections[GGML_MROPE_SECTIONS] = { cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2], 0 };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.code_pred_layers[il];
        
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
        
        struct ggml_tensor * k_cache = state_.code_pred_cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.code_pred_cache.v_cache[il];
        
        // Store at position 0 (prefill starts fresh)
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2], 0);
        
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2], 0);
        
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
                struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                struct ggml_tensor * K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
                struct ggml_tensor * V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);
        
                if (n_tokens == 1) {
                    struct ggml_tensor * KQV_fa = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, KQscale, 0.0f, 0.0f);
                    cur = ggml_cont_2d(ctx0, KQV_fa, n_head * head_dim, n_tokens);
                } else {
                    struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
                    KQ = ggml_scale(ctx0, KQ, KQscale);
                    KQ = ggml_diag_mask_inf(ctx0, KQ, 0);
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
     cur = ggml_mul(ctx0, cur, model_.code_pred_output_norm);
     
     struct ggml_tensor * last_hidden = ggml_view_2d(ctx0, cur, hidden_size, 1, 
                                                      cur->nb[1], hidden_size * sizeof(float));
     
     struct ggml_tensor * logits = ggml_mul_mat(ctx0, model_.code_pred_head[0], last_hidden);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    
    ggml_build_forward_expand(gf, logits);
    
    ggml_free(ctx0);
    
    return gf;
}

struct ggml_cgraph * TTSTransformer::build_code_pred_step_graph(int32_t n_past, int32_t generation_step) {
    const auto & cfg = model_.config;
    const int n_head = cfg.code_pred_n_attention_heads;
    const int n_kv_head = cfg.code_pred_n_key_value_heads;
    const int head_dim = cfg.code_pred_head_dim;
    const int hidden_size = cfg.code_pred_hidden_size;
    const int in_hidden_size = cfg.hidden_size;
    const float eps = cfg.code_pred_rms_norm_eps;
    const float rope_theta = cfg.code_pred_rope_theta;
    const int n_layer = cfg.code_pred_layers;
    const int n_tokens = 1;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.code_pred_compute_meta[generation_step].size(),
        /*.mem_buffer =*/ state_.code_pred_compute_meta[generation_step].data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
    
    struct ggml_tensor * inp_hidden = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, in_hidden_size);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);
    
    struct ggml_tensor * inp_code = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_code, "inp_code");
    ggml_set_input(inp_code);
    
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    struct ggml_tensor * inp_mrope_pos = nullptr;
    if (cfg.use_mrope) {
        inp_mrope_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4);
        ggml_set_name(inp_mrope_pos, "inp_mrope_pos");
        ggml_set_input(inp_mrope_pos);
    }
    
    struct ggml_tensor * inp_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, state_.code_pred_cache.n_ctx, 1);
    ggml_set_name(inp_mask, "inp_mask");
    ggml_set_input(inp_mask);
    
    struct ggml_tensor * cur;
    if (generation_step == 0) {
        cur = ggml_reshape_2d(ctx0, inp_hidden, in_hidden_size, 1);
    } else {
        cur = ggml_get_rows(ctx0, model_.code_pred_embd[generation_step - 1], inp_code);
        cur = ggml_reshape_2d(ctx0, cur, in_hidden_size, 1);
    }

    if (model_.code_pred_small_to_mtp_weight) {
        cur = ggml_mul_mat(ctx0, model_.code_pred_small_to_mtp_weight, cur);
        if (model_.code_pred_small_to_mtp_bias) {
            struct ggml_tensor * bias = ggml_repeat(ctx0,
                ggml_reshape_2d(ctx0, model_.code_pred_small_to_mtp_bias, hidden_size, 1), cur);
            cur = ggml_add(ctx0, cur, bias);
        }
    }
    
    struct ggml_tensor * inpL = cur;
    
    const float KQscale = 1.0f / sqrtf(float(head_dim));
    
    int mrope_sections[GGML_MROPE_SECTIONS] = { cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2], 0 };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.code_pred_layers[il];
        
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
        
        struct ggml_tensor * k_cache = state_.code_pred_cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.code_pred_cache.v_cache[il];
        
        struct ggml_tensor * k_cache_2d = ggml_view_2d(ctx0, k_cache, head_dim * n_kv_head, state_.code_pred_cache.n_ctx, k_cache->nb[2], 0);
        struct ggml_tensor * v_cache_2d = ggml_view_2d(ctx0, v_cache, head_dim * n_kv_head, state_.code_pred_cache.n_ctx, v_cache->nb[2], 0);
        
        struct ggml_tensor * Kcur_2d = ggml_view_2d(ctx0, Kcur, head_dim * n_kv_head, n_tokens, Kcur->nb[2], 0);
        struct ggml_tensor * Vcur_2d = ggml_view_2d(ctx0, Vcur, head_dim * n_kv_head, n_tokens, Vcur->nb[2], 0);
        
        struct ggml_tensor * k_updated = ggml_set_rows(ctx0, k_cache_2d, Kcur_2d, inp_pos);
        struct ggml_tensor * v_updated = ggml_set_rows(ctx0, v_cache_2d, Vcur_2d, inp_pos);
        
        ggml_build_forward_expand(gf, k_updated);
        ggml_build_forward_expand(gf, v_updated);
        
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, state_.code_pred_cache.n_ctx,
            k_cache->nb[1], k_cache->nb[2], 0);
        
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, state_.code_pred_cache.n_ctx,
            v_cache->nb[1], v_cache->nb[2], 0);
        
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        
        if (n_tokens == 1) {
            struct ggml_tensor * KQ_mask = ggml_get_tensor(ctx0, "inp_mask");
            struct ggml_tensor * KQV_fa = ggml_flash_attn_ext(ctx0, Q, K, V, KQ_mask, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
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
        
        struct ggml_tensor * step_ffn_down_f32 = ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32);
        cur = ggml_mul_mat(ctx0, step_ffn_down_f32, cur);
        
        inpL = ggml_add(ctx0, cur, inpFF);
    }
    
     cur = inpL;
     
     cur = ggml_rms_norm(ctx0, cur, eps);
     cur = ggml_mul(ctx0, cur, model_.code_pred_output_norm);
     
     struct ggml_tensor * logits = ggml_mul_mat(ctx0, model_.code_pred_head[generation_step], cur);
     ggml_set_name(logits, "logits");
     ggml_set_output(logits);
     
     ggml_build_forward_expand(gf, logits);
    
    ggml_free(ctx0);
    
    return gf;
}

bool TTSTransformer::forward_prefill(const float * prefill_embd, int32_t n_tokens,
                                     int32_t n_past, std::vector<float> & output,
                                     std::vector<float> * logits_out) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!prefill_embd) {
        error_msg_ = "prefill_embd is null";
        return false;
    }
    if (n_tokens <= 0) {
        error_msg_ = "n_tokens must be > 0";
        return false;
    }
    
    if (state_.cache.n_ctx == 0) {
        const int32_t min_ctx = std::max<int32_t>(256, n_past + n_tokens + 16);
        if (!init_kv_cache(min_ctx)) {
            return false;
        }
    }
    
    if (n_past + n_tokens > state_.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }
    
#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_cgraph * gf = build_prefill_forward_graph(n_tokens, n_past);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_tensor * inp_prefill = ggml_graph_get_tensor(gf, "inp_prefill_embd");
    if (inp_prefill) {
        ggml_backend_tensor_set(inp_prefill, prefill_embd, 0,
                                (size_t)n_tokens * model_.config.hidden_size * sizeof(float));
    }
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            positions[i] = n_past + i;
        }
        size_t write_size = positions.size() * sizeof(int32_t);
        if (write_size > ggml_nbytes(inp_pos)) {
            fprintf(stderr, "  ERROR: inp_pos write out of bounds! nbytes=%zu, write=%zu\n", ggml_nbytes(inp_pos), write_size);
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, write_size);
    }
    
    struct ggml_tensor * inp_mrope_pos = ggml_graph_get_tensor(gf, "inp_mrope_pos");
    if (inp_mrope_pos && model_.config.use_mrope) {
        std::vector<int32_t> positions(n_tokens * 4);
        for (int i = 0; i < n_tokens; ++i) {
            int32_t p = n_past + i;
            positions[i + n_tokens * 0] = p;
            positions[i + n_tokens * 1] = p;
            positions[i + n_tokens * 2] = p;
            positions[i + n_tokens * 3] = 0; // 4th dimension unused
        }
        size_t write_size = positions.size() * sizeof(int32_t);
        if (write_size > ggml_nbytes(inp_mrope_pos)) {
            fprintf(stderr, "  ERROR: inp_mrope_pos write out of bounds! nbytes=%zu, write=%zu\n", ggml_nbytes(inp_mrope_pos), write_size);
        }
        ggml_backend_tensor_set(inp_mrope_pos, positions.data(), 0, write_size);
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");
    if (!hidden) {
        error_msg_ = "Failed to find hidden_states tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    output.resize(n_tokens * model_.config.hidden_size);
    ggml_backend_tensor_get(hidden, output.data(), 0, output.size() * sizeof(float));
    
    last_hidden_.resize(model_.config.hidden_size);
    ggml_backend_tensor_get(hidden, last_hidden_.data(), 
                           (n_tokens - 1) * model_.config.hidden_size * sizeof(float),
                           model_.config.hidden_size * sizeof(float));

    if (logits_out) {
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

        logits_out->resize(model_.config.codec_vocab_size);
        ggml_backend_tensor_get(logits, logits_out->data(),
                                (n_tokens - 1) * model_.config.codec_vocab_size * sizeof(float),
                                model_.config.codec_vocab_size * sizeof(float));
    }
    
    state_.cache.n_used = n_past + n_tokens;
    
    ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_prefill_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    return true;
}

bool TTSTransformer::forward_text(const int32_t * text_tokens, int32_t n_tokens,
                                  const float * speaker_embd, int32_t n_past,
                                  std::vector<float> & output) {
    if (!text_tokens) {
        error_msg_ = "text_tokens is null";
        return false;
    }
    if (n_tokens <= 0) {
        error_msg_ = "n_tokens must be > 0";
        return false;
    }

    std::vector<float> projected;
    if (!project_text_tokens(text_tokens, n_tokens, projected)) {
        return false;
    }

    if (speaker_embd) {
        const int32_t hidden_size = model_.config.hidden_size;
        for (int32_t t = 0; t < n_tokens; ++t) {
            float * row = projected.data() + (size_t)t * hidden_size;
            for (int32_t h = 0; h < hidden_size; ++h) {
                row[h] += speaker_embd[h];
            }
        }
    }

    return forward_prefill(projected.data(), n_tokens, n_past, output, nullptr);
}

bool TTSTransformer::forward_step(const float * step_embd, int32_t n_past,
                                  std::vector<float> & output,
                                  std::vector<float> * hidden_out) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!step_embd) {
        error_msg_ = "step_embd is null";
        return false;
    }

    if (state_.cache.n_ctx == 0) {
        const int32_t min_ctx = std::max<int32_t>(256, n_past + 1 + 16);
        if (!init_kv_cache(min_ctx)) {
            return false;
        }
    }

    if (n_past + 1 > state_.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }
    
#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_cgraph * gf = build_step_graph(n_past);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_tensor * inp_step = ggml_graph_get_tensor(gf, "inp_step_embd");
    if (inp_step) {
        ggml_backend_tensor_set(inp_step, step_embd, 0,
                                model_.config.hidden_size * sizeof(float));
    }
    
    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        int32_t pos = n_past;
        ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
    }
    
    struct ggml_tensor * inp_mrope_pos = ggml_graph_get_tensor(gf, "inp_mrope_pos");
    if (inp_mrope_pos && model_.config.use_mrope) {
        int32_t positions[4] = {n_past, n_past, n_past, 0};
        ggml_backend_tensor_set(inp_mrope_pos, positions, 0, 4 * sizeof(int32_t));
    }

    struct ggml_tensor * inp_mask = ggml_graph_get_tensor(gf, "inp_mask");
    if (inp_mask) {
        std::vector<ggml_fp16_t> mask(state_.cache.n_ctx, ggml_fp32_to_fp16(-INFINITY));
        for (int i = 0; i <= n_past; i++) {
            mask[i] = ggml_fp32_to_fp16(0.0f);
        }
        ggml_backend_tensor_set(inp_mask, mask.data(), 0, state_.cache.n_ctx * sizeof(ggml_fp16_t));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (hidden) {
        last_hidden_.resize(model_.config.hidden_size);
        ggml_backend_tensor_get(hidden, last_hidden_.data(), 0, 
                               model_.config.hidden_size * sizeof(float));
        if (hidden_out) {
            *hidden_out = last_hidden_;
        }
    }
    
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    output.resize(model_.config.codec_vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
    
    state_.cache.n_used = n_past + 1;
    
    ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_talker_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    return true;
}

bool TTSTransformer::forward_codec(int32_t codec_token, int32_t n_past,
                                   std::vector<float> & output) {
    std::vector<float> codec_row;
    if (!lookup_embedding_rows(model_.codec_embd, &codec_token, 1,
                               "inp_legacy_codec_token", "legacy_codec_row",
                               codec_row)) {
        return false;
    }

    return forward_step(codec_row.data(), n_past, output, nullptr);
}

bool TTSTransformer::get_hidden_states(std::vector<float> & hidden) const {
    if (last_hidden_.empty()) {
        return false;
    }
    hidden = last_hidden_;
    return true;
}

bool TTSTransformer::predict_codes(const float * hidden, const int32_t * prev_codes,
                                    std::vector<float> & output) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    const auto & cfg = model_.config;
    int n_prev = (prev_codes != nullptr) ? cfg.n_codebooks - 1 : 0;
    
    struct ggml_cgraph * gf = build_code_pred_graph(n_prev);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate code predictor graph";
        return false;
    }
    
    struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
    if (inp_hidden) {
        ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
    }
    
    if (n_prev > 0) {
        struct ggml_tensor * inp_prev = ggml_graph_get_tensor(gf, "inp_prev_codes");
        if (inp_prev) {
            ggml_backend_tensor_set(inp_prev, prev_codes, 0, n_prev * sizeof(int32_t));
        }
    }
    
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute code predictor graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    output.resize((cfg.n_codebooks - 1) * cfg.code_pred_vocab_size);
    
    for (int cb = 0; cb < cfg.n_codebooks - 1; ++cb) {
        char name[32];
        snprintf(name, sizeof(name), "logits_cb%d", cb + 1);
        struct ggml_tensor * cb_logits = ggml_graph_get_tensor(gf, name);
        if (cb_logits) {
            ggml_backend_tensor_get(cb_logits, output.data() + cb * cfg.code_pred_vocab_size,
                                   0, cfg.code_pred_vocab_size * sizeof(float));
        }
    }
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

static int32_t argmax(const float * data, int32_t n) {
    int32_t max_idx = 0;
    float max_val = data[0];
    for (int32_t i = 1; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

bool TTSTransformer::predict_codes_autoregressive_coreml(const float * hidden,
                                                         int32_t codebook_0_token,
                                                         std::vector<int32_t> & output,
                                                         float temperature,
                                                         int32_t top_k,
                                                         int32_t trace_frame) {
    if (!use_coreml_code_predictor_ || !coreml_code_predictor_.is_loaded()) {
        error_msg_ = "CoreML code predictor is not loaded";
        return false;
    }

    const auto & cfg = model_.config;
    const int32_t n_steps = cfg.n_codebooks - 1;
    const auto & trace_cfg = transformer_internal::get_debug_trace_config();
    const bool trace_frame_enabled = transformer_internal::debug_trace_should_dump_frame(trace_cfg, trace_frame);

    output.resize(n_steps);
    std::vector<float> logits_data(cfg.code_pred_vocab_size);
    std::vector<float> code_probs(cfg.code_pred_vocab_size);
    std::vector<float> seq_embd((size_t)16 * cfg.hidden_size, 0.0f);

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
        if (temperature <= 0.0f) {
            return argmax(logits_ptr, vocab_size);
        }

        for (int32_t i = 0; i < vocab_size; ++i) {
            logits_ptr[i] /= temperature;
        }

        if (top_k > 0 && top_k < vocab_size) {
            std::vector<std::pair<float, int32_t>> scored(vocab_size);
            for (int32_t i = 0; i < vocab_size; ++i) {
                scored[i] = {logits_ptr[i], i};
            }
            std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first;
                });
            float threshold = scored[top_k - 1].first;
            for (int32_t i = 0; i < vocab_size; ++i) {
                if (logits_ptr[i] < threshold) {
                    logits_ptr[i] = -INFINITY;
                }
            }
        }

        float max_logit = *std::max_element(logits_ptr, logits_ptr + vocab_size);
        double sum = 0.0;
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = expf(logits_ptr[i] - max_logit);
            sum += code_probs[i];
        }
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = (float)(code_probs[i] / sum);
        }

        std::discrete_distribution<int32_t> dist(code_probs.begin(), code_probs.begin() + vocab_size);
        return dist(rng_);
    };

    memcpy(seq_embd.data(), hidden, (size_t)cfg.hidden_size * sizeof(float));
    if (!lookup_single_embedding_row(model_.codec_embd, codebook_0_token,
                                     seq_embd.data() + cfg.hidden_size)) {
        return false;
    }

    if (trace_frame_enabled) {
        char name[128];
        snprintf(name, sizeof(name), "frame%03d_codepred_input_hidden.f32.bin", trace_frame);
        transformer_internal::debug_trace_write_bin(trace_cfg, name, hidden, (size_t) cfg.hidden_size,
                                                    "f32", {(int64_t) cfg.hidden_size});
    }

#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_code_pred_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    for (int32_t step = 0; step < n_steps; ++step) {
        if (step > 0) {
            float * dst = seq_embd.data() + (size_t)(step + 1) * cfg.hidden_size;
            if (!lookup_single_embedding_row(model_.code_pred_embd[step - 1], output[step - 1], dst)) {
                return false;
            }
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!coreml_code_predictor_.predict_step(step, seq_embd.data(), step + 2, cfg.hidden_size, logits_data)) {
            error_msg_ = "CoreML predictor step failed: " + coreml_code_predictor_.get_error();
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (timing_) timing_->t_code_pred_compute_ms += dt_ms;
        if (timing_) timing_->t_code_pred_coreml_ms += dt_ms;
#endif

        if ((int32_t)logits_data.size() != cfg.code_pred_vocab_size) {
            error_msg_ = "CoreML predictor returned unexpected logits size";
            return false;
        }

        if (trace_frame_enabled && step < trace_cfg.max_code_steps) {
            char logits_name[128];
            snprintf(logits_name, sizeof(logits_name),
                     "frame%03d_codepred_logits_step%02d.f32.bin", trace_frame, step);
            transformer_internal::debug_trace_write_bin(trace_cfg, logits_name, logits_data.data(),
                                                        (size_t) cfg.code_pred_vocab_size,
                                                        "f32", {(int64_t) cfg.code_pred_vocab_size});
        }

        output[step] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);

#ifdef QWEN3_TTS_TIMING
        if (timing_) {
            if (step == 0) {
                timing_->t_code_pred_prefill_ms += dt_ms;
            } else {
                timing_->t_code_pred_steps_ms += dt_ms;
            }
        }
#endif
    }

    if (trace_frame_enabled) {
        char tokens_name[128];
        snprintf(tokens_name, sizeof(tokens_name),
                 "frame%03d_codepred_tokens_cb1_15.i32.bin", trace_frame);
        transformer_internal::debug_trace_write_bin(trace_cfg, tokens_name, output.data(), output.size(),
                                                    "i32", {(int64_t) output.size()});
    }

    return true;
}

bool TTSTransformer::predict_codes_autoregressive(const float * hidden, int32_t codebook_0_token,
                                                   std::vector<int32_t> & output,
                                                   float temperature, int32_t top_k,
                                                   int32_t trace_frame) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    const auto & cfg = model_.config;
    const auto & trace_cfg = transformer_internal::get_debug_trace_config();
    const bool trace_frame_enabled = transformer_internal::debug_trace_should_dump_frame(trace_cfg, trace_frame);

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    if (use_coreml_code_predictor_ && coreml_code_predictor_.is_loaded()) {
        if (predict_codes_autoregressive_coreml(hidden, codebook_0_token, output, temperature, top_k, trace_frame)) {
            return true;
        }
        if (skip_ggml_code_pred_layers_) {
            return false;
        }
        fprintf(stderr, "  CoreML code predictor failed, falling back to GGML: %s\n", error_msg_.c_str());
        use_coreml_code_predictor_ = false;
    }
    
    if (state_.code_pred_cache.n_ctx < 16) {
        if (!init_code_pred_kv_cache(16)) {
            return false;
        }
    }
    clear_code_pred_kv_cache();
    
    output.resize(15);
    std::vector<float> logits_data(cfg.code_pred_vocab_size);
    
    std::vector<float> code_probs(cfg.code_pred_vocab_size);
    
    // Helper lambda: temperature + top-k sampling (or greedy if temperature <= 0)
    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
        if (temperature <= 0.0f) {
            return argmax(logits_ptr, vocab_size);
        }
        // Temperature scaling
        for (int32_t i = 0; i < vocab_size; ++i) {
            logits_ptr[i] /= temperature;
        }
        // Top-k filtering
        if (top_k > 0 && top_k < vocab_size) {
            std::vector<std::pair<float, int32_t>> scored(vocab_size);
            for (int32_t i = 0; i < vocab_size; ++i) {
                scored[i] = {logits_ptr[i], i};
            }
            std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first;
                });
            float threshold = scored[top_k - 1].first;
            for (int32_t i = 0; i < vocab_size; ++i) {
                if (logits_ptr[i] < threshold) {
                    logits_ptr[i] = -INFINITY;
                }
            }
        }
        // Softmax
        float max_logit = *std::max_element(logits_ptr, logits_ptr + vocab_size);
        double sum = 0.0;
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = expf(logits_ptr[i] - max_logit);
            sum += code_probs[i];
        }
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = (float)(code_probs[i] / sum);
        }
        // Sample
        std::discrete_distribution<int32_t> dist(code_probs.begin(), code_probs.begin() + vocab_size);
        return dist(rng_);
    };
    
    std::vector<float> cb0_embd(cfg.hidden_size);
    if (!lookup_single_embedding_row(model_.codec_embd, codebook_0_token, cb0_embd.data())) {
        return false;
    }
    if (trace_frame_enabled) {
        char hidden_name[128];
        snprintf(hidden_name, sizeof(hidden_name),
                 "frame%03d_codepred_input_hidden.f32.bin", trace_frame);
        transformer_internal::debug_trace_write_bin(trace_cfg, hidden_name, hidden,
                                                    (size_t) cfg.hidden_size, "f32",
                                                    {(int64_t) cfg.hidden_size});

        char embd_name[128];
        snprintf(embd_name, sizeof(embd_name),
                 "frame%03d_codepred_input_cb0_embd.f32.bin", trace_frame);
        transformer_internal::debug_trace_write_bin(trace_cfg, embd_name, cb0_embd.data(),
                                                    (size_t) cfg.hidden_size, "f32",
                                                    {(int64_t) cfg.hidden_size});
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (timing_) timing_->t_code_pred_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    // Prefill with 2 tokens [past_hidden, cb0_embd]
    {
#ifdef QWEN3_TTS_TIMING
        auto t_pf_start = clk::now();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_cgraph * gf = build_code_pred_prefill_graph();
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
            error_msg_ = "Failed to allocate code predictor prefill graph";
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
        if (inp_hidden) {
            ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
        }
        
        struct ggml_tensor * inp_cb0_embd = ggml_graph_get_tensor(gf, "inp_cb0_embd");
        if (inp_cb0_embd) {
            ggml_backend_tensor_set(inp_cb0_embd, cb0_embd.data(), 0, cfg.hidden_size * sizeof(float));
        }
        
        struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
        if (inp_pos) {
            int32_t positions[2] = {0, 1};
            ggml_backend_tensor_set(inp_pos, positions, 0, 2 * sizeof(int32_t));
        }
        
        struct ggml_tensor * inp_mrope_pos = ggml_graph_get_tensor(gf, "inp_mrope_pos");
        if (inp_mrope_pos && model_.config.use_mrope) {
            int32_t positions[8] = {0, 1, 0, 1, 0, 1, 0, 0};
            ggml_backend_tensor_set(inp_mrope_pos, positions, 0, 8 * sizeof(int32_t));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute code predictor prefill graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
        
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor in prefill";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        ggml_backend_tensor_get(logits, logits_data.data(), 0, 
                                 cfg.code_pred_vocab_size * sizeof(float));

        if (trace_frame_enabled && 0 < trace_cfg.max_code_steps) {
            char logits_name[128];
            snprintf(logits_name, sizeof(logits_name),
                     "frame%03d_codepred_logits_step00.f32.bin", trace_frame);
            transformer_internal::debug_trace_write_bin(trace_cfg, logits_name, logits_data.data(),
                                                        (size_t) cfg.code_pred_vocab_size, "f32",
                                                        {(int64_t) cfg.code_pred_vocab_size});
        }
        
        output[0] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);
        
        ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (timing_) timing_->t_code_pred_prefill_ms += std::chrono::duration<double, std::milli>(t1 - t_pf_start).count();
#endif
    }
    
    // Generate 14 more tokens autoregressively
#ifdef QWEN3_TTS_TIMING
    auto t_steps_start = clk::now();
#endif
    if (state_.code_pred_mask.size() != (size_t) state_.code_pred_cache.n_ctx) {
        state_.code_pred_mask.resize((size_t) state_.code_pred_cache.n_ctx);
    }
    std::fill(state_.code_pred_mask.begin(), state_.code_pred_mask.end(), ggml_fp32_to_fp16(-INFINITY));
    const ggml_fp16_t zero_fp16 = ggml_fp32_to_fp16(0.0f);
    for (int i = 0; i <= 2 && i < state_.code_pred_cache.n_ctx; ++i) {
        state_.code_pred_mask[(size_t) i] = zero_fp16;
    }

    for (int step = 1; step < 15; ++step) {
        int32_t n_past = step + 1;
        if (n_past < state_.code_pred_cache.n_ctx) {
            state_.code_pred_mask[(size_t) n_past] = zero_fp16;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_cgraph * gf = build_code_pred_step_graph(n_past, step);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
            error_msg_ = "Failed to allocate code predictor step graph";
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
        if (inp_hidden) {
            ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
        }
        
        struct ggml_tensor * inp_code = ggml_graph_get_tensor(gf, "inp_code");
        if (inp_code) {
            int32_t prev_code = output[step - 1];
            ggml_backend_tensor_set(inp_code, &prev_code, 0, sizeof(int32_t));
        }
        
        struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
        if (inp_pos) {
            int32_t pos = n_past;
            ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
        }
        
        struct ggml_tensor * inp_mrope_pos = ggml_graph_get_tensor(gf, "inp_mrope_pos");
        if (inp_mrope_pos && model_.config.use_mrope) {
            int32_t positions[4] = {n_past, n_past, n_past, 0};
            ggml_backend_tensor_set(inp_mrope_pos, positions, 0, 4 * sizeof(int32_t));
        }

        struct ggml_tensor * inp_mask = ggml_graph_get_tensor(gf, "inp_mask");
        if (inp_mask) {
            ggml_backend_tensor_set(inp_mask, state_.code_pred_mask.data(), 0,
                                    state_.code_pred_cache.n_ctx * sizeof(ggml_fp16_t));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute code predictor step graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
        
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        ggml_backend_tensor_get(logits, logits_data.data(), 0, 
                                 cfg.code_pred_vocab_size * sizeof(float));

        if (trace_frame_enabled && step < trace_cfg.max_code_steps) {
            char logits_name[128];
            snprintf(logits_name, sizeof(logits_name),
                     "frame%03d_codepred_logits_step%02d.f32.bin", trace_frame, step);
            transformer_internal::debug_trace_write_bin(trace_cfg, logits_name, logits_data.data(),
                                                        (size_t) cfg.code_pred_vocab_size, "f32",
                                                        {(int64_t) cfg.code_pred_vocab_size});
        }
        
        output[step] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);
        
        ggml_backend_sched_reset(state_.sched);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (timing_) timing_->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    }
#ifdef QWEN3_TTS_TIMING
    if (timing_) timing_->t_code_pred_steps_ms += std::chrono::duration<double, std::milli>(clk::now() - t_steps_start).count();
#endif

    if (trace_frame_enabled) {
        char tokens_name[128];
        snprintf(tokens_name, sizeof(tokens_name),
                 "frame%03d_codepred_tokens_cb1_15.i32.bin", trace_frame);
        transformer_internal::debug_trace_write_bin(trace_cfg, tokens_name, output.data(), output.size(),
                                                    "i32", {(int64_t) output.size()});
    }
    
    return true;
}

bool TTSTransformer::generate(const int32_t * text_tokens, int32_t n_tokens,
                               const float * speaker_embd, int32_t max_len,
                               std::vector<int32_t> & output,
                               int32_t language_id,
                               float repetition_penalty,
                               float temperature,
                               int32_t top_k,
                               const int32_t * instruct_tokens,
                               int32_t n_instruct_tokens) {
#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    tts_timing timing = {};
    auto t_gen_start = clk::now();
    auto t0 = t_gen_start, t1 = t_gen_start;
    timing_ = &timing;
#endif

    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!text_tokens) {
        error_msg_ = "text_tokens is null";
        return false;
    }
    if (n_tokens < 4) {
        error_msg_ = "Need at least 4 text tokens for generation";
        return false;
    }
    if (max_len <= 0) {
        output.clear();
        return true;
    }
    
    const auto & cfg = model_.config;
    const auto & trace_cfg = transformer_internal::get_debug_trace_config();
    if (trace_cfg.enabled) {
        transformer_internal::debug_trace_write_text_line(trace_cfg, "hidden_size=" + std::to_string(cfg.hidden_size));
        transformer_internal::debug_trace_write_text_line(trace_cfg, "codec_vocab_size=" + std::to_string(cfg.codec_vocab_size));
        transformer_internal::debug_trace_write_text_line(trace_cfg, "code_pred_vocab_size=" + std::to_string(cfg.code_pred_vocab_size));
        transformer_internal::debug_trace_write_text_line(trace_cfg, "n_codebooks=" + std::to_string(cfg.n_codebooks));
        transformer_internal::debug_trace_write_text_line(trace_cfg, "n_tokens=" + std::to_string(n_tokens));
        transformer_internal::debug_trace_write_text_line(trace_cfg, "max_len=" + std::to_string(max_len));
    }

    std::vector<float> prefill_embd;
    std::vector<float> trailing_text_hidden;
    std::vector<float> tts_pad_embed;

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!build_prefill_graph(text_tokens, n_tokens, speaker_embd, language_id,
                             prefill_embd, trailing_text_hidden, tts_pad_embed,
                             instruct_tokens, n_instruct_tokens)) {
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    timing.t_prefill_build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    const int32_t prefill_len = (int32_t)(prefill_embd.size() / cfg.hidden_size);
    const int32_t trailing_len = (int32_t)(trailing_text_hidden.size() / cfg.hidden_size);

    if (trace_cfg.enabled) {
        transformer_internal::debug_trace_write_text_line(trace_cfg, "prefill_len=" + std::to_string(prefill_len));
        transformer_internal::debug_trace_write_text_line(trace_cfg, "trailing_len=" + std::to_string(trailing_len));
        transformer_internal::debug_trace_write_bin(trace_cfg, "input_text_tokens.i32.bin", text_tokens,
                                                    (size_t) n_tokens, "i32", {(int64_t) n_tokens});
        if (!prefill_embd.empty()) {
            transformer_internal::debug_trace_write_bin(trace_cfg, "prefill_embd.f32.bin", prefill_embd.data(),
                                                        prefill_embd.size(), "f32",
                                                        {(int64_t) prefill_len, (int64_t) cfg.hidden_size});
        }
        if (speaker_embd) {
            transformer_internal::debug_trace_write_bin(trace_cfg, "speaker_embd.f32.bin", speaker_embd,
                                                        (size_t) cfg.hidden_size, "f32",
                                                        {(int64_t) cfg.hidden_size});
        }
    }

    const int32_t required_ctx = prefill_len + max_len + 8;
    if (state_.cache.n_ctx < required_ctx || state_.cache.n_ctx > std::max<int32_t>(required_ctx * 2, 512)) {
        if (!init_kv_cache(required_ctx)) {
            return false;
        }
    }
    clear_kv_cache();

    if (state_.code_pred_cache.n_ctx < 16) {
        if (!init_code_pred_kv_cache(16)) {
            return false;
        }
    }
    maybe_reserve_scheduler_graphs(prefill_len, required_ctx);
    
    std::vector<float> hidden_out;
    std::vector<float> logits;

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!forward_prefill(prefill_embd.data(), prefill_len, 0, hidden_out, &logits)) {
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    timing.t_prefill_forward_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    
    output.clear();
    output.reserve(max_len * cfg.n_codebooks);
    
    int32_t n_past = prefill_len;
    std::vector<int32_t> frame_codes(cfg.n_codebooks);
    std::unordered_set<int32_t> generated_cb0_tokens;
    const int32_t suppress_start = std::min(cfg.code_pred_vocab_size, cfg.codec_vocab_size);
    
    std::vector<float> probs(cfg.codec_vocab_size);
    std::vector<float> step_embd(cfg.hidden_size, 0.0f);
    std::vector<float> embd_row(cfg.hidden_size);
    
    for (int frame = 0; frame < max_len; ++frame) {
        const bool trace_frame = transformer_internal::debug_trace_should_dump_frame(trace_cfg, frame);
        if (trace_frame) {
            char raw_name[128];
            snprintf(raw_name, sizeof(raw_name), "frame%03d_cb0_logits_raw.f32.bin", frame);
            transformer_internal::debug_trace_write_bin(trace_cfg, raw_name, logits.data(),
                                                        (size_t) cfg.codec_vocab_size, "f32",
                                                        {(int64_t) cfg.codec_vocab_size});
        }

        // Suppress tokens in [codec_vocab_size - 1024, codec_vocab_size), except codec_eos_id
        for (int32_t i = suppress_start; i < cfg.codec_vocab_size; ++i) {
            if (i != cfg.codec_eos_id) {
                logits[i] = -INFINITY;
            }
        }

        // Repetition penalty (HuggingFace style) on previously generated CB0 tokens
        if (repetition_penalty != 1.0f) {
            for (int32_t tok : generated_cb0_tokens) {
                if (tok >= 0 && tok < cfg.codec_vocab_size) {
                    if (logits[tok] > 0.0f) {
                        logits[tok] /= repetition_penalty;
                    } else {
                        logits[tok] *= repetition_penalty;
                    }
                }
            }
        }

        if (trace_frame) {
            char post_rules_name[128];
            snprintf(post_rules_name, sizeof(post_rules_name), "frame%03d_cb0_logits_post_rules.f32.bin", frame);
            transformer_internal::debug_trace_write_bin(trace_cfg, post_rules_name, logits.data(),
                                                        (size_t) cfg.codec_vocab_size, "f32",
                                                        {(int64_t) cfg.codec_vocab_size});
        }

        int32_t next_token;
        if (temperature <= 0.0f) {
            next_token = argmax(logits.data(), cfg.codec_vocab_size);
        } else {
            for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                logits[i] /= temperature;
            }

            if (top_k > 0 && top_k < cfg.codec_vocab_size) {
                std::vector<std::pair<float, int32_t>> scored(cfg.codec_vocab_size);
                for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                    scored[i] = {logits[i], i};
                }
                std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                    [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                        return a.first > b.first;
                    });
                float threshold = scored[top_k - 1].first;
                for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                    if (logits[i] < threshold) {
                        logits[i] = -INFINITY;
                    }
                }
            }

            float max_logit = *std::max_element(logits.data(), logits.data() + cfg.codec_vocab_size);
            double sum = 0.0;
            for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                probs[i] = expf(logits[i] - max_logit);
                sum += probs[i];
            }
            for (int32_t i = 0; i < cfg.codec_vocab_size; ++i) {
                probs[i] = (float)(probs[i] / sum);
            }

            std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
            next_token = dist(rng_);
        }
        
        if (next_token == cfg.codec_eos_id) {
            if (trace_frame) {
                int32_t eos_token = next_token;
                char eos_name[128];
                snprintf(eos_name, sizeof(eos_name), "frame%03d_cb0_token.i32.bin", frame);
                transformer_internal::debug_trace_write_bin(trace_cfg, eos_name, &eos_token, 1, "i32", {1});
            }
            break;
        }

        const bool is_thinking = (next_token >= cfg.codec_think_id && next_token <= cfg.codec_think_eos_id);
        if (is_thinking) {
            fprintf(stderr, "  [frame %d] Filtering thinking token: %d\n", frame, next_token);
        }
        
        frame_codes[0] = next_token;
        generated_cb0_tokens.insert(next_token);
        if (trace_frame) {
            char token_name[128];
            snprintf(token_name, sizeof(token_name), "frame%03d_cb0_token.i32.bin", frame);
            transformer_internal::debug_trace_write_bin(trace_cfg, token_name, &frame_codes[0], 1, "i32", {1});

            char hidden_name[128];
            snprintf(hidden_name, sizeof(hidden_name), "frame%03d_talker_hidden.f32.bin", frame);
            transformer_internal::debug_trace_write_bin(trace_cfg, hidden_name, last_hidden_.data(),
                                                        last_hidden_.size(), "f32",
                                                        {(int64_t) last_hidden_.size()});
        }
        
#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        std::vector<int32_t> codes_1_15;
        if (!predict_codes_autoregressive(last_hidden_.data(), frame_codes[0], codes_1_15,
                                          temperature, top_k, frame)) {
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        timing.t_code_pred_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
        
        for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
            frame_codes[cb] = codes_1_15[cb - 1];
        }
        if (trace_frame) {
            char frame_codes_name[128];
            snprintf(frame_codes_name, sizeof(frame_codes_name),
                     "frame%03d_codec_tokens_cb0_15.i32.bin", frame);
            transformer_internal::debug_trace_write_bin(trace_cfg, frame_codes_name,
                                                        frame_codes.data(), frame_codes.size(), "i32",
                                                        {(int64_t) frame_codes.size()});
        }
        
        if (!is_thinking) {
            for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
                output.push_back(frame_codes[cb]);
            }
        }

#ifdef QWEN3_TTS_TIMING
        timing.n_frames = frame + 1;
#endif

        if (frame + 1 >= max_len) {
            break;
        }

        std::fill(step_embd.begin(), step_embd.end(), 0.0f);

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!lookup_single_embedding_row(model_.codec_embd, frame_codes[0], embd_row.data())) {
            return false;
        }
        for (int32_t h = 0; h < cfg.hidden_size; ++h) {
            step_embd[h] = embd_row[h];
        }

        for (int cb = 1; cb < cfg.n_codebooks; ++cb) {
            int32_t code_token = frame_codes[cb];
            if (!lookup_single_embedding_row(model_.code_pred_embd[cb - 1], code_token, embd_row.data())) {
                return false;
            }
            for (int32_t h = 0; h < cfg.hidden_size; ++h) {
                step_embd[h] += embd_row[h];
            }
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        timing.t_embed_lookup_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

        const float * trailing_row = (frame < trailing_len)
            ? trailing_text_hidden.data() + (size_t)frame * cfg.hidden_size
            : tts_pad_embed.data();
        for (int32_t h = 0; h < cfg.hidden_size; ++h) {
            step_embd[h] += trailing_row[h];
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!forward_step(step_embd.data(), n_past, logits)) {
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        timing.t_talker_forward_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
        
        n_past++;
    }
    
#ifdef QWEN3_TTS_TIMING
    timing.t_generate_total_ms = std::chrono::duration<double, std::milli>(clk::now() - t_gen_start).count();
    timing_ = nullptr;
    const auto & t = timing;
    int nf = t.n_frames;
    fprintf(stderr, "\n=== Detailed Generation Timing (%d frames) ===\n", nf);
    fprintf(stderr, "\n  Prefill:\n");
    fprintf(stderr, "    Build graph:      %8.1f ms\n", t.t_prefill_build_ms);
    fprintf(stderr, "    Forward total:    %8.1f ms\n", t.t_prefill_forward_ms);
    fprintf(stderr, "      Graph build:    %8.1f ms\n", t.t_prefill_graph_build_ms);
    fprintf(stderr, "      Graph alloc:    %8.1f ms\n", t.t_prefill_graph_alloc_ms);
    fprintf(stderr, "      Compute:        %8.1f ms\n", t.t_prefill_compute_ms);
    fprintf(stderr, "      Data I/O:       %8.1f ms\n", t.t_prefill_data_ms);
    fprintf(stderr, "\n  Talker forward_step (total / per-frame):\n");
    fprintf(stderr, "    Total:            %8.1f ms   (%.1f ms/frame)\n", t.t_talker_forward_ms, nf > 0 ? t.t_talker_forward_ms / nf : 0.0);
    fprintf(stderr, "      Graph build:    %8.1f ms   (%.1f ms/frame)\n", t.t_talker_graph_build_ms, nf > 0 ? t.t_talker_graph_build_ms / nf : 0.0);
    fprintf(stderr, "      Graph alloc:    %8.1f ms   (%.1f ms/frame)\n", t.t_talker_graph_alloc_ms, nf > 0 ? t.t_talker_graph_alloc_ms / nf : 0.0);
    fprintf(stderr, "      Compute:        %8.1f ms   (%.1f ms/frame)\n", t.t_talker_compute_ms, nf > 0 ? t.t_talker_compute_ms / nf : 0.0);
    fprintf(stderr, "      Data I/O:       %8.1f ms   (%.1f ms/frame)\n", t.t_talker_data_ms, nf > 0 ? t.t_talker_data_ms / nf : 0.0);
    fprintf(stderr, "\n  Code predictor (total / per-frame):\n");
    fprintf(stderr, "    Backend:          %s\n", use_coreml_code_predictor_ ? "CoreML (CPU+NE)" : "GGML");
    if (use_coreml_code_predictor_ && !coreml_code_predictor_path_.empty()) {
        fprintf(stderr, "    CoreML model:     %s\n", coreml_code_predictor_path_.c_str());
    }
    fprintf(stderr, "    Total:            %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_ms, nf > 0 ? t.t_code_pred_ms / nf : 0.0);
    fprintf(stderr, "      Init/KV/embed:  %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_init_ms, nf > 0 ? t.t_code_pred_init_ms / nf : 0.0);
    fprintf(stderr, "      Prefill (2tok): %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_prefill_ms, nf > 0 ? t.t_code_pred_prefill_ms / nf : 0.0);
    fprintf(stderr, "      Steps (14):     %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_steps_ms, nf > 0 ? t.t_code_pred_steps_ms / nf : 0.0);
    fprintf(stderr, "      Graph build:    %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_graph_build_ms, nf > 0 ? t.t_code_pred_graph_build_ms / nf : 0.0);
    fprintf(stderr, "      Graph alloc:    %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_graph_alloc_ms, nf > 0 ? t.t_code_pred_graph_alloc_ms / nf : 0.0);
    fprintf(stderr, "      Compute:        %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_compute_ms, nf > 0 ? t.t_code_pred_compute_ms / nf : 0.0);
    fprintf(stderr, "      Data I/O:       %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_data_ms, nf > 0 ? t.t_code_pred_data_ms / nf : 0.0);
    fprintf(stderr, "      CoreML total:   %8.1f ms   (%.1f ms/frame)\n", t.t_code_pred_coreml_ms, nf > 0 ? t.t_code_pred_coreml_ms / nf : 0.0);
    fprintf(stderr, "\n  Embed lookups:      %8.1f ms   (%.1f ms/frame)\n", t.t_embed_lookup_ms, nf > 0 ? t.t_embed_lookup_ms / nf : 0.0);
    double accounted = t.t_prefill_build_ms + t.t_prefill_forward_ms + t.t_talker_forward_ms + t.t_code_pred_ms + t.t_embed_lookup_ms;
    fprintf(stderr, "  Other/overhead:     %8.1f ms\n", t.t_generate_total_ms - accounted);
    fprintf(stderr, "  ─────────────────────────────────────────\n");
    fprintf(stderr, "  Total generate:     %8.1f ms\n", t.t_generate_total_ms);
    if (nf > 0) {
        fprintf(stderr, "  Throughput:         %8.1f ms/frame (%.1f frames/s)\n",
                t.t_generate_total_ms / nf, 1000.0 * nf / t.t_generate_total_ms);
    }
#endif

    return true;
}

bool TTSTransformer::forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                              std::vector<float> & output) {
    return forward_text(tokens, n_tokens, nullptr, n_past, output);
}

bool TTSTransformer::forward_with_audio(const int32_t * tokens, int32_t n_tokens,
                                         const float * audio_embd, int32_t n_audio,
                                         int32_t audio_start_pos, int32_t n_past,
                                         std::vector<float> & output) {
    (void)audio_embd;
    (void)n_audio;
    (void)audio_start_pos;
    return forward_text(tokens, n_tokens, nullptr, n_past, output);
}

void free_transformer_model(tts_transformer_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
    model.layers.clear();
    model.code_pred_layers.clear();
    model.code_pred_small_to_mtp_weight = nullptr;
    model.code_pred_small_to_mtp_bias = nullptr;
    model.code_pred_output_norm = nullptr;
    model.code_pred_embd.clear();
    model.code_pred_head.clear();
}

void free_tts_kv_cache(tts_kv_cache & cache) {
    if (cache.buffer) {
        ggml_backend_buffer_free(cache.buffer);
        cache.buffer = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.n_ctx = 0;
    cache.n_used = 0;
}

} // namespace qwen3_tts
