#include "audio_tokenizer_decoder.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

#define QWEN3_TTS_DEC_MAX_NODES 32768

namespace qwen3_tts {

AudioTokenizerDecoder::AudioTokenizerDecoder() = default;

AudioTokenizerDecoder::~AudioTokenizerDecoder() {
    free_audio_decoder_model(model_);
    
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        ggml_backend_free(state_.backend);
        state_.backend = nullptr;
    }
}

void AudioTokenizerDecoder::normalize_codebooks() {
    const float epsilon = 1e-5f;
    
    auto normalize_codebook = [epsilon](struct ggml_tensor * codebook, struct ggml_tensor * usage, const char * name) {
        if (!codebook || !usage || !codebook->data || !usage->data) return;
        
        int64_t codebook_dim = codebook->ne[0];
        int64_t codebook_size = codebook->ne[1];
        
        ggml_fp16_t * cb_data = (ggml_fp16_t *)codebook->data;
        float * usage_data = (float *)usage->data;
        
        for (int64_t emb_idx = 0; emb_idx < codebook_size; ++emb_idx) {
            float u = usage_data[emb_idx];
            if (u < epsilon) u = epsilon;
            float inv_u = 1.0f / u;
            
            for (int64_t dim_idx = 0; dim_idx < codebook_dim; ++dim_idx) {
                int64_t mem_idx = dim_idx + emb_idx * codebook_dim;
                float val = ggml_fp16_to_fp32(cb_data[mem_idx]);
                cb_data[mem_idx] = ggml_fp32_to_fp16(val * inv_u);
            }
        }
        
        if (strcmp(name, "first") == 0) {
            fprintf(stderr, "DEBUG: %s codebook entry 1221 first 5: ", name);
            for (int i = 0; i < 5; ++i) {
                int64_t idx = i + 1221 * codebook_dim;
                fprintf(stderr, "%.4f ", ggml_fp16_to_fp32(cb_data[idx]));
            }
            fprintf(stderr, "\n");
        }
        if (strcmp(name, "rest0") == 0) {
            fprintf(stderr, "DEBUG: %s codebook entry 472 first 5: ", name);
            for (int i = 0; i < 5; ++i) {
                int64_t idx = i + 472 * codebook_dim;
                fprintf(stderr, "%.4f ", ggml_fp16_to_fp32(cb_data[idx]));
            }
            fprintf(stderr, "\n");
        }
    };
    
    normalize_codebook(model_.vq_first_codebook, model_.vq_first_usage, "first");
    
    for (int i = 0; i < 15; ++i) {
        char name[16];
        snprintf(name, sizeof(name), "rest%d", i);
        normalize_codebook(model_.vq_rest_codebook[i], model_.vq_rest_usage[i], name);
    }
}

bool AudioTokenizerDecoder::load_model(const std::string & model_path) {
    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }
    
    model_.config.sample_rate = loader.get_u32("qwen3-tts.tokenizer.sample_rate", 24000);
    model_.config.n_codebooks = loader.get_u32("qwen3-tts.tokenizer.num_codebooks", 16);
    model_.config.codebook_size = loader.get_u32("qwen3-tts.tokenizer.codebook_size", 2048);
    
    int64_t n_tensors = loader.get_n_tensors();
    int dec_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "tok_dec.", 8) == 0) {
            dec_tensor_count++;
        }
    }
    
    if (dec_tensor_count == 0) {
        error_msg_ = "No decoder tensors found in model";
        return false;
    }
    
    size_t ctx_size = ggml_tensor_overhead() * dec_tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to initialize GGML context";
        return false;
    }
    
    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "tok_dec.", 8) != 0) {
            continue;
        }
        

        
        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta_ctx, name);
        if (!meta_tensor) {
            continue;
        }
        
        struct ggml_tensor * tensor = ggml_dup_tensor(model_.ctx, meta_tensor);
        ggml_set_name(tensor, name);
        
        model_.tensors[name] = tensor;
        
        std::string sname(name);
        
        if (sname == "tok_dec.vq_first.input_proj.weight") model_.vq_first_input_proj = tensor;
        else if (sname == "tok_dec.vq_first.output_proj.weight") model_.vq_first_output_proj = tensor;
        else if (sname == "tok_dec.vq_first.0.codebook") model_.vq_first_codebook = tensor;
        else if (sname == "tok_dec.vq_first.0.usage") model_.vq_first_usage = tensor;
        else if (sname == "tok_dec.vq_rest.input_proj.weight") model_.vq_rest_input_proj = tensor;
        else if (sname == "tok_dec.vq_rest.output_proj.weight") model_.vq_rest_output_proj = tensor;
        else if (sname == "tok_dec.pre_conv.weight") model_.pre_conv_w = tensor;
        else if (sname == "tok_dec.pre_conv.bias") model_.pre_conv_b = tensor;
        else if (sname == "tok_dec.pre_tfm.input_proj.weight") model_.pre_tfm_input_proj_w = tensor;
        else if (sname == "tok_dec.pre_tfm.input_proj.bias") model_.pre_tfm_input_proj_b = tensor;
        else if (sname == "tok_dec.pre_tfm.norm.weight") model_.pre_tfm_norm_w = tensor;
        else if (sname == "tok_dec.pre_tfm.output_proj.weight") model_.pre_tfm_output_proj_w = tensor;
        else if (sname == "tok_dec.pre_tfm.output_proj.bias") model_.pre_tfm_output_proj_b = tensor;
        else if (sname == "tok_dec.dec.0.conv.weight") model_.dec0_conv_w = tensor;
        else if (sname == "tok_dec.dec.0.conv.bias") model_.dec0_conv_b = tensor;
        else if (sname == "tok_dec.dec.5.snake.alpha") model_.dec5_snake_alpha = tensor;
        else if (sname == "tok_dec.dec.5.snake.beta") model_.dec5_snake_beta = tensor;
        else if (sname == "tok_dec.dec.6.conv.weight") model_.dec6_conv_w = tensor;
        else if (sname == "tok_dec.dec.6.conv.bias") model_.dec6_conv_b = tensor;
        else if (sname.find("pre_tfm.blk.") != std::string::npos) {
            int blk_idx;
            if (sscanf(name, "tok_dec.pre_tfm.blk.%d.", &blk_idx) == 1 && blk_idx >= 0 && blk_idx < 8) {
                if (sname.find(".attn_v.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_v_w = tensor;
                else if (sname.find(".ffn_gate.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_gate_w = tensor;
                else if (sname.find(".attn_norm.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_norm_w = tensor;
                else if (sname.find(".attn_q.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_q_w = tensor;
                else if (sname.find(".attn_k.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_k_w = tensor;
                else if (sname.find(".attn_output.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_output_w = tensor;
                else if (sname.find(".attn_scale") != std::string::npos) model_.pre_tfm_layers[blk_idx].attn_scale = tensor;
                else if (sname.find(".ffn_norm.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_norm_w = tensor;
                else if (sname.find(".ffn_up.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_up_w = tensor;
                else if (sname.find(".ffn_down.weight") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_down_w = tensor;
                else if (sname.find(".ffn_scale") != std::string::npos) model_.pre_tfm_layers[blk_idx].ffn_scale = tensor;
            }
        }
        else {
            int blk_idx, res_idx, cb_idx, n;
            char suffix[64];
            size_t name_len = strlen(name);
            

            
            #define MATCH1(fmt, var) (sscanf(name, fmt "%n", &var, &n) == 1 && (size_t)n == name_len)
            #define MATCH2(fmt, v1, v2) (sscanf(name, fmt "%n", &v1, &v2, &n) == 2 && (size_t)n == name_len)
            #define MATCH1S(fmt, var, suf) (sscanf(name, fmt, &var, suf) == 2)
            
            if (MATCH1("tok_dec.vq_rest.%d.codebook", cb_idx)) {
                if (cb_idx >= 0 && cb_idx < 15) {
                    model_.vq_rest_codebook[cb_idx] = tensor;
                }
            }
            else if (MATCH1("tok_dec.vq_rest.%d.usage", cb_idx)) {
                if (cb_idx >= 0 && cb_idx < 15) {
                    model_.vq_rest_usage[cb_idx] = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.conv.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].conv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].conv_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.dwconv.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].dwconv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].dwconv_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.norm.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].norm_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].norm_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.pwconv1.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].pwconv1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].pwconv1_b = tensor;
                }
            }
            else if (MATCH1S("tok_dec.upsample.%d.pwconv2.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model_.upsample[blk_idx].pwconv2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.upsample[blk_idx].pwconv2_b = tensor;
                }
            }
            else if (MATCH1("tok_dec.upsample.%d.gamma", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 2) model_.upsample[blk_idx].gamma = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_norm.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_norm_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_q.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_q_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_k.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_k_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_v.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_v_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_output.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_output_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_scale", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].attn_scale = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_norm.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_norm_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_gate.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_gate_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_up.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_up_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_down.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_down_w = tensor;
            }
            else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_scale", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model_.pre_tfm_layers[blk_idx].ffn_scale = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.snake.alpha", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].snake_alpha = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.snake.beta", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].snake_beta = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.conv_t.weight", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].conv_t_w = tensor;
            }
            else if (MATCH1("tok_dec.dec.%d.conv_t.bias", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model_.dec_blocks[blk_idx-1].conv_t_b = tensor;
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act1.alpha", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act1_alpha = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act1.beta", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act1_beta = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.weight", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv1_w = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.bias", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv1_b = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act2.alpha", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act2_alpha = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.act2.beta", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].act2_beta = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.weight", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv2_w = tensor;
                }
            }
            else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.bias", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model_.dec_blocks[blk_idx-1].res[res_idx-2].conv2_b = tensor;
                }
            }
            #undef MATCH1
            #undef MATCH2
            #undef MATCH1S
        }
    }
    
    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx, 
                                     model_.tensors, model_.buffer, error_msg_)) {
        return false;
    }
    
    for (int i = 0; i < 4; ++i) {
        model_.dec_blocks[i].res[0].dilation = 1;
        model_.dec_blocks[i].res[1].dilation = 3;
        model_.dec_blocks[i].res[2].dilation = 9;
    }
    
    normalize_codebooks();
    
    state_.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!state_.backend) {
        error_msg_ = "Failed to initialize CPU backend";
        return false;
    }
    
    std::vector<ggml_backend_t> backends = { state_.backend };
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, 1, QWEN3_TTS_DEC_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_DEC_MAX_NODES + ggml_graph_overhead());
    
    return true;
}

struct ggml_tensor * AudioTokenizerDecoder::apply_snake(struct ggml_context * ctx,
                                                         struct ggml_tensor * x,
                                                         struct ggml_tensor * alpha,
                                                         struct ggml_tensor * beta) {
    int64_t seq_len = x->ne[0];
    int64_t channels = x->ne[1];
    int64_t batch = x->ne[2];
    
    struct ggml_tensor * alpha_exp = ggml_exp(ctx, alpha);
    
    struct ggml_tensor * alpha_3d = ggml_reshape_3d(ctx, alpha_exp, 1, channels, 1);
    struct ggml_tensor * alpha_broad = ggml_repeat(ctx, alpha_3d, 
                                                    ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));
    
    struct ggml_tensor * ax = ggml_mul(ctx, x, alpha_broad);
    struct ggml_tensor * sin_ax = ggml_sin(ctx, ax);
    struct ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);
    
    struct ggml_tensor * neg_beta = ggml_scale(ctx, beta, -1.0f);
    struct ggml_tensor * inv_beta_exp = ggml_exp(ctx, neg_beta);
    struct ggml_tensor * inv_beta_3d = ggml_reshape_3d(ctx, inv_beta_exp, 1, channels, 1);
    struct ggml_tensor * inv_beta = ggml_repeat(ctx, inv_beta_3d, 
                                                 ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));
    
    struct ggml_tensor * scaled_sin = ggml_mul(ctx, sin_sq, inv_beta);
    
    return ggml_add(ctx, x, scaled_sin);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_rms_norm(struct ggml_context * ctx,
                                                            struct ggml_tensor * x,
                                                            struct ggml_tensor * w,
                                                            float eps) {
    struct ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normed, w);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_pre_tfm_layer(struct ggml_context * ctx,
                                                                 struct ggml_tensor * x,
                                                                 const pre_tfm_layer & layer,
                                                                 int32_t n_frames,
                                                                 struct ggml_tensor * positions) {
    const auto & cfg = model_.config;
    const int n_heads = cfg.n_heads;
    const int qkv_dim = cfg.latent_dim;
    const int head_dim = qkv_dim / n_heads;
    
    if (!layer.attn_norm_w || !layer.attn_q_w || !layer.attn_k_w || !layer.attn_v_w ||
        !layer.attn_output_w || !layer.ffn_norm_w || !layer.ffn_gate_w || 
        !layer.ffn_up_w || !layer.ffn_down_w) {
        return x;
    }
    
    struct ggml_tensor * residual = x;
    
    struct ggml_tensor * normed = apply_rms_norm(ctx, x, layer.attn_norm_w, cfg.rms_norm_eps);
    
    struct ggml_tensor * Qcur = ggml_mul_mat(ctx, layer.attn_q_w, normed);
    struct ggml_tensor * Kcur = ggml_mul_mat(ctx, layer.attn_k_w, normed);
    struct ggml_tensor * Vcur = ggml_mul_mat(ctx, layer.attn_v_w, normed);
    
    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_heads, n_frames);
    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_heads, n_frames);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_heads, n_frames);
    
    Qcur = ggml_rope_ext(ctx, Qcur, positions, nullptr,
                         head_dim, GGML_ROPE_TYPE_NEOX, 0,
                         cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    
    Kcur = ggml_rope_ext(ctx, Kcur, positions, nullptr,
                         head_dim, GGML_ROPE_TYPE_NEOX, 0,
                         cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    
    struct ggml_tensor * Q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
    struct ggml_tensor * K = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
    struct ggml_tensor * V = ggml_permute(ctx, Vcur, 0, 2, 1, 3);
    
    struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
    KQ = ggml_scale(ctx, KQ, 1.0f / sqrtf((float)head_dim));
    // Apply causal mask (each position can only attend to itself and previous positions)
    KQ = ggml_diag_mask_inf(ctx, KQ, 0);
    KQ = ggml_soft_max(ctx, KQ);
    
    V = ggml_cont(ctx, ggml_transpose(ctx, V));
    
    struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);
    KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
    struct ggml_tensor * attn_out = ggml_cont_2d(ctx, KQV, n_heads * head_dim, n_frames);
    
    attn_out = ggml_mul_mat(ctx, layer.attn_output_w, attn_out);
    
    if (layer.attn_scale) {
        attn_out = ggml_mul(ctx, attn_out, layer.attn_scale);
    }
    
    x = ggml_add(ctx, residual, attn_out);
    residual = x;
    
    normed = apply_rms_norm(ctx, x, layer.ffn_norm_w, cfg.rms_norm_eps);
    
    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate_w, normed);
    struct ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up_w, normed);
    
    gate = ggml_silu(ctx, gate);
    struct ggml_tensor * ffn_out = ggml_mul(ctx, gate, up);
    
    ffn_out = ggml_mul_mat(ctx, layer.ffn_down_w, ffn_out);
    
    if (layer.ffn_scale) {
        ffn_out = ggml_mul(ctx, ffn_out, layer.ffn_scale);
    }
    
    return ggml_add(ctx, residual, ffn_out);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_upsample_block(struct ggml_context * ctx,
                                                                   struct ggml_tensor * x,
                                                                   const upsample_block & block,
                                                                   int block_idx) {
    int64_t seq_len = x->ne[0];
    int64_t channels = x->ne[1];
    
    struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, seq_len, channels);
    x_2d = ggml_conv_transpose_1d(ctx, block.conv_w, x_2d, 2, 0, 1);
    
    int64_t new_seq_len = x_2d->ne[0];
    x = ggml_reshape_3d(ctx, x_2d, new_seq_len, channels, 1);
    
    if (block.conv_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_b, 1, channels, 1));
    }
    
    // Debug: after conv_transpose
    if (block_idx == 0) {
        char name[64];
        snprintf(name, sizeof(name), "up%d_conv_t", block_idx);
        ggml_set_name(x, name);
        ggml_set_output(x);
    }
    
    struct ggml_tensor * residual = x;
    
    if (block.dwconv_w) {
        // Causal padding: pad left with 6 zeros (kernel_size - 1 = 7 - 1 = 6)
        x = ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0);  // left pad only
        x = ggml_conv_1d_dw(ctx, block.dwconv_w, x, 1, 0, 1);  // no padding in conv
        if (block.dwconv_b) {
            x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.dwconv_b, 1, channels, 1));
        }
    }
    
    // Debug: after dwconv
    if (block_idx == 0) {
        char name[64];
        snprintf(name, sizeof(name), "up%d_dwconv", block_idx);
        ggml_set_name(x, name);
        ggml_set_output(x);
    }
    
    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    
    if (block.norm_w && block.norm_b) {
        x = ggml_norm(ctx, x, 1e-6f);
        x = ggml_mul(ctx, x, block.norm_w);
        x = ggml_add(ctx, x, block.norm_b);
    }
    
    // Debug: after norm
    if (block_idx == 0) {
        char name[64];
        snprintf(name, sizeof(name), "up%d_norm", block_idx);
        ggml_set_name(x, name);
        ggml_set_output(x);
    }
    
    x = ggml_mul_mat(ctx, block.pwconv1_w, x);
    if (block.pwconv1_b) {
        x = ggml_add(ctx, x, block.pwconv1_b);
    }
    
    // Debug: after pwconv1
    if (block_idx == 0) {
        char name[64];
        snprintf(name, sizeof(name), "up%d_pwconv1", block_idx);
        ggml_set_name(x, name);
        ggml_set_output(x);
    }
    
    x = ggml_gelu(ctx, x);
    
    // Debug: after gelu
    if (block_idx == 0) {
        char name[64];
        snprintf(name, sizeof(name), "up%d_gelu", block_idx);
        ggml_set_name(x, name);
        ggml_set_output(x);
    }
    
    x = ggml_mul_mat(ctx, block.pwconv2_w, x);
    if (block.pwconv2_b) {
        x = ggml_add(ctx, x, block.pwconv2_b);
    }
    
    // Debug: after pwconv2
    if (block_idx == 0) {
        char name[64];
        snprintf(name, sizeof(name), "up%d_pwconv2", block_idx);
        ggml_set_name(x, name);
        ggml_set_output(x);
    }
    
    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);
    
    if (block.gamma) {
        struct ggml_tensor * gamma_3d = ggml_reshape_3d(ctx, block.gamma, 1, channels, 1);
        x = ggml_mul(ctx, x, ggml_repeat(ctx, gamma_3d, 
                                          ggml_new_tensor_3d(ctx, GGML_TYPE_F32, new_seq_len, channels, 1)));
    }
    
    // Debug: after gamma
    if (block_idx == 0) {
        char name[64];
        snprintf(name, sizeof(name), "up%d_gamma", block_idx);
        ggml_set_name(x, name);
        ggml_set_output(x);
    }
    
    return ggml_add(ctx, residual, x);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_residual_block(struct ggml_context * ctx,
                                                                  struct ggml_tensor * x,
                                                                  const residual_block & block) {
    struct ggml_tensor * residual = x;
    
    if (block.act1_alpha) {
        x = apply_snake(ctx, x, block.act1_alpha, block.act1_beta);
    }
    
    int64_t out_channels = block.conv1_w->ne[2];
    int padding = 6 * block.dilation;
    x = ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0);
    x = ggml_conv_1d(ctx, block.conv1_w, x, 1, 0, block.dilation);
    if (block.conv1_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv1_b, 1, out_channels, 1));
    }
    
    if (block.act2_alpha) {
        x = apply_snake(ctx, x, block.act2_alpha, block.act2_beta);
    }
    
    out_channels = block.conv2_w->ne[2];
    x = ggml_conv_1d(ctx, block.conv2_w, x, 1, 0, 1);
    if (block.conv2_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv2_b, 1, out_channels, 1));
    }
    
    return ggml_add(ctx, residual, x);
}

struct ggml_tensor * AudioTokenizerDecoder::apply_decoder_block(struct ggml_context * ctx,
                                                                  struct ggml_tensor * x,
                                                                  const decoder_block & block,
                                                                  int upsample_rate,
                                                                  int block_idx) {
    if (block_idx == 0) {
        fprintf(stderr, "DEBUG: apply_decoder_block[0] snake_alpha=%p, snake_beta=%p\n",
                (void*)block.snake_alpha, (void*)block.snake_beta);
    }
    
    if (block.snake_alpha && block.snake_beta) {
        x = apply_snake(ctx, x, block.snake_alpha, block.snake_beta);
        if (block_idx == 0) {
            fprintf(stderr, "DEBUG: apply_decoder_block[0] snake applied\n");
        }
    } else {
        if (block_idx == 0) {
            fprintf(stderr, "DEBUG: apply_decoder_block[0] snake NOT applied\n");
        }
    }
    
    if (block_idx == 0) {
        ggml_set_name(x, "dec1_after_snake");
        ggml_set_output(x);
    }
    
    int64_t seq_len = x->ne[0];
    int64_t in_channels = x->ne[1];
    int64_t out_channels = block.conv_t_w->ne[1];
    int kernel_size = block.conv_t_w->ne[0];
    
    struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, seq_len, in_channels);
    x_2d = ggml_conv_transpose_1d(ctx, block.conv_t_w, x_2d, upsample_rate, 0, 1);
    
    int64_t new_seq_len = x_2d->ne[0];
    x = ggml_reshape_3d(ctx, x_2d, new_seq_len, out_channels, 1);
    
    if (block_idx == 0) {
        ggml_set_name(x, "dec1_after_conv_t_raw");
        ggml_set_output(x);
    }
    
    // Python CausalTransConvNet: left_pad = right_pad = kernel_size - stride
    int pad = kernel_size - upsample_rate;
    int left_pad = pad;
    int right_pad = pad;
    int64_t out_seq_len = new_seq_len - left_pad - right_pad;
    
    x = ggml_view_3d(ctx, x, out_seq_len, out_channels, 1,
                     x->nb[1], x->nb[2], left_pad * x->nb[0]);
    x = ggml_cont(ctx, x);
    
    if (block_idx == 0) {
        ggml_set_name(x, "dec1_after_trim");
        ggml_set_output(x);
    }
    
    if (block.conv_t_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_t_b, 1, out_channels, 1));
    }
    
    if (block_idx == 0) {
        ggml_set_name(x, "dec1_after_bias");
        ggml_set_output(x);
    }
    
    for (int i = 0; i < 3; ++i) {
        x = apply_residual_block(ctx, x, block.res[i]);
    }
    
    return x;
}

struct ggml_cgraph * AudioTokenizerDecoder::build_graph(int32_t n_frames) {
    const auto & cfg = model_.config;
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_DEC_MAX_NODES, false);
    
    static const char * cb_names[16] = {
        "codes_cb0", "codes_cb1", "codes_cb2", "codes_cb3",
        "codes_cb4", "codes_cb5", "codes_cb6", "codes_cb7",
        "codes_cb8", "codes_cb9", "codes_cb10", "codes_cb11",
        "codes_cb12", "codes_cb13", "codes_cb14", "codes_cb15"
    };
    
    struct ggml_tensor * cb_codes_tensors[16];
    for (int cb = 0; cb < 16; ++cb) {
        cb_codes_tensors[cb] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
        ggml_set_name(cb_codes_tensors[cb], cb_names[cb]);
        ggml_set_input(cb_codes_tensors[cb]);
    }
    
    struct ggml_tensor * first_codes = cb_codes_tensors[0];
    
    struct ggml_tensor * first_emb = ggml_get_rows(ctx0, model_.vq_first_codebook, first_codes);
    ggml_set_name(first_emb, "first_emb_raw");
    ggml_set_output(first_emb);
    
    struct ggml_tensor * rest_emb[15];
    for (int cb = 0; cb < 15; ++cb) {
        struct ggml_tensor * cb_codes = cb_codes_tensors[cb + 1];
        rest_emb[cb] = ggml_get_rows(ctx0, model_.vq_rest_codebook[cb], cb_codes);
        
        if (cb == 0) {
            ggml_set_name(rest_emb[cb], "rest_cb0_emb_raw");
            ggml_set_output(rest_emb[cb]);
        }
    }
    
    ggml_set_name(first_emb, "first_emb_raw");
    ggml_set_output(first_emb);
    
    struct ggml_tensor * first_emb_2d = ggml_reshape_2d(ctx0, first_emb, cfg.codebook_dim, n_frames);
    ggml_set_name(first_emb_2d, "first_emb_2d");
    ggml_set_output(first_emb_2d);
    
    struct ggml_tensor * first_proj_weight_2d = ggml_reshape_2d(ctx0, model_.vq_first_output_proj, 
                                                                  cfg.codebook_dim, cfg.hidden_dim);
    struct ggml_tensor * first_proj_2d = ggml_mul_mat(ctx0, first_proj_weight_2d, first_emb_2d);
    ggml_set_name(first_proj_2d, "first_proj_2d");
    ggml_set_output(first_proj_2d);
    
    struct ggml_tensor * rest_proj_weight_2d = ggml_reshape_2d(ctx0, model_.vq_rest_output_proj,
                                                                 cfg.codebook_dim, cfg.hidden_dim);
    
    struct ggml_tensor * rest_proj_2d = nullptr;
    for (int cb = 0; cb < 15; ++cb) {
        struct ggml_tensor * cb_emb_2d = ggml_reshape_2d(ctx0, rest_emb[cb], cfg.codebook_dim, n_frames);
        
        if (cb == 0) {
            ggml_set_name(cb_emb_2d, "rest_cb0_emb_2d");
            ggml_set_output(cb_emb_2d);
        }
        
        struct ggml_tensor * cb_proj_2d = ggml_mul_mat(ctx0, rest_proj_weight_2d, cb_emb_2d);
        
        if (rest_proj_2d == nullptr) {
            rest_proj_2d = cb_proj_2d;
        } else {
            rest_proj_2d = ggml_add(ctx0, rest_proj_2d, cb_proj_2d);
        }
    }
    ggml_set_name(rest_proj_2d, "rest_proj_2d");
    ggml_set_output(rest_proj_2d);
    
    struct ggml_tensor * latent_2d = ggml_add(ctx0, first_proj_2d, rest_proj_2d);
    ggml_set_name(latent_2d, "latent_2d");
    ggml_set_output(latent_2d);
    
    struct ggml_tensor * latent_t = ggml_transpose(ctx0, latent_2d);
    ggml_set_name(latent_t, "latent_t");
    ggml_set_output(latent_t);
    
    struct ggml_tensor * latent_cont = ggml_cont(ctx0, latent_t);
    ggml_set_name(latent_cont, "latent_cont");
    ggml_set_output(latent_cont);
    
    struct ggml_tensor * latent = ggml_reshape_3d(ctx0, latent_cont, n_frames, cfg.hidden_dim, 1);
    
    ggml_set_name(latent, "vq_output");
    ggml_set_output(latent);
    
    struct ggml_tensor * latent_for_conv = ggml_cont(ctx0, latent);
    struct ggml_tensor * latent_padded = ggml_pad_ext(ctx0, latent_for_conv, 2, 0, 0, 0, 0, 0, 0, 0);
    struct ggml_tensor * cur = ggml_conv_1d(ctx0, model_.pre_conv_w, latent_padded, 1, 0, 1);
    if (model_.pre_conv_b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.pre_conv_b, 1, cfg.latent_dim, 1));
    }
    
    ggml_set_name(cur, "pre_conv_output");
    ggml_set_output(cur);
    
    struct ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, n_frames, cfg.latent_dim);
    struct ggml_tensor * cur_t = ggml_transpose(ctx0, cur_2d);
    cur = ggml_cont(ctx0, cur_t);
    
    ggml_set_name(cur, "pre_conv_reshaped");
    ggml_set_output(cur);
    
    cur = ggml_mul_mat(ctx0, model_.pre_tfm_input_proj_w, cur);
    if (model_.pre_tfm_input_proj_b) {
        cur = ggml_add(ctx0, cur, model_.pre_tfm_input_proj_b);
    }
    
    ggml_set_name(cur, "pre_tfm_input");
    ggml_set_output(cur);
    
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    
    for (int i = 0; i < cfg.n_pre_tfm_layers; ++i) {
        cur = apply_pre_tfm_layer(ctx0, cur, model_.pre_tfm_layers[i], n_frames, positions);
        if (i == 0) {
            ggml_set_name(cur, "layer0_output");
            ggml_set_output(cur);
        }
    }
    
    if (model_.pre_tfm_norm_w) {
        cur = apply_rms_norm(ctx0, cur, model_.pre_tfm_norm_w, cfg.rms_norm_eps);
    }
    
    cur = ggml_mul_mat(ctx0, model_.pre_tfm_output_proj_w, cur);
    if (model_.pre_tfm_output_proj_b) {
        cur = ggml_add(ctx0, cur, model_.pre_tfm_output_proj_b);
    }
    
    ggml_set_name(cur, "pre_tfm_output");
    ggml_set_output(cur);
    
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_3d(ctx0, cur, n_frames, cfg.latent_dim, 1);
    
    ggml_set_name(cur, "pre_tfm_reshaped");
    ggml_set_output(cur);
    
    for (int i = 0; i < 2; ++i) {
        cur = apply_upsample_block(ctx0, cur, model_.upsample[i], i);
    }
    
    ggml_set_name(cur, "upsample_output");
    ggml_set_output(cur);
    
    // Causal padding: left pad with 6 (kernel_size - 1 = 7 - 1 = 6)
    cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, model_.dec0_conv_w, cur, 1, 0, 1);
    if (model_.dec0_conv_b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.dec0_conv_b, 1, cfg.decoder_dim, 1));
    }
    
    ggml_set_name(cur, "dec0_output");
    ggml_set_output(cur);
    
    int upsample_rates[4] = {8, 5, 4, 3};
    for (int i = 0; i < 4; ++i) {
        cur = apply_decoder_block(ctx0, cur, model_.dec_blocks[i], upsample_rates[i], i);
        char name[32];
        snprintf(name, sizeof(name), "dec%d_output", i + 1);
        ggml_set_name(cur, name);
        ggml_set_output(cur);
    }
    
    if (model_.dec5_snake_alpha) {
        cur = apply_snake(ctx0, cur, model_.dec5_snake_alpha, model_.dec5_snake_beta);
    }
    
    ggml_set_name(cur, "dec5_output");
    ggml_set_output(cur);
    
    // Causal padding: left pad with 6 (kernel_size - 1 = 7 - 1 = 6)
    cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, model_.dec6_conv_w, cur, 1, 0, 1);
    if (model_.dec6_conv_b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.dec6_conv_b, 1, 1, 1));
    }
    
    ggml_set_name(cur, "dec6_output");
    ggml_set_output(cur);
    
    cur = ggml_tanh(ctx0, cur);
    
    cur = ggml_reshape_1d(ctx0, cur, cur->ne[0]);
    
    ggml_set_name(cur, "audio");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

bool AudioTokenizerDecoder::decode(const int32_t * codes, int32_t n_frames,
                                    std::vector<float> & samples) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    const auto & cfg = model_.config;
    
    codes_buf_.resize(n_frames * cfg.n_codebooks);
    for (int f = 0; f < n_frames; ++f) {
        for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
            codes_buf_[cb + f * cfg.n_codebooks] = codes[f * cfg.n_codebooks + cb];
        }
    }
    
    struct weight_data {
        struct ggml_tensor * tensor;
        void * data;
        size_t nbytes;
    };
    std::vector<weight_data> weights_to_copy;
    
    auto save_weight = [&weights_to_copy](struct ggml_tensor * t) {
        if (t && t->data) {
            weights_to_copy.push_back({t, t->data, ggml_nbytes(t)});
        }
    };
    
    save_weight(model_.vq_first_input_proj);
    save_weight(model_.vq_first_output_proj);
    
    if (model_.vq_first_output_proj && model_.vq_first_output_proj->data) {
        ggml_fp16_t * w_data = (ggml_fp16_t *)model_.vq_first_output_proj->data;
        fprintf(stderr, "DEBUG: vq_first_output_proj shape [%lld, %lld, %lld], first 5: %.6f %.6f %.6f %.6f %.6f\n",
                (long long)model_.vq_first_output_proj->ne[0],
                (long long)model_.vq_first_output_proj->ne[1],
                (long long)model_.vq_first_output_proj->ne[2],
                ggml_fp16_to_fp32(w_data[0]), ggml_fp16_to_fp32(w_data[1]),
                ggml_fp16_to_fp32(w_data[2]), ggml_fp16_to_fp32(w_data[3]),
                ggml_fp16_to_fp32(w_data[4]));
    }
    
    save_weight(model_.vq_first_codebook);
    save_weight(model_.vq_first_usage);
    save_weight(model_.vq_rest_input_proj);
    save_weight(model_.vq_rest_output_proj);
    
    if (model_.vq_rest_output_proj && model_.vq_rest_output_proj->data) {
        ggml_fp16_t * w_data = (ggml_fp16_t *)model_.vq_rest_output_proj->data;
        fprintf(stderr, "DEBUG: vq_rest_output_proj shape [%lld, %lld, %lld], strides [%lld, %lld, %lld], first 5: %.6f %.6f %.6f %.6f %.6f\n",
                (long long)model_.vq_rest_output_proj->ne[0],
                (long long)model_.vq_rest_output_proj->ne[1],
                (long long)model_.vq_rest_output_proj->ne[2],
                (long long)model_.vq_rest_output_proj->nb[0],
                (long long)model_.vq_rest_output_proj->nb[1],
                (long long)model_.vq_rest_output_proj->nb[2],
                ggml_fp16_to_fp32(w_data[0]), ggml_fp16_to_fp32(w_data[1]),
                ggml_fp16_to_fp32(w_data[2]), ggml_fp16_to_fp32(w_data[3]),
                ggml_fp16_to_fp32(w_data[4]));
    }
    if (model_.vq_first_output_proj && model_.vq_first_output_proj->data) {
        fprintf(stderr, "DEBUG: vq_first_output_proj shape [%lld, %lld, %lld], strides [%lld, %lld, %lld]\n",
                (long long)model_.vq_first_output_proj->ne[0],
                (long long)model_.vq_first_output_proj->ne[1],
                (long long)model_.vq_first_output_proj->ne[2],
                (long long)model_.vq_first_output_proj->nb[0],
                (long long)model_.vq_first_output_proj->nb[1],
                (long long)model_.vq_first_output_proj->nb[2]);
    }
    for (int i = 0; i < 15; ++i) {
        save_weight(model_.vq_rest_codebook[i]);
        save_weight(model_.vq_rest_usage[i]);
    }
    
    for (int i = 0; i < 2; ++i) {
        save_weight(model_.upsample[i].conv_w);
        save_weight(model_.upsample[i].conv_b);
        save_weight(model_.upsample[i].dwconv_w);
        save_weight(model_.upsample[i].dwconv_b);
        save_weight(model_.upsample[i].norm_w);
        save_weight(model_.upsample[i].norm_b);
        save_weight(model_.upsample[i].pwconv1_w);
        save_weight(model_.upsample[i].pwconv1_b);
        save_weight(model_.upsample[i].pwconv2_w);
        save_weight(model_.upsample[i].pwconv2_b);
        save_weight(model_.upsample[i].gamma);
    }
    
    save_weight(model_.pre_tfm_input_proj_w);
    save_weight(model_.pre_tfm_input_proj_b);
    save_weight(model_.pre_tfm_norm_w);
    save_weight(model_.pre_tfm_output_proj_w);
    save_weight(model_.pre_tfm_output_proj_b);
    
    for (int i = 0; i < 8; ++i) {
        save_weight(model_.pre_tfm_layers[i].attn_norm_w);
        save_weight(model_.pre_tfm_layers[i].attn_q_w);
        save_weight(model_.pre_tfm_layers[i].attn_k_w);
        save_weight(model_.pre_tfm_layers[i].attn_v_w);
        save_weight(model_.pre_tfm_layers[i].attn_output_w);
        save_weight(model_.pre_tfm_layers[i].attn_scale);
        save_weight(model_.pre_tfm_layers[i].ffn_norm_w);
        save_weight(model_.pre_tfm_layers[i].ffn_gate_w);
        save_weight(model_.pre_tfm_layers[i].ffn_up_w);
        save_weight(model_.pre_tfm_layers[i].ffn_down_w);
        save_weight(model_.pre_tfm_layers[i].ffn_scale);
    }
    
    save_weight(model_.pre_conv_w);
    save_weight(model_.pre_conv_b);
    save_weight(model_.dec0_conv_w);
    save_weight(model_.dec0_conv_b);
    
    for (int i = 0; i < 4; ++i) {
        if (i == 0 && model_.dec_blocks[i].snake_alpha) {
            fprintf(stderr, "DEBUG: dec_blocks[0].snake_alpha = %p, data = %p\n", 
                    (void*)model_.dec_blocks[i].snake_alpha,
                    model_.dec_blocks[i].snake_alpha->data);
        }
        if (i == 0 && model_.dec_blocks[i].snake_beta) {
            fprintf(stderr, "DEBUG: dec_blocks[0].snake_beta = %p, data = %p\n",
                    (void*)model_.dec_blocks[i].snake_beta,
                    model_.dec_blocks[i].snake_beta->data);
        }
        save_weight(model_.dec_blocks[i].snake_alpha);
        save_weight(model_.dec_blocks[i].snake_beta);
        save_weight(model_.dec_blocks[i].conv_t_w);
        save_weight(model_.dec_blocks[i].conv_t_b);
        for (int j = 0; j < 3; ++j) {
            save_weight(model_.dec_blocks[i].res[j].act1_alpha);
            save_weight(model_.dec_blocks[i].res[j].act1_beta);
            save_weight(model_.dec_blocks[i].res[j].conv1_w);
            save_weight(model_.dec_blocks[i].res[j].conv1_b);
            save_weight(model_.dec_blocks[i].res[j].act2_alpha);
            save_weight(model_.dec_blocks[i].res[j].act2_beta);
            save_weight(model_.dec_blocks[i].res[j].conv2_w);
            save_weight(model_.dec_blocks[i].res[j].conv2_b);
        }
    }
    
    save_weight(model_.dec5_snake_alpha);
    save_weight(model_.dec5_snake_beta);
    save_weight(model_.dec6_conv_w);
    save_weight(model_.dec6_conv_b);
    
    struct ggml_cgraph * gf = build_graph(n_frames);
    
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
    
    for (const auto & w : weights_to_copy) {
        ggml_backend_tensor_set(w.tensor, w.data, 0, w.nbytes);
    }
    
    fprintf(stderr, "DEBUG: codes_buf_ first 5: %d %d %d %d %d\n",
            codes_buf_[0], codes_buf_[1], codes_buf_[2], codes_buf_[3], codes_buf_[4]);
    if (n_frames > 4) {
        fprintf(stderr, "DEBUG: codes_buf_[0,16,32,48,64]: %d %d %d %d %d\n",
                codes_buf_[0], codes_buf_[16], codes_buf_[32], codes_buf_[48], codes_buf_[64]);
    }
    
    for (int cb = 0; cb < 16; ++cb) {
        char name[32];
        snprintf(name, sizeof(name), "codes_cb%d", cb);
        struct ggml_tensor * cb_tensor = ggml_graph_get_tensor(gf, name);
        if (!cb_tensor) {
            error_msg_ = "Failed to find codes tensor for codebook " + std::to_string(cb);
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
        
        std::vector<int32_t> cb_codes(n_frames);
        for (int f = 0; f < n_frames; ++f) {
            cb_codes[f] = codes_buf_[f * cfg.n_codebooks + cb];
        }
        
        if (cb == 0) {
            fprintf(stderr, "DEBUG: cb0_codes first 5: %d %d %d %d %d\n",
                    cb_codes[0], cb_codes[1], cb_codes[2], cb_codes[3], cb_codes[4]);
        }
        if (cb == 1) {
            fprintf(stderr, "DEBUG: cb1_codes first 5: %d %d %d %d %d\n",
                    cb_codes[0], cb_codes[1], cb_codes[2], cb_codes[3], cb_codes[4]);
        }
        
        ggml_backend_tensor_set(cb_tensor, cb_codes.data(), 0, n_frames * sizeof(int32_t));
    }
    

    
    struct ggml_tensor * positions_tensor = ggml_graph_get_tensor(gf, "positions");
    if (positions_tensor) {
        std::vector<int32_t> positions(n_frames);
        for (int i = 0; i < n_frames; ++i) {
            positions[i] = i;
        }
        ggml_backend_tensor_set(positions_tensor, positions.data(), 0, 
                                n_frames * sizeof(int32_t));
    }
    

    
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    struct ggml_tensor * codes_cb0_readback = ggml_graph_get_tensor(gf, "codes_cb0");
    if (codes_cb0_readback) {
        std::vector<int32_t> data(ggml_nelements(codes_cb0_readback));
        ggml_backend_tensor_get(codes_cb0_readback, data.data(), 0, data.size() * sizeof(int32_t));
        fprintf(stderr, "DEBUG: codes_cb0 shape [%lld], first 5: %d %d %d %d %d\n",
                (long long)codes_cb0_readback->ne[0],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * codes_cb1_readback = ggml_graph_get_tensor(gf, "codes_cb1");
    if (codes_cb1_readback) {
        std::vector<int32_t> data(ggml_nelements(codes_cb1_readback));
        ggml_backend_tensor_get(codes_cb1_readback, data.data(), 0, data.size() * sizeof(int32_t));
        fprintf(stderr, "DEBUG: codes_cb1 shape [%lld], first 5: %d %d %d %d %d\n",
                (long long)codes_cb1_readback->ne[0],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * first_emb_raw = ggml_graph_get_tensor(gf, "first_emb_raw");
    if (first_emb_raw) {
        std::vector<float> data(ggml_nelements(first_emb_raw));
        ggml_backend_tensor_get(first_emb_raw, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: first_emb_raw shape [%lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)first_emb_raw->ne[0], (long long)first_emb_raw->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * first_emb_2d = ggml_graph_get_tensor(gf, "first_emb_2d");
    if (first_emb_2d) {
        std::vector<float> data(ggml_nelements(first_emb_2d));
        ggml_backend_tensor_get(first_emb_2d, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: first_emb_2d shape [%lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)first_emb_2d->ne[0], (long long)first_emb_2d->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * first_proj_2d = ggml_graph_get_tensor(gf, "first_proj_2d");
    if (first_proj_2d) {
        std::vector<float> data(ggml_nelements(first_proj_2d));
        ggml_backend_tensor_get(first_proj_2d, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: first_proj_2d shape [%lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)first_proj_2d->ne[0], (long long)first_proj_2d->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * rest_proj_2d = ggml_graph_get_tensor(gf, "rest_proj_2d");
    if (rest_proj_2d) {
        std::vector<float> data(ggml_nelements(rest_proj_2d));
        ggml_backend_tensor_get(rest_proj_2d, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: rest_proj_2d shape [%lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)rest_proj_2d->ne[0], (long long)rest_proj_2d->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * latent_2d = ggml_graph_get_tensor(gf, "latent_2d");
    if (latent_2d) {
        std::vector<float> data(ggml_nelements(latent_2d));
        ggml_backend_tensor_get(latent_2d, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: latent_2d shape [%lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)latent_2d->ne[0], (long long)latent_2d->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * latent_t = ggml_graph_get_tensor(gf, "latent_t");
    if (latent_t) {
        std::vector<float> data(ggml_nelements(latent_t));
        ggml_backend_tensor_get(latent_t, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: latent_t shape [%lld, %lld], strides [%lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)latent_t->ne[0], (long long)latent_t->ne[1],
                (long long)latent_t->nb[0], (long long)latent_t->nb[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * latent_cont = ggml_graph_get_tensor(gf, "latent_cont");
    if (latent_cont) {
        std::vector<float> data(ggml_nelements(latent_cont));
        ggml_backend_tensor_get(latent_cont, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: latent_cont shape [%lld, %lld], strides [%lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)latent_cont->ne[0], (long long)latent_cont->ne[1],
                (long long)latent_cont->nb[0], (long long)latent_cont->nb[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * rest_cb0_codes = ggml_graph_get_tensor(gf, "rest_cb0_codes");
    if (rest_cb0_codes) {
        std::vector<int32_t> data(ggml_nelements(rest_cb0_codes));
        ggml_backend_tensor_get(rest_cb0_codes, data.data(), 0, data.size() * sizeof(int32_t));
        fprintf(stderr, "DEBUG: rest_cb0_codes shape [%lld], first 5: %d %d %d %d %d\n",
                (long long)rest_cb0_codes->ne[0],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * rest_cb0_emb_raw = ggml_graph_get_tensor(gf, "rest_cb0_emb_raw");
    if (rest_cb0_emb_raw) {
        std::vector<float> data(ggml_nelements(rest_cb0_emb_raw));
        ggml_backend_tensor_get(rest_cb0_emb_raw, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: rest_cb0_emb_raw shape [%lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)rest_cb0_emb_raw->ne[0], (long long)rest_cb0_emb_raw->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * rest_cb0_emb_2d = ggml_graph_get_tensor(gf, "rest_cb0_emb_2d");
    if (rest_cb0_emb_2d) {
        std::vector<float> data(ggml_nelements(rest_cb0_emb_2d));
        ggml_backend_tensor_get(rest_cb0_emb_2d, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: rest_cb0_emb_2d shape [%lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)rest_cb0_emb_2d->ne[0], (long long)rest_cb0_emb_2d->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * rest_sum_tensor = ggml_graph_get_tensor(gf, "rest_sum");
    if (rest_sum_tensor) {
        std::vector<float> data(ggml_nelements(rest_sum_tensor));
        ggml_backend_tensor_get(rest_sum_tensor, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: rest_sum shape [%lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)rest_sum_tensor->ne[0], (long long)rest_sum_tensor->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * rest_sum_cont_tensor = ggml_graph_get_tensor(gf, "rest_sum_cont");
    if (rest_sum_cont_tensor) {
        std::vector<float> data(ggml_nelements(rest_sum_cont_tensor));
        ggml_backend_tensor_get(rest_sum_cont_tensor, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: rest_sum_cont shape [%lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)rest_sum_cont_tensor->ne[0], (long long)rest_sum_cont_tensor->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    if (model_.vq_rest_output_proj) {
        size_t nbytes = ggml_nbytes(model_.vq_rest_output_proj);
        std::vector<uint8_t> raw_data(nbytes);
        ggml_backend_tensor_get(model_.vq_rest_output_proj, raw_data.data(), 0, nbytes);
        std::vector<float> data(ggml_nelements(model_.vq_rest_output_proj));
        if (model_.vq_rest_output_proj->type == GGML_TYPE_F16) {
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = ggml_fp16_to_fp32(((ggml_fp16_t*)raw_data.data())[i]);
            }
        } else {
            memcpy(data.data(), raw_data.data(), nbytes);
        }
        fprintf(stderr, "DEBUG: vq_rest_output_proj (model) shape [%lld, %lld, %lld], type=%d, first 5: %.6f %.6f %.6f %.6f %.6f\n",
                (long long)model_.vq_rest_output_proj->ne[0], (long long)model_.vq_rest_output_proj->ne[1],
                (long long)model_.vq_rest_output_proj->ne[2], (int)model_.vq_rest_output_proj->type,
                data[0], data[1], data[2], data[3], data[4]);
        fprintf(stderr, "DEBUG: vq_rest_output_proj [256:261]: %.6f %.6f %.6f %.6f %.6f\n",
                data[256], data[257], data[258], data[259], data[260]);
        fprintf(stderr, "DEBUG: vq_rest_output_proj [512:517]: %.6f %.6f %.6f %.6f %.6f\n",
                data[512], data[513], data[514], data[515], data[516]);
    }
    
    struct ggml_tensor * rest_proj_tensor = ggml_graph_get_tensor(gf, "rest_proj");
    if (rest_proj_tensor) {
        std::vector<float> data(ggml_nelements(rest_proj_tensor));
        ggml_backend_tensor_get(rest_proj_tensor, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: rest_proj shape [%lld, %lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)rest_proj_tensor->ne[0], (long long)rest_proj_tensor->ne[1], (long long)rest_proj_tensor->ne[2],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * vq_tensor = ggml_graph_get_tensor(gf, "vq_output");
    if (vq_tensor) {
        std::vector<float> vq_data(ggml_nelements(vq_tensor));
        ggml_backend_tensor_get(vq_tensor, vq_data.data(), 0, vq_data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: VQ output shape [%lld, %lld, %lld], first 5: %.4f %.4f %.4f %.4f %.4f\n",
                (long long)vq_tensor->ne[0], (long long)vq_tensor->ne[1], (long long)vq_tensor->ne[2],
                vq_data[0], vq_data[1], vq_data[2], vq_data[3], vq_data[4]);
        FILE * f = fopen("debug_vq_output.bin", "wb");
        if (f) {
            fwrite(vq_data.data(), sizeof(float), vq_data.size(), f);
            fclose(f);
        }
    } else {
        fprintf(stderr, "DEBUG: vq_output tensor not found\n");
    }
    
    struct ggml_tensor * pre_conv_tensor = ggml_graph_get_tensor(gf, "pre_conv_output");
    if (pre_conv_tensor) {
        std::vector<float> pre_conv_data(ggml_nelements(pre_conv_tensor));
        ggml_backend_tensor_get(pre_conv_tensor, pre_conv_data.data(), 0, pre_conv_data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: pre_conv output shape [%lld, %lld, %lld], first 5: %.6f %.6f %.6f %.6f %.6f\n",
                (long long)pre_conv_tensor->ne[0], (long long)pre_conv_tensor->ne[1], (long long)pre_conv_tensor->ne[2],
                pre_conv_data[0], pre_conv_data[1], pre_conv_data[2], pre_conv_data[3], pre_conv_data[4]);
        FILE * f = fopen("debug_pre_conv_output.bin", "wb");
        if (f) {
            fwrite(pre_conv_data.data(), sizeof(float), pre_conv_data.size(), f);
            fclose(f);
        }
    } else {
        fprintf(stderr, "DEBUG: pre_conv_output tensor not found\n");
    }
    
    struct ggml_tensor * pre_conv_reshaped = ggml_graph_get_tensor(gf, "pre_conv_reshaped");
    if (pre_conv_reshaped) {
        std::vector<float> data(ggml_nelements(pre_conv_reshaped));
        ggml_backend_tensor_get(pre_conv_reshaped, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: pre_conv_reshaped shape [%lld, %lld], first 5: %.6f %.6f %.6f %.6f %.6f\n",
                (long long)pre_conv_reshaped->ne[0], (long long)pre_conv_reshaped->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    if (model_.pre_tfm_input_proj_w) {
        fprintf(stderr, "DEBUG: pre_tfm_input_proj_w shape [%lld, %lld]\n",
                (long long)model_.pre_tfm_input_proj_w->ne[0], (long long)model_.pre_tfm_input_proj_w->ne[1]);
    }
    
    fprintf(stderr, "DEBUG: layer0 weights loaded: attn_norm=%p, attn_q=%p, attn_k=%p, attn_v=%p, attn_out=%p\n",
            (void*)model_.pre_tfm_layers[0].attn_norm_w,
            (void*)model_.pre_tfm_layers[0].attn_q_w,
            (void*)model_.pre_tfm_layers[0].attn_k_w,
            (void*)model_.pre_tfm_layers[0].attn_v_w,
            (void*)model_.pre_tfm_layers[0].attn_output_w);
    fprintf(stderr, "DEBUG: layer0 ffn weights: norm=%p, gate=%p, up=%p, down=%p\n",
            (void*)model_.pre_tfm_layers[0].ffn_norm_w,
            (void*)model_.pre_tfm_layers[0].ffn_gate_w,
            (void*)model_.pre_tfm_layers[0].ffn_up_w,
            (void*)model_.pre_tfm_layers[0].ffn_down_w);
    
    struct ggml_tensor * pre_tfm_input = ggml_graph_get_tensor(gf, "pre_tfm_input");
    if (pre_tfm_input) {
        std::vector<float> data(ggml_nelements(pre_tfm_input));
        ggml_backend_tensor_get(pre_tfm_input, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: pre_tfm_input shape [%lld, %lld], first 5: %.6f %.6f %.6f %.6f %.6f\n",
                (long long)pre_tfm_input->ne[0], (long long)pre_tfm_input->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * layer0_output = ggml_graph_get_tensor(gf, "layer0_output");
    if (layer0_output) {
        std::vector<float> data(ggml_nelements(layer0_output));
        ggml_backend_tensor_get(layer0_output, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: layer0_output shape [%lld, %lld], first 5: %.6f %.6f %.6f %.6f %.6f\n",
                (long long)layer0_output->ne[0], (long long)layer0_output->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
    }
    
    struct ggml_tensor * pre_tfm_tensor = ggml_graph_get_tensor(gf, "pre_tfm_output");
    if (pre_tfm_tensor) {
        std::vector<float> data(ggml_nelements(pre_tfm_tensor));
        ggml_backend_tensor_get(pre_tfm_tensor, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: pre_tfm output shape [%lld, %lld], first 5: %.6f %.6f %.6f %.6f %.6f\n",
                (long long)pre_tfm_tensor->ne[0], (long long)pre_tfm_tensor->ne[1],
                data[0], data[1], data[2], data[3], data[4]);
        FILE * f = fopen("debug_pre_tfm_output.bin", "wb");
        if (f) {
            fwrite(data.data(), sizeof(float), data.size(), f);
            fclose(f);
        }
    }
    
    const char * up_step_names[] = {"up0_conv_t", "up0_dwconv", "up0_norm", "up0_pwconv1", "up0_gelu", "up0_pwconv2", "up0_gamma"};
    for (int i = 0; i < 7; ++i) {
        struct ggml_tensor * t = ggml_graph_get_tensor(gf, up_step_names[i]);
        if (t) {
            std::vector<float> data(ggml_nelements(t));
            ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
            fprintf(stderr, "DEBUG: %s shape [%lld, %lld, %lld], first 5: %.6f %.6f %.6f %.6f %.6f\n",
                    up_step_names[i],
                    (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                    data[0], data[1], data[2], data[3], data[4]);
            char fname[64];
            snprintf(fname, sizeof(fname), "debug_%s.bin", up_step_names[i]);
            FILE * f = fopen(fname, "wb");
            if (f) {
                fwrite(data.data(), sizeof(float), data.size(), f);
                fclose(f);
            }
        }
    }
    
    struct ggml_tensor * upsample_tensor = ggml_graph_get_tensor(gf, "upsample_output");
    if (upsample_tensor) {
        std::vector<float> data(ggml_nelements(upsample_tensor));
        ggml_backend_tensor_get(upsample_tensor, data.data(), 0, data.size() * sizeof(float));
        fprintf(stderr, "DEBUG: upsample output shape [%lld, %lld, %lld], first 5: %.6f %.6f %.6f %.6f %.6f\n",
                (long long)upsample_tensor->ne[0], (long long)upsample_tensor->ne[1], (long long)upsample_tensor->ne[2],
                data[0], data[1], data[2], data[3], data[4]);
        FILE * f = fopen("debug_upsample_output.bin", "wb");
        if (f) {
            fwrite(data.data(), sizeof(float), data.size(), f);
            fclose(f);
        }
    }
    
    const char * dec_names[] = {"dec0_output", "dec1_after_snake", "dec1_after_conv_t_raw", "dec1_after_trim", "dec1_after_bias", "dec1_output", "dec2_output", "dec3_output", "dec4_output", "dec5_output", "dec6_output"};
    for (int i = 0; i < 11; ++i) {
        struct ggml_tensor * t = ggml_graph_get_tensor(gf, dec_names[i]);
        if (t) {
            std::vector<float> data(ggml_nelements(t));
            ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
            fprintf(stderr, "DEBUG: %s shape [%lld, %lld, %lld], first 5: %.6f %.6f %.6f %.6f %.6f\n",
                    dec_names[i],
                    (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
                    data[0], data[1], data[2], data[3], data[4]);
            char fname[64];
            snprintf(fname, sizeof(fname), "debug_%s.bin", dec_names[i]);
            FILE * f = fopen(fname, "wb");
            if (f) {
                fwrite(data.data(), sizeof(float), data.size(), f);
                fclose(f);
            }
        }
    }
    
    struct ggml_tensor * audio_tensor = ggml_graph_get_tensor(gf, "audio");
    if (!audio_tensor) {
        error_msg_ = "Failed to find audio tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    int64_t n_samples = audio_tensor->ne[0];
    samples.resize(n_samples);
    ggml_backend_tensor_get(audio_tensor, samples.data(), 0, n_samples * sizeof(float));
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

void free_audio_decoder_model(audio_decoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
}

} // namespace qwen3_tts
