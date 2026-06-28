#include "speech_tokenizer_encoder.h"

#include "gguf_loader.h"
#include "qwen3_tts.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <vector>

namespace qwen3_tts {

namespace {

static constexpr int QWEN3_TTS_SPEECH_ENC_MAX_NODES = 4096;

struct conv1d_weights {
    struct ggml_tensor * w = nullptr;
    struct ggml_tensor * b = nullptr;
};

struct speech_tfm_layer {
    struct ggml_tensor * attn_norm_w = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;
    struct ggml_tensor * ffn_norm_w = nullptr;
    struct ggml_tensor * ffn_norm_b = nullptr;
    struct ggml_tensor * attn_q_w = nullptr;
    struct ggml_tensor * attn_k_w = nullptr;
    struct ggml_tensor * attn_v_w = nullptr;
    struct ggml_tensor * attn_o_w = nullptr;
    struct ggml_tensor * attn_scale = nullptr;
    struct ggml_tensor * ffn_up_w = nullptr;
    struct ggml_tensor * ffn_down_w = nullptr;
    struct ggml_tensor * ffn_scale = nullptr;
};

struct speech_tokenizer_encoder_model {
    speech_tokenizer_encoder_config config;
    conv1d_weights conv[15];
    conv1d_weights res[15][4];
    speech_tfm_layer layers[8];
    struct ggml_tensor * downsample_w = nullptr;
    struct ggml_tensor * vq_semantic_input_proj_w = nullptr;
    struct ggml_tensor * vq_acoustic_input_proj_w = nullptr;
    struct ggml_tensor * vq_semantic_codebook = nullptr;
    struct ggml_tensor * vq_acoustic_codebook[31] = {nullptr};
    std::vector<float> vq_semantic_codebook_f32;
    std::vector<std::vector<float>> vq_acoustic_codebook_f32;

    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct speech_tokenizer_encoder_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
    std::vector<int32_t> positions;
    std::vector<float> attention_mask;
};

} // namespace

struct speech_tokenizer_encoder_private {
    speech_tokenizer_encoder_model model;
    speech_tokenizer_encoder_state state;
};

namespace {

bool tensor_to_f32(const struct ggml_tensor * tensor, std::vector<float> & out) {
    if (!tensor || !tensor->data) {
        return false;
    }
    const int64_t n = ggml_nelements(tensor);
    out.resize((size_t) n);
    if (tensor->type == GGML_TYPE_F32) {
        const float * src = (const float *) tensor->data;
        std::copy(src, src + n, out.begin());
        return true;
    }
    if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *) tensor->data;
        for (int64_t i = 0; i < n; ++i) {
            out[(size_t) i] = ggml_fp16_to_fp32(src[i]);
        }
        return true;
    }
    return false;
}

bool debug_enabled() {
    const char * value = std::getenv("QWEN3_TTS_SPEECH_ENCODER_DEBUG");
    return value && value[0] != '\0' && value[0] != '0';
}

const char * dump_dir() {
    const char * value = std::getenv("QWEN3_TTS_SPEECH_ENCODER_DUMP_DIR");
    return value && value[0] != '\0' ? value : nullptr;
}

struct ggml_tensor * add_bias_1d(struct ggml_context * ctx,
                                 struct ggml_tensor * x,
                                 struct ggml_tensor * b) {
    if (!b) {
        return x;
    }
    return ggml_add(ctx, x, ggml_reshape_3d(ctx, b, 1, x->ne[1], 1));
}

struct ggml_tensor * as_f16_conv_weight(struct ggml_context * ctx, struct ggml_tensor * w) {
    if (!w || w->type == GGML_TYPE_F16) {
        return w;
    }
    return ggml_cont(ctx, ggml_cast(ctx, w, GGML_TYPE_F16));
}

struct ggml_tensor * causal_conv1d(struct ggml_context * ctx,
                                   const conv1d_weights & conv,
                                   struct ggml_tensor * x,
                                   int stride,
                                   int pad_total) {
    const int64_t input_len = x->ne[0];
    const int64_t target_len = (input_len + stride - 1) / stride;
    x = ggml_conv_1d(ctx, as_f16_conv_weight(ctx, conv.w), x, stride, pad_total, 1);
    if (x->ne[0] > target_len) {
        x = ggml_view_3d(ctx, x, target_len, x->ne[1], x->ne[2],
                         x->nb[1], x->nb[2], 0);
    }
    x = add_bias_1d(ctx, x, conv.b);
    return ggml_cont(ctx, x);
}

struct ggml_tensor * causal_conv1d_w(struct ggml_context * ctx,
                                     struct ggml_tensor * w,
                                     struct ggml_tensor * x,
                                     int stride,
                                     int pad_total) {
    const int64_t input_len = x->ne[0];
    const int64_t target_len = (input_len + stride - 1) / stride;
    x = ggml_conv_1d(ctx, as_f16_conv_weight(ctx, w), x, stride, pad_total, 1);
    if (x->ne[0] > target_len) {
        x = ggml_view_3d(ctx, x, target_len, x->ne[1], x->ne[2],
                         x->nb[1], x->nb[2], 0);
    }
    return ggml_cont(ctx, x);
}

struct ggml_tensor * replicate_pad_left_1d(struct ggml_context * ctx,
                                           struct ggml_tensor * x,
                                           int left_pad) {
    if (left_pad <= 0) {
        return x;
    }
    struct ggml_tensor * first = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2],
                                             x->nb[1], x->nb[2], 0);
    struct ggml_tensor * prefix_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                           left_pad, x->ne[1], x->ne[2]);
    struct ggml_tensor * prefix = ggml_repeat(ctx, first, prefix_shape);
    return ggml_cont(ctx, ggml_concat(ctx, prefix, x, 0));
}

struct ggml_tensor * conv1d_replicate_left_w(struct ggml_context * ctx,
                                             struct ggml_tensor * w,
                                             struct ggml_tensor * x,
                                             int stride,
                                             int left_pad) {
    x = replicate_pad_left_1d(ctx, x, left_pad);
    x = ggml_conv_1d(ctx, as_f16_conv_weight(ctx, w), x, stride, 0, 1);
    return ggml_cont(ctx, x);
}

struct ggml_tensor * apply_layer_norm(struct ggml_context * ctx,
                                      struct ggml_tensor * x,
                                      struct ggml_tensor * w,
                                      struct ggml_tensor * b,
                                      float eps) {
    x = ggml_norm(ctx, x, eps);
    if (w) {
        x = ggml_mul(ctx, x, w);
    }
    if (b) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}

void maybe_set_debug_output(struct ggml_tensor * tensor, bool enabled) {
    if (enabled && tensor) {
        ggml_set_output(tensor);
    }
}

void dump_tensor_f32(struct ggml_cgraph * gf,
                     const char * tensor_name,
                     const char * out_dir) {
    if (!out_dir) {
        return;
    }
    struct ggml_tensor * tensor = ggml_graph_get_tensor(gf, tensor_name);
    if (!tensor) {
        return;
    }
    const size_t n = (size_t) ggml_nelements(tensor);
    std::vector<float> values(n);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(tensor, tmp.data(), 0, tmp.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) {
            values[i] = ggml_fp16_to_fp32(tmp[i]);
        }
    } else {
        return;
    }

    const std::string base = std::string(out_dir) + "/" + tensor_name;
    std::ofstream data(base + ".f32.bin", std::ios::binary | std::ios::trunc);
    data.write(reinterpret_cast<const char *>(values.data()), (std::streamsize) (values.size() * sizeof(float)));
    std::ofstream shape(base + ".shape.txt", std::ios::out | std::ios::trunc);
    shape << tensor->ne[0] << " " << tensor->ne[1] << " " << tensor->ne[2] << " " << tensor->ne[3] << "\n";
}

struct ggml_tensor * apply_resnet_block(struct ggml_context * ctx,
                                        const speech_tokenizer_encoder_model & model,
                                        struct ggml_tensor * x,
                                        int idx) {
    struct ggml_tensor * residual = x;
    x = ggml_elu(ctx, x);
    x = causal_conv1d(ctx, model.res[idx][1], x, 1, 2);
    x = ggml_elu(ctx, x);
    x = causal_conv1d(ctx, model.res[idx][3], x, 1, 0);
    return ggml_cont(ctx, ggml_add(ctx, residual, x));
}

struct ggml_tensor * apply_transformer_layer(struct ggml_context * ctx,
                                             const speech_tokenizer_encoder_model & model,
                                             struct ggml_tensor * x,
                                             const speech_tfm_layer & layer,
                                             int32_t n_frames,
                                             struct ggml_tensor * positions,
                                             struct ggml_tensor * attention_mask) {
    const auto & cfg = model.config;
    const int n_heads = cfg.n_heads;
    const int head_dim = cfg.head_dim;
    const float scale = 1.0f / sqrtf((float) head_dim);

    struct ggml_tensor * residual = x;
    struct ggml_tensor * normed = apply_layer_norm(ctx, x, layer.attn_norm_w, layer.attn_norm_b, cfg.norm_eps);

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
    KQ = ggml_scale(ctx, KQ, scale);
    if (attention_mask) {
        KQ = ggml_add(ctx, KQ, attention_mask);
    } else {
        KQ = ggml_diag_mask_inf(ctx, KQ, 0);
    }
    KQ = ggml_soft_max(ctx, KQ);

    V = ggml_cont(ctx, ggml_transpose(ctx, V));
    struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);
    KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
    struct ggml_tensor * attn_out = ggml_cont_2d(ctx, KQV, n_heads * head_dim, n_frames);
    attn_out = ggml_mul_mat(ctx, layer.attn_o_w, attn_out);
    if (layer.attn_scale) {
        attn_out = ggml_mul(ctx, attn_out, layer.attn_scale);
    }

    x = ggml_add(ctx, residual, attn_out);
    residual = x;
    normed = apply_layer_norm(ctx, x, layer.ffn_norm_w, layer.ffn_norm_b, cfg.norm_eps);
    struct ggml_tensor * ffn = ggml_mul_mat(ctx, layer.ffn_up_w, normed);
    ffn = ggml_gelu(ctx, ffn);
    ffn = ggml_mul_mat(ctx, layer.ffn_down_w, ffn);
    if (layer.ffn_scale) {
        ffn = ggml_mul(ctx, ffn, layer.ffn_scale);
    }
    return ggml_cont(ctx, ggml_add(ctx, residual, ffn));
}

struct ggml_cgraph * build_project_graph(speech_tokenizer_encoder_private & impl,
                                         int32_t n_samples,
                                         int32_t & n_frames,
                                         int32_t & n_projected_frames,
                                         struct ggml_context ** graph_ctx_out) {
    const auto & model = impl.model;
    const auto & state = impl.state;
    const auto & cfg = model.config;

    struct ggml_init_params params = {
        /*.mem_size   =*/ state.compute_meta.size(),
        /*.mem_buffer =*/ const_cast<uint8_t *>(state.compute_meta.data()),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        return nullptr;
    }
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_SPEECH_ENC_MAX_NODES, false);
    if (!gf) {
        ggml_free(ctx0);
        return nullptr;
    }

    struct ggml_tensor * input = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_samples, 1, 1);
    ggml_set_name(input, "input_values");
    ggml_set_input(input);

    const bool debug = debug_enabled();
    const bool dump_intermediates = dump_dir() != nullptr;

    struct ggml_tensor * cur = causal_conv1d(ctx0, model.conv[0], input, 1, 6);
    ggml_set_name(cur, "encoder_layer_0");
    maybe_set_debug_output(cur, dump_intermediates);
    if (debug) fprintf(stderr, "  speech enc conv0 frames=%lld\n", (long long) cur->ne[0]);

    cur = apply_resnet_block(ctx0, model, cur, 1);
    cur = ggml_elu(ctx0, cur);
    cur = causal_conv1d(ctx0, model.conv[3], cur, 4, 4);
    ggml_set_name(cur, "encoder_layer_3");
    maybe_set_debug_output(cur, dump_intermediates);
    if (debug) fprintf(stderr, "  speech enc conv3 frames=%lld\n", (long long) cur->ne[0]);
    cur = apply_resnet_block(ctx0, model, cur, 4);
    cur = ggml_elu(ctx0, cur);
    cur = causal_conv1d(ctx0, model.conv[6], cur, 5, 5);
    ggml_set_name(cur, "encoder_layer_6");
    maybe_set_debug_output(cur, dump_intermediates);
    if (debug) fprintf(stderr, "  speech enc conv6 frames=%lld\n", (long long) cur->ne[0]);
    cur = apply_resnet_block(ctx0, model, cur, 7);
    cur = ggml_elu(ctx0, cur);
    cur = causal_conv1d(ctx0, model.conv[9], cur, 6, 6);
    ggml_set_name(cur, "encoder_layer_9");
    maybe_set_debug_output(cur, dump_intermediates);
    if (debug) fprintf(stderr, "  speech enc conv9 frames=%lld\n", (long long) cur->ne[0]);
    cur = apply_resnet_block(ctx0, model, cur, 10);
    cur = ggml_elu(ctx0, cur);
    cur = causal_conv1d(ctx0, model.conv[12], cur, 8, 8);
    ggml_set_name(cur, "encoder_layer_12");
    maybe_set_debug_output(cur, dump_intermediates);
    if (debug) fprintf(stderr, "  speech enc conv12 frames=%lld\n", (long long) cur->ne[0]);
    cur = ggml_elu(ctx0, cur);
    cur = causal_conv1d(ctx0, model.conv[14], cur, 1, 2);
    ggml_set_name(cur, "encoder_layer_14");
    maybe_set_debug_output(cur, dump_intermediates);
    if (debug) fprintf(stderr, "  speech enc conv14 frames=%lld\n", (long long) cur->ne[0]);
    n_frames = (int32_t) cur->ne[0];

    struct ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, n_frames, cfg.hidden_size);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur_2d));

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    struct ggml_tensor * attention_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_frames, n_frames, 1);
    ggml_set_name(attention_mask, "attention_mask");
    ggml_set_input(attention_mask);

    for (int i = 0; i < cfg.n_layers; ++i) {
        cur = apply_transformer_layer(ctx0, model, cur, model.layers[i], n_frames, positions, attention_mask);
        if (i == 0) {
            ggml_set_name(cur, "transformer_layer_0");
            maybe_set_debug_output(cur, dump_intermediates);
        } else if (i == cfg.n_layers - 1) {
            ggml_set_name(cur, "transformer_layer_7");
            maybe_set_debug_output(cur, dump_intermediates);
        }
    }

    struct ggml_tensor * cur_t = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = ggml_reshape_3d(ctx0, cur_t, n_frames, cfg.hidden_size, 1);
    cur = conv1d_replicate_left_w(ctx0, model.downsample_w, cur, 2, 2);
    ggml_set_name(cur, "downsample_output");
    maybe_set_debug_output(cur, dump_intermediates);
    n_projected_frames = (int32_t) cur->ne[0];

    struct ggml_tensor * semantic = causal_conv1d_w(ctx0, model.vq_semantic_input_proj_w, cur, 1, 0);
    ggml_set_name(semantic, "semantic_projected");
    ggml_set_output(semantic);
    struct ggml_tensor * acoustic = causal_conv1d_w(ctx0, model.vq_acoustic_input_proj_w, cur, 1, 0);
    ggml_set_name(acoustic, "acoustic_projected");
    ggml_set_output(acoustic);

    ggml_build_forward_expand(gf, semantic);
    ggml_build_forward_expand(gf, acoustic);

    if (graph_ctx_out) {
        *graph_ctx_out = ctx0;
    } else {
        ggml_free(ctx0);
    }
    return gf;
}

} // namespace

SpeechTokenizerEncoder::SpeechTokenizerEncoder()
    : impl_(std::make_unique<speech_tokenizer_encoder_private>()) {}

SpeechTokenizerEncoder::~SpeechTokenizerEncoder() {
    unload_model();
}

const speech_tokenizer_encoder_config & SpeechTokenizerEncoder::get_config() const {
    return impl_->model.config;
}

const std::string & SpeechTokenizerEncoder::get_error() const {
    return error_msg_;
}

void SpeechTokenizerEncoder::unload_model() {
    auto & model = impl_->model;
    auto & state = impl_->state;
    if (state.sched) {
        ggml_backend_sched_free(state.sched);
        state.sched = nullptr;
    }
    if (state.backend) {
        release_preferred_backend(state.backend);
        state.backend = nullptr;
    }
    if (state.backend_cpu) {
        ggml_backend_free(state.backend_cpu);
        state.backend_cpu = nullptr;
    }
    state.compute_meta.clear();
    state.positions.clear();
    state.attention_mask.clear();
    free_ggml_resources(model.ctx, model.buffer);
    model = speech_tokenizer_encoder_model{};
    error_msg_.clear();
}

bool SpeechTokenizerEncoder::load_model(const std::string & tokenizer_model_path) {
    unload_model();

    GGUFLoader loader;
    if (!loader.open(tokenizer_model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }

    auto & model = impl_->model;
    auto & cfg = model.config;
    cfg.sample_rate = loader.get_u32(
        "qwen3-tts-tokenizer.sample_rate",
        loader.get_u32("qwen3-tts-tokenizer.input_sample_rate",
                       loader.get_u32("qwen3-tts.tokenizer.sample_rate", 24000)));
    cfg.frame_rate = loader.get_f32(
        "qwen3-tts-tokenizer.frame_rate",
        loader.get_f32("qwen3-tts.tokenizer.frame_rate", 12.5f));
    cfg.hidden_size = loader.get_u32(
        "qwen3-tts-tokenizer.encoder.hidden_size",
        loader.get_u32("qwen3-tts.tokenizer.encoder.hidden_size", 512));
    cfg.n_layers = loader.get_u32(
        "qwen3-tts-tokenizer.encoder.num_hidden_layers",
        loader.get_u32("qwen3-tts-tokenizer.encoder.num_layers",
                       loader.get_u32("qwen3-tts.tokenizer.encoder.num_layers", 8)));
    cfg.n_heads = loader.get_u32(
        "qwen3-tts-tokenizer.encoder.num_attention_heads",
        loader.get_u32("qwen3-tts-tokenizer.encoder.num_heads",
                       loader.get_u32("qwen3-tts.tokenizer.encoder.num_heads", 8)));
    cfg.head_dim = cfg.hidden_size / cfg.n_heads;
    cfg.n_quantizers = loader.get_u32(
        "qwen3-tts-tokenizer.encoder.num_quantizers",
        loader.get_u32("qwen3-tts.tokenizer.encoder.num_quantizers", 32));
    cfg.n_valid_quantizers = loader.get_u32(
        "qwen3-tts-tokenizer.encoder_valid_num_quantizers",
        loader.get_u32("qwen3-tts-tokenizer.encoder.valid_quantizers",
                       loader.get_u32("qwen3-tts.tokenizer.encoder.valid_quantizers", 16)));
    cfg.codebook_dim = loader.get_u32(
        "qwen3-tts-tokenizer.encoder.codebook_dim",
        loader.get_u32("qwen3-tts.tokenizer.encoder.codebook_dim", 256));
    cfg.codebook_size = loader.get_u32(
        "qwen3-tts-tokenizer.codebook_size",
        loader.get_u32("qwen3-tts-tokenizer.encoder.codebook_size",
                       loader.get_u32("qwen3-tts.tokenizer.codebook_size", 2048)));

    int tok_enc_tensor_count = 0;
    const int64_t n_tensors = loader.get_n_tensors();
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "tok_enc.", 8) == 0) {
            tok_enc_tensor_count++;
        }
    }
    if (tok_enc_tensor_count == 0) {
        error_msg_ = "No speech tokenizer encoder tensors found in tokenizer GGUF";
        return false;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * tok_enc_tensor_count,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    model.ctx = ggml_init(params);
    if (!model.ctx) {
        error_msg_ = "Failed to initialize speech tokenizer encoder GGML context";
        return false;
    }

    struct ggml_context * meta_ctx = loader.get_meta_ctx();
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "tok_enc.", 8) != 0) {
            continue;
        }

        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta_ctx, name);
        if (!meta_tensor) {
            continue;
        }
        struct ggml_tensor * tensor = ggml_dup_tensor(model.ctx, meta_tensor);
        ggml_set_name(tensor, name);
        model.tensors[name] = tensor;

        const std::string sname(name);
        int idx = -1;
        int sub = -1;
        char suffix[64] = {0};
        if (sscanf(name, "tok_enc.conv.%d.%63s", &idx, suffix) == 2 && idx >= 0 && idx < 15) {
            if (strcmp(suffix, "weight") == 0) model.conv[idx].w = tensor;
            else if (strcmp(suffix, "bias") == 0) model.conv[idx].b = tensor;
        } else if (sscanf(name, "tok_enc.res.%d.blk.%d.%63s", &idx, &sub, suffix) == 3 &&
                   idx >= 0 && idx < 15 && sub >= 0 && sub < 4) {
            if (strcmp(suffix, "weight") == 0) model.res[idx][sub].w = tensor;
            else if (strcmp(suffix, "bias") == 0) model.res[idx][sub].b = tensor;
        } else if (sname == "tok_enc.downsample.weight") {
            model.downsample_w = tensor;
        } else if (sname == "tok_enc.vq_semantic.input_proj.weight") {
            model.vq_semantic_input_proj_w = tensor;
        } else if (sname == "tok_enc.vq_acoustic.input_proj.weight") {
            model.vq_acoustic_input_proj_w = tensor;
        } else if (sname == "tok_enc.vq_semantic.0.codebook") {
            model.vq_semantic_codebook = tensor;
        } else if (sname.rfind("tok_enc.vq_acoustic.", 0) == 0 &&
                   sname.size() > strlen("tok_enc.vq_acoustic.") + strlen(".codebook") &&
                   sname.compare(sname.size() - strlen(".codebook"), strlen(".codebook"), ".codebook") == 0) {
            const size_t prefix_len = strlen("tok_enc.vq_acoustic.");
            const std::string idx_text = sname.substr(prefix_len, sname.size() - prefix_len - strlen(".codebook"));
            idx = std::stoi(idx_text);
            if (idx >= 0 && idx < 31) {
                model.vq_acoustic_codebook[idx] = tensor;
            }
        } else if (sscanf(name, "tok_enc.blk.%d.%63s", &idx, suffix) == 2 &&
                   idx >= 0 && idx < 8) {
            auto & layer = model.layers[idx];
            if (strcmp(suffix, "attn_norm.weight") == 0) layer.attn_norm_w = tensor;
            else if (strcmp(suffix, "attn_norm.bias") == 0) layer.attn_norm_b = tensor;
            else if (strcmp(suffix, "ffn_norm.weight") == 0) layer.ffn_norm_w = tensor;
            else if (strcmp(suffix, "ffn_norm.bias") == 0) layer.ffn_norm_b = tensor;
            else if (strcmp(suffix, "attn_q.weight") == 0) layer.attn_q_w = tensor;
            else if (strcmp(suffix, "attn_k.weight") == 0) layer.attn_k_w = tensor;
            else if (strcmp(suffix, "attn_v.weight") == 0) layer.attn_v_w = tensor;
            else if (strcmp(suffix, "attn_output.weight") == 0) layer.attn_o_w = tensor;
            else if (strcmp(suffix, "attn_scale") == 0) layer.attn_scale = tensor;
            else if (strcmp(suffix, "ffn_up.weight") == 0) layer.ffn_up_w = tensor;
            else if (strcmp(suffix, "ffn_down.weight") == 0) layer.ffn_down_w = tensor;
            else if (strcmp(suffix, "ffn_scale") == 0) layer.ffn_scale = tensor;
        }
    }

    if (!model.downsample_w || !model.vq_semantic_input_proj_w ||
        !model.vq_acoustic_input_proj_w || !model.vq_semantic_codebook) {
        error_msg_ = "Speech tokenizer encoder GGUF is missing required tok_enc tensors";
        return false;
    }

    if (!load_tensor_data_from_file(tokenizer_model_path, loader.get_ctx(), model.ctx,
                                    model.tensors, model.buffer, error_msg_,
                                    get_preferred_backend_type())) {
        return false;
    }

    if (!tensor_to_f32(model.vq_semantic_codebook, model.vq_semantic_codebook_f32)) {
        error_msg_ = "Failed to materialize semantic codebook";
        return false;
    }
    if (cfg.n_valid_quantizers < 1 || cfg.n_valid_quantizers > 16) {
        error_msg_ = "Unsupported speech tokenizer valid quantizer count";
        return false;
    }
    model.vq_acoustic_codebook_f32.assign(31, {});
    const int32_t acoustic_codebooks_needed = cfg.n_valid_quantizers - 1;
    for (int i = 0; i < acoustic_codebooks_needed; ++i) {
        if (!model.vq_acoustic_codebook[i]) {
            error_msg_ = "Speech tokenizer encoder GGUF is missing acoustic VQ codebook " + std::to_string(i);
            return false;
        }
        if (!tensor_to_f32(model.vq_acoustic_codebook[i], model.vq_acoustic_codebook_f32[i])) {
            error_msg_ = "Failed to materialize acoustic VQ codebook " + std::to_string(i);
            return false;
        }
    }

    fprintf(stderr, "  Speech tokenizer encoder loaded: tensors=%d, hidden=%d, quantizers=%d valid=%d\n",
            tok_enc_tensor_count, cfg.hidden_size, cfg.n_quantizers, cfg.n_valid_quantizers);

    auto & state = impl_->state;
    state.backend = init_preferred_backend("SpeechTokenizerEncoder", &error_msg_);
    if (!state.backend) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(state.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  SpeechTokenizerEncoder backend: %s\n", device_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for SpeechTokenizerEncoder";
            return false;
        }
    }
    std::vector<ggml_backend_t> backends;
    backends.push_back(state.backend);
    if (state.backend_cpu) {
        backends.push_back(state.backend_cpu);
    }
    state.sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(),
                                         QWEN3_TTS_SPEECH_ENC_MAX_NODES, false, true);
    if (!state.sched) {
        error_msg_ = "Failed to create SpeechTokenizerEncoder backend scheduler";
        return false;
    }
    state.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_SPEECH_ENC_MAX_NODES +
                              ggml_graph_overhead());
    return true;
}

bool SpeechTokenizerEncoder::encode(const float * samples, int32_t n_samples, speech_codes & codes) {
    std::vector<float> semantic_features;
    std::vector<float> acoustic_features;
    int32_t n_frames = 0;
    if (!project(samples, n_samples, semantic_features, acoustic_features, n_frames)) {
        return false;
    }
    return quantize_projected(semantic_features.data(), acoustic_features.data(), n_frames, codes);
}

bool SpeechTokenizerEncoder::project(const float * samples,
                                     int32_t n_samples,
                                     std::vector<float> & semantic_features,
                                     std::vector<float> & acoustic_features,
                                     int32_t & n_frames) {
    auto & model = impl_->model;
    auto & state = impl_->state;
    const auto & cfg = model.config;
    if (!model.ctx || !state.sched) {
        error_msg_ = "Speech tokenizer encoder model is not loaded";
        return false;
    }
    if (!samples || n_samples <= 0) {
        error_msg_ = "Invalid speech tokenizer encoder input samples";
        return false;
    }

    int32_t n_encoder_frames = 0;
    int32_t n_projected_frames = 0;
    struct ggml_context * graph_ctx = nullptr;
    struct ggml_cgraph * gf = build_project_graph(*impl_, n_samples, n_encoder_frames,
                                                  n_projected_frames, &graph_ctx);
    if (!gf || !graph_ctx) {
        error_msg_ = "Failed to build speech tokenizer encoder graph";
        if (graph_ctx) ggml_free(graph_ctx);
        return false;
    }
    if (debug_enabled()) {
        fprintf(stderr, "  SpeechTokenizerEncoder graph frames: encoder=%d projected=%d\n",
                n_encoder_frames, n_projected_frames);
    }

    if (!ggml_backend_sched_alloc_graph(state.sched, gf)) {
        error_msg_ = "Failed to allocate speech tokenizer encoder graph";
        ggml_free(graph_ctx);
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    struct ggml_tensor * input = ggml_graph_get_tensor(gf, "input_values");
    struct ggml_tensor * positions = ggml_graph_get_tensor(gf, "positions");
    struct ggml_tensor * attention_mask = ggml_graph_get_tensor(gf, "attention_mask");
    struct ggml_tensor * semantic = ggml_graph_get_tensor(gf, "semantic_projected");
    struct ggml_tensor * acoustic = ggml_graph_get_tensor(gf, "acoustic_projected");
    if (!input || !positions || !attention_mask || !semantic || !acoustic) {
        error_msg_ = "Speech tokenizer encoder graph is missing required tensors";
        ggml_free(graph_ctx);
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    ggml_backend_tensor_set(input, samples, 0, (size_t) n_samples * sizeof(float));

    state.positions.resize((size_t) n_encoder_frames);
    for (int32_t i = 0; i < n_encoder_frames; ++i) {
        state.positions[(size_t) i] = i;
    }
    ggml_backend_tensor_set(positions, state.positions.data(), 0,
                            state.positions.size() * sizeof(int32_t));

    state.attention_mask.assign((size_t) n_encoder_frames * (size_t) n_encoder_frames, 0.0f);
    for (int32_t q = 0; q < n_encoder_frames; ++q) {
        for (int32_t k = 0; k < n_encoder_frames; ++k) {
            if (k > q) {
                state.attention_mask[(size_t) k + (size_t) q * (size_t) n_encoder_frames] = -1.0e30f;
            }
        }
    }
    ggml_backend_tensor_set(attention_mask, state.attention_mask.data(), 0,
                            state.attention_mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(state.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute speech tokenizer encoder graph";
        ggml_free(graph_ctx);
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    const char * tensor_dump_dir = dump_dir();
    if (tensor_dump_dir) {
        dump_tensor_f32(gf, "encoder_layer_0", tensor_dump_dir);
        dump_tensor_f32(gf, "encoder_layer_3", tensor_dump_dir);
        dump_tensor_f32(gf, "encoder_layer_6", tensor_dump_dir);
        dump_tensor_f32(gf, "encoder_layer_9", tensor_dump_dir);
        dump_tensor_f32(gf, "encoder_layer_12", tensor_dump_dir);
        dump_tensor_f32(gf, "encoder_layer_14", tensor_dump_dir);
        dump_tensor_f32(gf, "transformer_layer_0", tensor_dump_dir);
        dump_tensor_f32(gf, "transformer_layer_7", tensor_dump_dir);
        dump_tensor_f32(gf, "downsample_output", tensor_dump_dir);
        dump_tensor_f32(gf, "semantic_projected", tensor_dump_dir);
        dump_tensor_f32(gf, "acoustic_projected", tensor_dump_dir);
    }

    semantic_features.resize((size_t) n_projected_frames * (size_t) cfg.codebook_dim);
    acoustic_features.resize((size_t) n_projected_frames * (size_t) cfg.codebook_dim);
    ggml_backend_tensor_get(semantic, semantic_features.data(), 0,
                            semantic_features.size() * sizeof(float));
    ggml_backend_tensor_get(acoustic, acoustic_features.data(), 0,
                            acoustic_features.size() * sizeof(float));

    ggml_free(graph_ctx);
    ggml_backend_sched_reset(state.sched);

    n_frames = n_projected_frames;
    return true;
}

bool SpeechTokenizerEncoder::quantize_projected(const float * semantic_features,
                                                const float * acoustic_features,
                                                int32_t n_frames,
                                                speech_codes & codes) {
    auto & model = impl_->model;
    const auto & cfg = model.config;
    if (!model.ctx || model.vq_semantic_codebook_f32.empty() ||
        model.vq_acoustic_codebook_f32.empty()) {
        error_msg_ = "Speech tokenizer encoder model is not loaded";
        return false;
    }
    if (!semantic_features || !acoustic_features || n_frames <= 0) {
        error_msg_ = "Invalid projected speech tokenizer features";
        return false;
    }

    const int32_t codebook_dim = cfg.codebook_dim;
    const int32_t codebook_size = cfg.codebook_size;
    const int32_t valid_quantizers = cfg.n_valid_quantizers;
    const size_t expected_codebook_values = (size_t) codebook_dim * (size_t) codebook_size;
    if (model.vq_semantic_codebook_f32.size() != expected_codebook_values) {
        error_msg_ = "Semantic VQ codebook has unexpected size";
        return false;
    }
    for (int32_t q = 0; q < valid_quantizers - 1; ++q) {
        if (q >= (int32_t) model.vq_acoustic_codebook_f32.size() ||
            model.vq_acoustic_codebook_f32[(size_t) q].size() != expected_codebook_values) {
            const size_t actual = q < (int32_t) model.vq_acoustic_codebook_f32.size()
                ? model.vq_acoustic_codebook_f32[(size_t) q].size()
                : 0;
            error_msg_ = "Acoustic VQ codebook has unexpected size at quantizer " +
                         std::to_string(q) + ": " + std::to_string(actual);
            return false;
        }
    }
    if (valid_quantizers < 1 || valid_quantizers > 16) {
        error_msg_ = "Unsupported speech tokenizer valid quantizer count";
        return false;
    }

    auto find_nearest = [&](const std::vector<float> & codebook,
                            const std::vector<float> & residual) -> int32_t {
        int32_t best_idx = 0;
        float best_dist = std::numeric_limits<float>::infinity();
        for (int32_t code = 0; code < codebook_size; ++code) {
            const float * cb = codebook.data() + (size_t) code * (size_t) codebook_dim;
            float dist = 0.0f;
            for (int32_t d = 0; d < codebook_dim; ++d) {
                const float diff = residual[(size_t) d] - cb[d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = code;
            }
        }
        return best_idx;
    };

    codes.n_frames = n_frames;
    codes.n_codebooks = valid_quantizers;
    codes.codes.assign((size_t) n_frames * (size_t) valid_quantizers, 0);

    std::vector<float> residual((size_t) codebook_dim);
    for (int32_t frame = 0; frame < n_frames; ++frame) {
        for (int32_t d = 0; d < codebook_dim; ++d) {
            residual[(size_t) d] = semantic_features[(size_t) d * (size_t) n_frames + (size_t) frame];
        }
        const int32_t semantic_code = find_nearest(model.vq_semantic_codebook_f32, residual);
        codes.codes[(size_t) frame * (size_t) valid_quantizers] = semantic_code;

        for (int32_t d = 0; d < codebook_dim; ++d) {
            residual[(size_t) d] = acoustic_features[(size_t) d * (size_t) n_frames + (size_t) frame];
        }
        for (int32_t q = 0; q < valid_quantizers - 1; ++q) {
            const int32_t acoustic_code = find_nearest(model.vq_acoustic_codebook_f32[(size_t) q], residual);
            codes.codes[(size_t) frame * (size_t) valid_quantizers + (size_t) q + 1] = acoustic_code;

            const float * cb = model.vq_acoustic_codebook_f32[(size_t) q].data() +
                               (size_t) acoustic_code * (size_t) codebook_dim;
            for (int32_t d = 0; d < codebook_dim; ++d) {
                residual[(size_t) d] -= cb[d];
            }
        }
    }

    return true;
}

} // namespace qwen3_tts
