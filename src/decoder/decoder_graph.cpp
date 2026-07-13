#include "audio_tokenizer_decoder.h"
#include "decoder/decoder_state_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace qwen3_tts {

namespace {

struct ggml_tensor * as_f16_conv_weight(struct ggml_context * ctx, struct ggml_tensor * w) {
    if (!w || w->type == GGML_TYPE_F16) {
        return w;
    }
    return ggml_cont(ctx, ggml_cast(ctx, w, GGML_TYPE_F16));
}

enum class rest_codebook_projection_mode {
    automatic,
    legacy,
    summed,
};

rest_codebook_projection_mode get_rest_codebook_projection_mode() {
    static const rest_codebook_projection_mode mode = []() {
#if defined(_MSC_VER)
        char * value = nullptr;
        size_t value_len = 0;
        if (_dupenv_s(&value, &value_len, "QWEN3_TTS_DECODER_SUM_REST_EMBEDDINGS") != 0 || !value) {
            return rest_codebook_projection_mode::automatic;
        }
        const char first = value[0];
        std::free(value);
#else
        const char * value = std::getenv("QWEN3_TTS_DECODER_SUM_REST_EMBEDDINGS");
        const char first = value ? value[0] : '\0';
#endif
        if (first == '0') {
            return rest_codebook_projection_mode::legacy;
        }
        if (first == '1') {
            return rest_codebook_projection_mode::summed;
        }
        return rest_codebook_projection_mode::automatic;
    }();
    return mode;
}

bool is_cuda_backend(ggml_backend_t backend) {
    ggml_backend_dev_t device = backend ? ggml_backend_get_device(backend) : nullptr;
    const char * name = device ? ggml_backend_dev_name(device) : nullptr;
    return name && std::strstr(name, "CUDA") != nullptr;
}

bool is_cpu_backend(ggml_backend_t backend) {
    ggml_backend_dev_t device = backend ? ggml_backend_get_device(backend) : nullptr;
    return device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

bool summed_rest_codebook_projection_enabled(ggml_backend_t backend, int32_t n_frames) {
    const rest_codebook_projection_mode mode = get_rest_codebook_projection_mode();
    if (mode == rest_codebook_projection_mode::legacy) {
        return false;
    }
    if (mode == rest_codebook_projection_mode::summed) {
        return true;
    }

    // CPU benefits from removing fourteen repeated projections across the
    // tested decoder lengths. CUDA retains the summed path only for short
    // inputs, where its launch reduction offsets the changed GEMM shape.
    return is_cpu_backend(backend) || (is_cuda_backend(backend) && n_frames < 64);
}

} // namespace

struct ggml_cgraph * decoder_internal::ops::build_graph(AudioTokenizerDecoder & self, int32_t n_frames) {
    return build_graph_impl(self, n_frames, nullptr);
}

struct ggml_cgraph * decoder_internal::ops::build_graph_impl(AudioTokenizerDecoder & self,
                                                             int32_t n_frames,
                                                             struct ggml_context ** graph_ctx_out) {
    auto & model = self.impl_->model;
    auto & state = self.impl_->state;
    const auto & cfg = model.config;

    struct ggml_init_params params = {
        /*.mem_size   =*/ state.compute_meta.size(),
        /*.mem_buffer =*/ state.compute_meta.data(),
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

    struct ggml_tensor * first_emb = ggml_get_rows(ctx0, model.vq_first_codebook, first_codes);
    ggml_set_name(first_emb, "first_emb_raw");

    struct ggml_tensor * rest_emb[15];
    for (int cb = 0; cb < 15; ++cb) {
        struct ggml_tensor * cb_codes = cb_codes_tensors[cb + 1];
        rest_emb[cb] = ggml_get_rows(ctx0, model.vq_rest_codebook[cb], cb_codes);

        if (cb == 0) {
            ggml_set_name(rest_emb[cb], "rest_cb0_emb_raw");
        }
    }

    struct ggml_tensor * first_emb_2d = ggml_reshape_2d(ctx0, first_emb, cfg.codebook_dim, n_frames);
    ggml_set_name(first_emb_2d, "first_emb_2d");

    struct ggml_tensor * first_proj_weight_2d = ggml_reshape_2d(ctx0, model.vq_first_output_proj,
                                                                cfg.codebook_dim, cfg.hidden_dim);
    struct ggml_tensor * first_proj_2d = ggml_mul_mat(ctx0, first_proj_weight_2d, first_emb_2d);
    ggml_set_name(first_proj_2d, "first_proj_2d");

    struct ggml_tensor * rest_proj_weight_2d = ggml_reshape_2d(ctx0, model.vq_rest_output_proj,
                                                               cfg.codebook_dim, cfg.hidden_dim);

    struct ggml_tensor * rest_emb_2d[15];
    for (int cb = 0; cb < 15; ++cb) {
        rest_emb_2d[cb] = ggml_reshape_2d(ctx0, rest_emb[cb], cfg.codebook_dim, n_frames);

        if (cb == 0) {
            ggml_set_name(rest_emb_2d[cb], "rest_cb0_emb_2d");
        }
    }

    struct ggml_tensor * rest_proj_2d = nullptr;
    if (summed_rest_codebook_projection_enabled(state.backend, n_frames)) {
        struct ggml_tensor * rest_emb_sum_2d = rest_emb_2d[0];
        for (int cb = 1; cb < 15; ++cb) {
            rest_emb_sum_2d = ggml_add(ctx0, rest_emb_sum_2d, rest_emb_2d[cb]);
        }
        ggml_set_name(rest_emb_sum_2d, "rest_emb_sum_2d");
        rest_proj_2d = ggml_mul_mat(ctx0, rest_proj_weight_2d, rest_emb_sum_2d);
    } else {
        for (int cb = 0; cb < 15; ++cb) {
            struct ggml_tensor * cb_proj_2d = ggml_mul_mat(ctx0, rest_proj_weight_2d, rest_emb_2d[cb]);

            if (rest_proj_2d == nullptr) {
                rest_proj_2d = cb_proj_2d;
            } else {
                rest_proj_2d = ggml_add(ctx0, rest_proj_2d, cb_proj_2d);
            }
        }
    }
    ggml_set_name(rest_proj_2d, "rest_proj_2d");

    struct ggml_tensor * latent_2d = ggml_add(ctx0, first_proj_2d, rest_proj_2d);
    ggml_set_name(latent_2d, "latent_2d");

    struct ggml_tensor * latent_t = ggml_transpose(ctx0, latent_2d);
    ggml_set_name(latent_t, "latent_t");

    struct ggml_tensor * latent_cont = ggml_cont(ctx0, latent_t);
    ggml_set_name(latent_cont, "latent_cont");

    struct ggml_tensor * latent = ggml_reshape_3d(ctx0, latent_cont, n_frames, cfg.hidden_dim, 1);
    ggml_set_name(latent, "vq_output");

    struct ggml_tensor * latent_for_conv = ggml_cont(ctx0, latent);
    struct ggml_tensor * latent_padded = ggml_pad_ext(ctx0, latent_for_conv, 2, 0, 0, 0, 0, 0, 0, 0);
    struct ggml_tensor * cur = ggml_conv_1d(ctx0, as_f16_conv_weight(ctx0, model.pre_conv_w), latent_padded, 1, 0, 1);
    if (model.pre_conv_b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.pre_conv_b, 1, cfg.latent_dim, 1));
    }

    ggml_set_name(cur, "pre_conv_output");

    struct ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, n_frames, cfg.latent_dim);
    struct ggml_tensor * cur_t = ggml_transpose(ctx0, cur_2d);
    cur = ggml_cont(ctx0, cur_t);

    ggml_set_name(cur, "pre_conv_reshaped");

    cur = ggml_mul_mat(ctx0, model.pre_tfm_input_proj_w, cur);
    if (model.pre_tfm_input_proj_b) {
        cur = ggml_add(ctx0, cur, model.pre_tfm_input_proj_b);
    }

    ggml_set_name(cur, "pre_tfm_input");

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, n_frames);
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    for (int i = 0; i < cfg.n_pre_tfm_layers; ++i) {
        cur = apply_pre_tfm_layer(ctx0, self, cur, model.pre_tfm_layers[i], n_frames, positions, mask);
    }

    if (model.pre_tfm_norm_w) {
        cur = apply_rms_norm(ctx0, cur, model.pre_tfm_norm_w, cfg.rms_norm_eps);
    }

    cur = ggml_mul_mat(ctx0, model.pre_tfm_output_proj_w, cur);
    if (model.pre_tfm_output_proj_b) {
        cur = ggml_add(ctx0, cur, model.pre_tfm_output_proj_b);
    }

    ggml_set_name(cur, "pre_tfm_output");

    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_3d(ctx0, cur, n_frames, cfg.latent_dim, 1);

    ggml_set_name(cur, "pre_tfm_reshaped");

    for (int i = 0; i < 2; ++i) {
        cur = apply_upsample_block(ctx0, cur, model.upsample[i], i);
    }

    ggml_set_name(cur, "upsample_output");

    cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, as_f16_conv_weight(ctx0, model.dec0_conv_w), cur, 1, 0, 1);
    if (model.dec0_conv_b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.dec0_conv_b, 1, cfg.decoder_dim, 1));
    }

    ggml_set_name(cur, "dec0_output");

    int upsample_rates[4] = {8, 5, 4, 3};
    for (int i = 0; i < 4; ++i) {
        cur = apply_decoder_block(ctx0, self, cur, model.dec_blocks[i], upsample_rates[i], i);
        char name[32];
        snprintf(name, sizeof(name), "dec%d_output", i + 1);
        ggml_set_name(cur, name);
    }

    if (model.dec5_snake_alpha) {
        cur = apply_snake(ctx0, cur, model.dec5_snake_alpha, model.dec5_snake_beta,
                          model.dec5_snake_alpha_exp, model.dec5_snake_inv_beta_exp);
    }

    ggml_set_name(cur, "dec5_output");

    cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, as_f16_conv_weight(ctx0, model.dec6_conv_w), cur, 1, 0, 1);
    if (model.dec6_conv_b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.dec6_conv_b, 1, 1, 1));
    }

    ggml_set_name(cur, "dec6_output");

    cur = ggml_tanh(ctx0, cur);
    cur = ggml_reshape_1d(ctx0, cur, cur->ne[0]);

    ggml_set_name(cur, "audio");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);

    if (graph_ctx_out) {
        *graph_ctx_out = ctx0;
    } else {
        ggml_free(ctx0);
    }

    return gf;
}

bool decoder_internal::ops::ensure_stream_graph(AudioTokenizerDecoder & self, int32_t n_frames) {
    auto & impl = self.impl_;
    auto & state = impl->state;
    auto & model = impl->model;
    auto & error_msg = impl->error_msg;
    const auto & cfg = model.config;
    if (n_frames <= 0 || n_frames > 32) {
        error_msg = "Streaming decoder chunk width must be between 1 and 32";
        return false;
    }
    auto existing = state.stream_graphs.find(n_frames);
    if (existing != state.stream_graphs.end() && existing->second.gf) {
        return true;
    }
    if (!ensure_stream_state(self)) {
        return false;
    }

    audio_decoder_state::decoder_stream_graph & graph = state.stream_graphs[n_frames];
    struct ggml_init_params params = {
        ggml_tensor_overhead() * QWEN3_TTS_DEC_MAX_NODES +
            ggml_graph_overhead_custom(QWEN3_TTS_DEC_MAX_NODES, false),
        nullptr,
        true,
    };
    graph.ctx = ggml_init(params);
    if (!graph.ctx) {
        error_msg = "Failed to initialize streaming decoder graph";
        state.stream_graphs.erase(n_frames);
        return false;
    }
    struct ggml_context * ctx0 = graph.ctx;
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_DEC_MAX_NODES, false);
    struct ggml_tensor * codes = ggml_new_tensor_2d(
        ctx0, GGML_TYPE_I32, n_frames, cfg.n_codebooks);
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    struct ggml_tensor * rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_frames);
    struct ggml_tensor * mask = ggml_new_tensor_2d(
        ctx0, GGML_TYPE_F32, state.stream_ring, n_frames);
    ggml_set_name(codes, "stream_codes");
    ggml_set_name(positions, "stream_positions");
    ggml_set_name(rows, "stream_rows");
    ggml_set_name(mask, "stream_mask");
    ggml_set_input(codes);
    ggml_set_input(positions);
    ggml_set_input(rows);
    ggml_set_input(mask);

    auto codebook_indices = [&](int cb) {
        return ggml_cont(ctx0, ggml_view_1d(
            ctx0, codes, n_frames, (size_t) cb * codes->nb[1]));
    };
    struct ggml_tensor * first_emb = ggml_get_rows(
        ctx0, model.vq_first_codebook, codebook_indices(0));
    struct ggml_tensor * first_weight = ggml_reshape_2d(
        ctx0, model.vq_first_output_proj, cfg.codebook_dim, cfg.hidden_dim);
    struct ggml_tensor * first_proj = ggml_mul_mat(ctx0, first_weight, first_emb);
    struct ggml_tensor * rest_weight = ggml_reshape_2d(
        ctx0, model.vq_rest_output_proj, cfg.codebook_dim, cfg.hidden_dim);
    struct ggml_tensor * rest_sum = nullptr;
    for (int cb = 0; cb < 15; ++cb) {
        struct ggml_tensor * emb = ggml_get_rows(
            ctx0, model.vq_rest_codebook[cb], codebook_indices(cb + 1));
        rest_sum = rest_sum ? ggml_add(ctx0, rest_sum, emb) : emb;
    }
    struct ggml_tensor * cur = ggml_add(
        ctx0, first_proj, ggml_mul_mat(ctx0, rest_weight, rest_sum));
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [T, hidden_dim]
    cur = apply_causal_conv_stream(ctx0, gf, cur, model.pre_conv_w, model.pre_conv_b,
                                   3, 1, state.stream_pre_conv);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [latent_dim, T]
    cur = ggml_mul_mat(ctx0, model.pre_tfm_input_proj_w, cur);
    if (model.pre_tfm_input_proj_b) {
        cur = ggml_add(ctx0, cur, model.pre_tfm_input_proj_b);
    }
    for (int i = 0; i < cfg.n_pre_tfm_layers; ++i) {
        cur = apply_pre_tfm_layer_stream(
            ctx0, gf, self, cur, model.pre_tfm_layers[i], n_frames,
            positions, rows, mask, state.stream_k[(size_t) i],
            state.stream_v[(size_t) i], state.stream_ring);
    }
    if (model.pre_tfm_norm_w) {
        cur = apply_rms_norm(ctx0, cur, model.pre_tfm_norm_w, cfg.rms_norm_eps);
    }
    cur = ggml_mul_mat(ctx0, model.pre_tfm_output_proj_w, cur);
    if (model.pre_tfm_output_proj_b) {
        cur = ggml_add(ctx0, cur, model.pre_tfm_output_proj_b);
    }
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [T, latent_dim]
    for (int i = 0; i < 2; ++i) {
        cur = apply_upsample_block_stream(
            ctx0, gf, cur, model.upsample[i], state.stream_upsample_dw[i], i);
    }
    cur = apply_causal_conv_stream(ctx0, gf, cur, model.dec0_conv_w, model.dec0_conv_b,
                                   7, 1, state.stream_dec_pre);
    for (int i = 0; i < 4; ++i) {
        cur = apply_decoder_block_stream(
            ctx0, gf, self, cur, model.dec_blocks[i], cfg.upsample_rates[i],
            state.stream_dec_carry[i], state.stream_dec_res[i], i);
    }
    cur = apply_snake(ctx0, cur, model.dec5_snake_alpha, model.dec5_snake_beta,
                      model.dec5_snake_alpha_exp, model.dec5_snake_inv_beta_exp);
    cur = apply_causal_conv_stream(ctx0, gf, cur, model.dec6_conv_w, model.dec6_conv_b,
                                   7, 1, state.stream_dec_post);
    cur = ggml_tanh(ctx0, cur);
    cur = ggml_reshape_1d(ctx0, cur, cur->ne[0]);
    ggml_set_name(cur, "stream_audio");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    graph.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(state.backend));
    if (!graph.galloc || !ggml_gallocr_alloc_graph(graph.galloc, gf)) {
        error_msg = "Failed to allocate streaming decoder graph";
        if (graph.galloc) {
            ggml_gallocr_free(graph.galloc);
        }
        ggml_free(graph.ctx);
        state.stream_graphs.erase(n_frames);
        return false;
    }
    graph.gf = gf;
    graph.codes = codes;
    graph.positions = positions;
    graph.rows = rows;
    graph.mask = mask;
    graph.audio = cur;
    fprintf(stderr, "  Stateful decoder graph: %d frames, %.2f MiB scratch\n",
            n_frames,
            (double) ggml_gallocr_get_buffer_size(graph.galloc, 0) / (1024.0 * 1024.0));
    return true;
}

} // namespace qwen3_tts
