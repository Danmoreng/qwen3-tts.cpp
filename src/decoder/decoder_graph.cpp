#include "audio_tokenizer_decoder.h"
#include "decoder/decoder_state_internal.h"

#include <cstdio>

namespace qwen3_tts {

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

    struct ggml_tensor * rest_proj_2d = nullptr;
    for (int cb = 0; cb < 15; ++cb) {
        struct ggml_tensor * cb_emb_2d = ggml_reshape_2d(ctx0, rest_emb[cb], cfg.codebook_dim, n_frames);

        if (cb == 0) {
            ggml_set_name(cb_emb_2d, "rest_cb0_emb_2d");
        }

        struct ggml_tensor * cb_proj_2d = ggml_mul_mat(ctx0, rest_proj_weight_2d, cb_emb_2d);

        if (rest_proj_2d == nullptr) {
            rest_proj_2d = cb_proj_2d;
        } else {
            rest_proj_2d = ggml_add(ctx0, rest_proj_2d, cb_proj_2d);
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
    struct ggml_tensor * cur = ggml_conv_1d(ctx0, model.pre_conv_w, latent_padded, 1, 0, 1);
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

    for (int i = 0; i < cfg.n_pre_tfm_layers; ++i) {
        cur = apply_pre_tfm_layer(ctx0, self, cur, model.pre_tfm_layers[i], n_frames, positions);
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
    cur = ggml_conv_1d(ctx0, model.dec0_conv_w, cur, 1, 0, 1);
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
        cur = apply_snake(ctx0, cur, model.dec5_snake_alpha, model.dec5_snake_beta);
    }

    ggml_set_name(cur, "dec5_output");

    cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, model.dec6_conv_w, cur, 1, 0, 1);
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

} // namespace qwen3_tts
