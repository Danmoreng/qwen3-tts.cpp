#include "audio_tokenizer_decoder.h"
#include "decoder/decoder_state_internal.h"

#include <cmath>

namespace qwen3_tts {

struct ggml_tensor * decoder_internal::ops::apply_snake(struct ggml_context * ctx,
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

struct ggml_tensor * decoder_internal::ops::apply_rms_norm(struct ggml_context * ctx,
                                                           struct ggml_tensor * x,
                                                           struct ggml_tensor * w,
                                                           float eps) {
    struct ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normed, w);
}

struct ggml_tensor * decoder_internal::ops::apply_pre_tfm_layer(struct ggml_context * ctx,
                                                                AudioTokenizerDecoder & self,
                                                                struct ggml_tensor * x,
                                                                const pre_tfm_layer & layer,
                                                                int32_t n_frames,
                                                                struct ggml_tensor * positions) {
    const auto & cfg = self.impl_->model.config;
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
    KQ = ggml_scale(ctx, KQ, 1.0f / sqrtf((float) head_dim));
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

struct ggml_tensor * decoder_internal::ops::apply_upsample_block(struct ggml_context * ctx,
                                                                 struct ggml_tensor * x,
                                                                 const upsample_block & block,
                                                                 int block_idx) {
    (void) block_idx;
    int64_t seq_len = x->ne[0];
    int64_t channels = x->ne[1];

    struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, seq_len, channels);
    x_2d = ggml_conv_transpose_1d(ctx, block.conv_w, x_2d, 2, 0, 1);

    int64_t new_seq_len = x_2d->ne[0];
    x = ggml_reshape_3d(ctx, x_2d, new_seq_len, channels, 1);

    if (block.conv_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_b, 1, channels, 1));
    }

    struct ggml_tensor * residual = x;

    if (block.dwconv_w) {
        x = ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0);
        x = ggml_conv_1d_dw(ctx, block.dwconv_w, x, 1, 0, 1);
        if (block.dwconv_b) {
            x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.dwconv_b, 1, channels, 1));
        }
    }

    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);

    if (block.norm_w && block.norm_b) {
        x = ggml_norm(ctx, x, 1e-6f);
        x = ggml_mul(ctx, x, block.norm_w);
        x = ggml_add(ctx, x, block.norm_b);
    }

    x = ggml_mul_mat(ctx, block.pwconv1_w, x);
    if (block.pwconv1_b) {
        x = ggml_add(ctx, x, block.pwconv1_b);
    }

    x = ggml_gelu(ctx, x);

    x = ggml_mul_mat(ctx, block.pwconv2_w, x);
    if (block.pwconv2_b) {
        x = ggml_add(ctx, x, block.pwconv2_b);
    }

    x = ggml_permute(ctx, x, 1, 0, 2, 3);
    x = ggml_cont(ctx, x);

    if (block.gamma) {
        struct ggml_tensor * gamma_3d = ggml_reshape_3d(ctx, block.gamma, 1, channels, 1);
        x = ggml_mul(ctx, x, ggml_repeat(ctx, gamma_3d,
                                          ggml_new_tensor_3d(ctx, GGML_TYPE_F32, new_seq_len, channels, 1)));
    }

    return ggml_add(ctx, residual, x);
}

struct ggml_tensor * decoder_internal::ops::apply_residual_block(struct ggml_context * ctx,
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

struct ggml_tensor * decoder_internal::ops::apply_decoder_block(struct ggml_context * ctx,
                                                                AudioTokenizerDecoder & self,
                                                                struct ggml_tensor * x,
                                                                const decoder_block & block,
                                                                int upsample_rate,
                                                                int block_idx) {
    (void) self;
    (void) block_idx;
    if (block.snake_alpha && block.snake_beta) {
        x = apply_snake(ctx, x, block.snake_alpha, block.snake_beta);
    }

    int64_t seq_len = x->ne[0];
    int64_t in_channels = x->ne[1];
    int64_t out_channels = block.conv_t_w->ne[1];
    int kernel_size = block.conv_t_w->ne[0];

    struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, seq_len, in_channels);
    x_2d = ggml_conv_transpose_1d(ctx, block.conv_t_w, x_2d, upsample_rate, 0, 1);

    int64_t new_seq_len = x_2d->ne[0];
    x = ggml_reshape_3d(ctx, x_2d, new_seq_len, out_channels, 1);

    int pad = kernel_size - upsample_rate;
    int left_pad = pad;
    int right_pad = pad;
    int64_t out_seq_len = new_seq_len - left_pad - right_pad;

    x = ggml_view_3d(ctx, x, out_seq_len, out_channels, 1,
                     x->nb[1], x->nb[2], left_pad * x->nb[0]);
    x = ggml_cont(ctx, x);

    if (block.conv_t_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_t_b, 1, out_channels, 1));
    }

    for (int i = 0; i < 3; ++i) {
        x = apply_residual_block(ctx, x, block.res[i]);
    }

    return x;
}

} // namespace qwen3_tts
