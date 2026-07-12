#include "audio_tokenizer_decoder.h"
#include "decoder/decoder_state_internal.h"

#include <cmath>

namespace qwen3_tts {

namespace {

struct ggml_tensor * as_f16_conv_weight(struct ggml_context * ctx, struct ggml_tensor * w) {
    if (!w || w->type == GGML_TYPE_F16) {
        return w;
    }
    return ggml_cont(ctx, ggml_cast(ctx, w, GGML_TYPE_F16));
}

struct ggml_tensor * qwen_causal_trans_conv1d(struct ggml_context * ctx,
                                              struct ggml_tensor * w_perm,
                                              struct ggml_tensor * b,
                                              struct ggml_tensor * x,
                                              int stride,
                                              int kernel,
                                              int out_channels) {
    const int trim = kernel - stride;

    struct ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor * col = ggml_mul_mat(ctx, w_perm, xt);
    struct ggml_tensor * y = ggml_col2im_1d(ctx, col, stride, out_channels, 0);

    if (trim > 0) {
        const int64_t keep = y->ne[0] - trim;
        y = ggml_view_2d(ctx, y, keep, y->ne[1], y->nb[1], 0);
    }

    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, out_channels));
    }

    return y;
}

struct ggml_tensor * qwen_causal_trans_conv1d_stream(struct ggml_context * ctx,
                                                      struct ggml_cgraph * gf,
                                                      struct ggml_tensor * w_perm,
                                                      struct ggml_tensor * b,
                                                      struct ggml_tensor * x,
                                                      int stride,
                                                      int kernel,
                                                      int out_channels,
                                                      struct ggml_tensor * carry) {
    const int n_frames = (int) x->ne[0];
    const int trim = kernel - stride;
    const int emit = n_frames * stride;
    if (trim <= 0) {
        return qwen_causal_trans_conv1d(ctx, w_perm, b, x, stride, kernel, out_channels);
    }

    struct ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor * col = ggml_mul_mat(ctx, w_perm, xt);
    struct ggml_tensor * raw = ggml_col2im_1d(ctx, col, stride, out_channels, 0);

    struct ggml_tensor * head = ggml_view_2d(ctx, raw, trim, raw->ne[1], raw->nb[1], 0);
    struct ggml_tensor * y = ggml_add(ctx, head, carry);
    if (emit > trim) {
        struct ggml_tensor * middle = ggml_view_2d(
            ctx, raw, emit - trim, raw->ne[1], raw->nb[1], (size_t) trim * raw->nb[0]);
        y = ggml_concat(ctx, y, middle, 0);
    }
    ggml_build_forward_expand(gf, y);

    struct ggml_tensor * tail = ggml_view_2d(
        ctx, raw, trim, raw->ne[1], raw->nb[1], (size_t) emit * raw->nb[0]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, tail, carry));
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, out_channels));
    }
    return y;
}

} // namespace

struct ggml_tensor * decoder_internal::ops::apply_snake(struct ggml_context * ctx,
                                                        struct ggml_tensor * x,
                                                        struct ggml_tensor * alpha,
                                                        struct ggml_tensor * beta,
                                                        struct ggml_tensor * alpha_exp,
                                                        struct ggml_tensor * inv_beta_exp) {
    if (alpha_exp && inv_beta_exp) {
        struct ggml_tensor * ax = ggml_mul(ctx, x, alpha_exp);
        struct ggml_tensor * sin_ax = ggml_sin(ctx, ax);
        struct ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);
        struct ggml_tensor * scaled_sin = ggml_mul(ctx, sin_sq, inv_beta_exp);
        return ggml_add(ctx, x, scaled_sin);
    }

    int64_t seq_len = x->ne[0];
    int64_t channels = x->ne[1];
    int64_t batch = x->ne[2];

    struct ggml_tensor * alpha_exp_runtime = ggml_exp(ctx, alpha);

    struct ggml_tensor * alpha_3d = ggml_reshape_3d(ctx, alpha_exp_runtime, 1, channels, 1);
    struct ggml_tensor * alpha_broad = ggml_repeat(ctx, alpha_3d,
                                                    ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));

    struct ggml_tensor * neg_beta = ggml_scale(ctx, beta, -1.0f);
    struct ggml_tensor * inv_beta_exp_runtime = ggml_exp(ctx, neg_beta);
    struct ggml_tensor * inv_beta_3d = ggml_reshape_3d(ctx, inv_beta_exp_runtime, 1, channels, 1);
    struct ggml_tensor * inv_beta = ggml_repeat(ctx, inv_beta_3d,
                                                 ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, channels, batch));

    struct ggml_tensor * ax = ggml_mul(ctx, x, alpha_broad);
    struct ggml_tensor * sin_ax = ggml_sin(ctx, ax);
    struct ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);
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
                                                                struct ggml_tensor * positions,
                                                                struct ggml_tensor * mask) {
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
    KQ = ggml_soft_max_ext(ctx, KQ, mask, 1.0f / sqrtf((float) head_dim), 0.0f);

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

struct ggml_tensor * decoder_internal::ops::apply_pre_tfm_layer_stream(
                                                                struct ggml_context * ctx,
                                                                struct ggml_cgraph * gf,
                                                                AudioTokenizerDecoder & self,
                                                                struct ggml_tensor * x,
                                                                const pre_tfm_layer & layer,
                                                                int32_t n_frames,
                                                                struct ggml_tensor * positions,
                                                                struct ggml_tensor * rows,
                                                                struct ggml_tensor * mask,
                                                                struct ggml_tensor * k_cache,
                                                                struct ggml_tensor * v_cache,
                                                                int32_t ring) {
    const auto & cfg = self.impl_->model.config;
    const int n_heads = cfg.n_heads;
    const int head_dim = cfg.latent_dim / n_heads;
    struct ggml_tensor * residual = x;
    struct ggml_tensor * normed = apply_rms_norm(ctx, x, layer.attn_norm_w, cfg.rms_norm_eps);
    struct ggml_tensor * q = ggml_mul_mat(ctx, layer.attn_q_w, normed);
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer.attn_k_w, normed);
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer.attn_v_w, normed);
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_frames);
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads, n_frames);
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads, n_frames);
    q = ggml_rope_ext(ctx, q, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                      cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                      cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    struct ggml_tensor * k_rows = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    struct ggml_tensor * v_rows = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache, k_rows, rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache, v_rows, rows));

    struct ggml_tensor * q_rows = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    struct ggml_tensor * k_full = ggml_view_3d(
        ctx, k_cache, head_dim, ring, n_heads, k_cache->nb[1], k_cache->nb[2], 0);
    struct ggml_tensor * v_full = ggml_view_3d(
        ctx, v_cache, head_dim, ring, n_heads, v_cache->nb[1], v_cache->nb[2], 0);
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k_full, q_rows);
    scores = ggml_soft_max_ext(ctx, scores, mask, 1.0f / sqrtf((float) head_dim), 0.0f);
    struct ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, v_full));
    struct ggml_tensor * attn = ggml_mul_mat(ctx, vt, scores);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx, attn, n_heads * head_dim, n_frames);
    attn = ggml_mul_mat(ctx, layer.attn_output_w, attn);
    if (layer.attn_scale) {
        attn = ggml_mul(ctx, attn, layer.attn_scale);
    }
    x = ggml_add(ctx, residual, attn);
    residual = x;
    normed = apply_rms_norm(ctx, x, layer.ffn_norm_w, cfg.rms_norm_eps);
    struct ggml_tensor * gate = ggml_silu(ctx, ggml_mul_mat(ctx, layer.ffn_gate_w, normed));
    struct ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up_w, normed);
    struct ggml_tensor * ffn = ggml_mul_mat(ctx, layer.ffn_down_w, ggml_mul(ctx, gate, up));
    if (layer.ffn_scale) {
        ffn = ggml_mul(ctx, ffn, layer.ffn_scale);
    }
    return ggml_add(ctx, residual, ffn);
}

struct ggml_tensor * decoder_internal::ops::apply_causal_conv_stream(
                                                                struct ggml_context * ctx,
                                                                struct ggml_cgraph * gf,
                                                                struct ggml_tensor * x,
                                                                struct ggml_tensor * weight,
                                                                struct ggml_tensor * bias,
                                                                int kernel,
                                                                int dilation,
                                                                struct ggml_tensor * stream_state) {
    const int left = (kernel - 1) * dilation;
    const int out_channels = (int) weight->ne[2];
    struct ggml_tensor * extended = ggml_concat(ctx, stream_state, x, 0);
    struct ggml_tensor * tail = ggml_view_2d(
        ctx, extended, left, extended->ne[1], extended->nb[1],
        (size_t) (extended->ne[0] - left) * extended->nb[0]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, tail, stream_state));
    struct ggml_tensor * y = ggml_reshape_3d(ctx, extended, extended->ne[0], extended->ne[1], 1);
    y = ggml_conv_1d(ctx, as_f16_conv_weight(ctx, weight), y, 1, 0, dilation);
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (bias) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, bias, 1, out_channels));
    }
    return y;
}

struct ggml_tensor * decoder_internal::ops::apply_upsample_block(struct ggml_context * ctx,
                                                                 struct ggml_tensor * x,
                                                                 const upsample_block & block,
                                                                 int block_idx) {
    (void) block_idx;
    int64_t seq_len = x->ne[0];
    int64_t channels = x->ne[1];

    struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, seq_len, channels);
    if (block.conv_w_perm) {
        x_2d = qwen_causal_trans_conv1d(ctx, block.conv_w_perm, block.conv_b,
                                        x_2d, 2, 2, (int) channels);
    } else {
        x_2d = ggml_conv_transpose_1d(ctx, block.conv_w, x_2d, 2, 0, 1);
    }

    int64_t new_seq_len = x_2d->ne[0];
    x = ggml_reshape_3d(ctx, x_2d, new_seq_len, channels, 1);

    if (!block.conv_w_perm && block.conv_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_b, 1, channels, 1));
    }

    struct ggml_tensor * residual = x;

    if (block.dwconv_w) {
        x = ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0);
        x = ggml_conv_1d_dw(ctx, as_f16_conv_weight(ctx, block.dwconv_w), x, 1, 0, 1);
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

struct ggml_tensor * decoder_internal::ops::apply_upsample_block_stream(
                                                                 struct ggml_context * ctx,
                                                                 struct ggml_cgraph * gf,
                                                                 struct ggml_tensor * x,
                                                                 const upsample_block & block,
                                                                 struct ggml_tensor * dw_state,
                                                                 int block_idx) {
    (void) block_idx;
    const int channels = (int) x->ne[1];
    x = qwen_causal_trans_conv1d(ctx, block.conv_w_perm, block.conv_b, x, 2, 2, channels);
    struct ggml_tensor * residual = x;
    struct ggml_tensor * extended = ggml_concat(ctx, dw_state, x, 0);
    struct ggml_tensor * tail = ggml_view_2d(
        ctx, extended, 6, channels, extended->nb[1],
        (size_t) (extended->ne[0] - 6) * extended->nb[0]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, tail, dw_state));
    struct ggml_tensor * y = ggml_reshape_3d(ctx, extended, extended->ne[0], channels, 1);
    y = ggml_conv_1d_dw(ctx, as_f16_conv_weight(ctx, block.dwconv_w), y, 1, 0, 1);
    y = ggml_reshape_2d(ctx, y, y->ne[0], channels);
    if (block.dwconv_b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, block.dwconv_b, 1, channels));
    }
    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    y = ggml_norm(ctx, y, 1e-6f);
    y = ggml_mul(ctx, y, block.norm_w);
    y = ggml_add(ctx, y, block.norm_b);
    y = ggml_add(ctx, ggml_mul_mat(ctx, block.pwconv1_w, y), block.pwconv1_b);
    y = ggml_gelu(ctx, y);
    y = ggml_add(ctx, ggml_mul_mat(ctx, block.pwconv2_w, y), block.pwconv2_b);
    if (block.gamma) {
        y = ggml_mul(ctx, y, block.gamma);
    }
    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    return ggml_add(ctx, y, residual);
}

struct ggml_tensor * decoder_internal::ops::apply_residual_block(struct ggml_context * ctx,
                                                                 struct ggml_tensor * x,
                                                                 const residual_block & block) {
    struct ggml_tensor * residual = x;

    if (block.act1_alpha) {
        x = apply_snake(ctx, x, block.act1_alpha, block.act1_beta,
                        block.act1_alpha_exp, block.act1_inv_beta_exp);
    }

    int64_t out_channels = block.conv1_w->ne[2];
    int padding = 6 * block.dilation;
    x = ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0);
    x = ggml_conv_1d(ctx, as_f16_conv_weight(ctx, block.conv1_w), x, 1, 0, block.dilation);
    if (block.conv1_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv1_b, 1, out_channels, 1));
    }

    if (block.act2_alpha) {
        x = apply_snake(ctx, x, block.act2_alpha, block.act2_beta,
                        block.act2_alpha_exp, block.act2_inv_beta_exp);
    }

    out_channels = block.conv2_w->ne[2];
    x = ggml_conv_1d(ctx, as_f16_conv_weight(ctx, block.conv2_w), x, 1, 0, 1);
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
        x = apply_snake(ctx, x, block.snake_alpha, block.snake_beta,
                        block.snake_alpha_exp, block.snake_inv_beta_exp);
    }

    int64_t seq_len = x->ne[0];
    int64_t in_channels = x->ne[1];
    int64_t out_channels = block.conv_t_w->ne[1];
    int kernel_size = block.conv_t_w->ne[0];

    struct ggml_tensor * x_2d = ggml_reshape_2d(ctx, x, seq_len, in_channels);
    if (block.conv_t_w_perm) {
        x_2d = qwen_causal_trans_conv1d(ctx, block.conv_t_w_perm, block.conv_t_b,
                                        x_2d, upsample_rate, kernel_size, (int) out_channels);
    } else {
        x_2d = ggml_conv_transpose_1d(ctx, block.conv_t_w, x_2d, upsample_rate, 0, 1);
    }

    int64_t new_seq_len = x_2d->ne[0];
    x = ggml_reshape_3d(ctx, x_2d, new_seq_len, out_channels, 1);

    if (!block.conv_t_w_perm) {
        int pad = kernel_size - upsample_rate;
        int left_pad = pad;
        int right_pad = pad;
        int64_t out_seq_len = new_seq_len - left_pad - right_pad;

        x = ggml_view_3d(ctx, x, out_seq_len, out_channels, 1,
                         x->nb[1], x->nb[2], left_pad * x->nb[0]);
        x = ggml_cont(ctx, x);
    }

    if (!block.conv_t_w_perm && block.conv_t_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, block.conv_t_b, 1, out_channels, 1));
    }

    for (int i = 0; i < 3; ++i) {
        x = apply_residual_block(ctx, x, block.res[i]);
    }

    return x;
}

struct ggml_tensor * decoder_internal::ops::apply_decoder_block_stream(
                                                                struct ggml_context * ctx,
                                                                struct ggml_cgraph * gf,
                                                                AudioTokenizerDecoder & self,
                                                                struct ggml_tensor * x,
                                                                const decoder_block & block,
                                                                int upsample_rate,
                                                                struct ggml_tensor * carry,
                                                                struct ggml_tensor * const res_state[3],
                                                                int block_idx) {
    (void) self;
    (void) block_idx;
    x = apply_snake(ctx, x, block.snake_alpha, block.snake_beta,
                    block.snake_alpha_exp, block.snake_inv_beta_exp);
    const int out_channels = (int) block.conv_t_w->ne[1];
    const int kernel = (int) block.conv_t_w->ne[0];
    x = qwen_causal_trans_conv1d_stream(ctx, gf, block.conv_t_w_perm, block.conv_t_b,
                                        x, upsample_rate, kernel, out_channels, carry);
    for (int i = 0; i < 3; ++i) {
        const residual_block & res = block.res[i];
        struct ggml_tensor * residual = x;
        x = apply_snake(ctx, x, res.act1_alpha, res.act1_beta,
                        res.act1_alpha_exp, res.act1_inv_beta_exp);
        x = apply_causal_conv_stream(ctx, gf, x, res.conv1_w, res.conv1_b,
                                     7, res.dilation, res_state[i]);
        x = apply_snake(ctx, x, res.act2_alpha, res.act2_beta,
                        res.act2_alpha_exp, res.act2_inv_beta_exp);
        x = ggml_conv_1d(ctx, as_f16_conv_weight(ctx, res.conv2_w),
                         ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1), 1, 0, 1);
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
        if (res.conv2_b) {
            x = ggml_add(ctx, x, ggml_reshape_2d(ctx, res.conv2_b, 1, out_channels));
        }
        x = ggml_add(ctx, residual, x);
    }
    return x;
}

} // namespace qwen3_tts
