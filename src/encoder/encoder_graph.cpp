#include "audio_tokenizer_encoder.h"
#include "encoder/encoder_state_internal.h"

#include <cstdio>

namespace {

struct ggml_tensor * apply_reflect_pad_1d(struct ggml_context * ctx,
                                          struct ggml_tensor * x,
                                          int pad) {
    if (pad == 0) {
        return x;
    }
    return ggml_pad_reflect_1d(ctx, x, pad, pad);
}

struct ggml_tensor * apply_conv1d(struct ggml_context * ctx,
                                  struct ggml_tensor * w,
                                  struct ggml_tensor * b,
                                  struct ggml_tensor * x,
                                  int stride, int pad, int dilation,
                                  const char * debug_name = nullptr,
                                  bool use_reflect_pad = true) {
    struct ggml_tensor * input = x;
    int actual_pad = pad;

    if (use_reflect_pad && pad > 0) {
        input = ggml_pad_reflect_1d(ctx, x, pad, pad);
        actual_pad = 0;
    }

    struct ggml_tensor * y = ggml_conv_1d(ctx, w, input, stride, actual_pad, dilation);
    if (debug_name) {
        char name[64];
        snprintf(name, sizeof(name), "%s_conv", debug_name);
        ggml_set_name(y, name);
    }
    if (b) {
        const int64_t oc = y->ne[1];
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, oc, 1));
    }
    return y;
}

} // namespace

namespace qwen3_tts {

struct ggml_cgraph * encoder_internal::ops::build_graph(AudioTokenizerEncoder & self, int32_t n_frames) {
    const auto & model = self.impl_->model;
    const auto & state = self.impl_->state;
    const auto & cfg = model.config;
    const int hidden_dim = cfg.hidden_dim;
    const int scale = cfg.res2net_scale;
    const int branch_dim = hidden_dim / scale;

    struct ggml_init_params params = {
        /*.mem_size   =*/ state.compute_meta.size(),
        /*.mem_buffer =*/ const_cast<uint8_t *>(state.compute_meta.data()),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        self.error_msg_ = "Failed to initialize encoder graph context";
        return nullptr;
    }

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, encoder_internal::QWEN3_TTS_ENC_MAX_NODES, false);
    if (!gf) {
        self.error_msg_ = "Failed to create encoder graph";
        ggml_free(ctx0);
        return nullptr;
    }

    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, cfg.n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    struct ggml_tensor * cur = ggml_reshape_3d(ctx0, mel, n_frames, cfg.n_mels, 1);
    ggml_set_name(cur, "mel_3d");

    struct ggml_tensor * mel_padded = apply_reflect_pad_1d(ctx0, cur, 2);
    ggml_set_name(mel_padded, "mel_padded");

    cur = ggml_conv_1d(ctx0, model.conv0_w, mel_padded, 1, 0, 1);
    ggml_set_name(cur, "conv0_conv");
    if (model.conv0_b) {
        const int64_t oc = cur->ne[1];
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.conv0_b, 1, oc, 1));
    }
    ggml_set_name(cur, "conv0_pre_relu");
    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "conv0_out");

    const int64_t seq_len = cur->ne[0];
    struct ggml_tensor * block_outputs[4];
    block_outputs[0] = cur;

    const int dilations[3] = {2, 3, 4};
    for (int blk = 0; blk < 3; ++blk) {
        const auto & block = model.blocks[blk];
        const int dilation = dilations[blk];

        struct ggml_tensor * residual = cur;

        cur = apply_conv1d(ctx0, block.tdnn1_w, block.tdnn1_b, cur, 1, 0, 1);
        cur = ggml_relu(ctx0, cur);
        if (blk == 0) {
            ggml_set_name(cur, "blk1_tdnn1");
        }

        struct ggml_tensor * branches[8];
        for (int b = 0; b < scale; ++b) {
            branches[b] = ggml_view_3d(ctx0, cur, seq_len, branch_dim, 1,
                                       cur->nb[1], cur->nb[2], b * branch_dim * cur->nb[1]);
            branches[b] = ggml_cont(ctx0, branches[b]);
        }

        struct ggml_tensor * outputs[8];
        outputs[0] = branches[0];
        for (int b = 1; b < scale; ++b) {
            struct ggml_tensor * input = (b == 1) ? branches[b] : ggml_add(ctx0, branches[b], outputs[b - 1]);
            if (block.res2net_w[b - 1]) {
                outputs[b] = apply_conv1d(ctx0, block.res2net_w[b - 1], block.res2net_b[b - 1],
                                          input, 1, dilation, dilation);
                outputs[b] = ggml_relu(ctx0, outputs[b]);
            } else {
                outputs[b] = input;
            }
        }

        cur = outputs[0];
        for (int b = 1; b < scale; ++b) {
            cur = ggml_concat(ctx0, cur, outputs[b], 1);
        }
        if (blk == 0) {
            ggml_set_name(cur, "blk1_res2net");
            for (int b = 0; b < scale; ++b) {
                char name[32];
                snprintf(name, sizeof(name), "blk1_branch%d", b);
                ggml_set_name(outputs[b], name);
            }
        }

        cur = apply_conv1d(ctx0, block.tdnn2_w, block.tdnn2_b, cur, 1, 0, 1);
        cur = ggml_relu(ctx0, cur);
        if (blk == 0) {
            ggml_set_name(cur, "blk1_tdnn2");
        }

        struct ggml_tensor * se = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
        se = ggml_reshape_3d(ctx0, se, 1, hidden_dim, 1);
        se = apply_conv1d(ctx0, block.se_conv1_w, block.se_conv1_b, se, 1, 0, 1);
        se = ggml_relu(ctx0, se);
        se = apply_conv1d(ctx0, block.se_conv2_w, block.se_conv2_b, se, 1, 0, 1);
        se = ggml_sigmoid(ctx0, se);

        cur = ggml_mul(ctx0, cur, se);
        if (blk == 0) {
            ggml_set_name(cur, "blk1_se");
        }

        cur = ggml_add(ctx0, cur, residual);

        char block_name[32];
        snprintf(block_name, sizeof(block_name), "block_%d", blk + 1);
        ggml_set_name(cur, block_name);
        block_outputs[blk + 1] = cur;
    }

    struct ggml_tensor * mfa_input = ggml_concat(ctx0, block_outputs[1], block_outputs[2], 1);
    mfa_input = ggml_concat(ctx0, mfa_input, block_outputs[3], 1);
    ggml_set_name(mfa_input, "mfa_input");

    cur = apply_conv1d(ctx0, model.mfa_w, model.mfa_b, mfa_input, 1, 0, 1);
    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "mfa_out");

    struct ggml_tensor * global_mean = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    global_mean = ggml_reshape_3d(ctx0, global_mean, 1, 1536, 1);

    struct ggml_tensor * sq = ggml_sqr(ctx0, cur);
    struct ggml_tensor * mean_sq = ggml_pool_1d(ctx0, sq, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    mean_sq = ggml_reshape_3d(ctx0, mean_sq, 1, 1536, 1);
    struct ggml_tensor * var = ggml_sub(ctx0, mean_sq, ggml_sqr(ctx0, global_mean));
    var = ggml_clamp(ctx0, var, 1e-12f, 1e10f);
    struct ggml_tensor * global_std = ggml_sqrt(ctx0, var);

    struct ggml_tensor * mean_expanded = ggml_repeat(
        ctx0, global_mean, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, 1536, 1));
    struct ggml_tensor * std_expanded = ggml_repeat(
        ctx0, global_std, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, 1536, 1));
    struct ggml_tensor * attention = ggml_concat(ctx0, cur, mean_expanded, 1);
    attention = ggml_concat(ctx0, attention, std_expanded, 1);

    attention = apply_conv1d(ctx0, model.asp_tdnn_w, model.asp_tdnn_b, attention, 1, 0, 1);
    attention = ggml_relu(ctx0, attention);
    ggml_set_name(attention, "asp_tdnn");
    attention = ggml_tanh(ctx0, attention);

    attention = apply_conv1d(ctx0, model.asp_conv_w, model.asp_conv_b, attention, 1, 0, 1);
    ggml_set_name(attention, "asp_conv");

    attention = ggml_soft_max(ctx0, attention);
    ggml_set_name(attention, "asp_softmax");

    struct ggml_tensor * weighted = ggml_mul(ctx0, attention, cur);
    struct ggml_tensor * weighted_mean = ggml_pool_1d(ctx0, weighted, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    weighted_mean = ggml_scale(ctx0, weighted_mean, (float) seq_len);
    weighted_mean = ggml_reshape_3d(ctx0, weighted_mean, 1, 1536, 1);

    struct ggml_tensor * mean_for_std = ggml_repeat(
        ctx0, weighted_mean, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, 1536, 1));
    struct ggml_tensor * diff = ggml_sub(ctx0, cur, mean_for_std);
    struct ggml_tensor * diff_sq = ggml_sqr(ctx0, diff);
    struct ggml_tensor * weighted_var = ggml_mul(ctx0, attention, diff_sq);
    struct ggml_tensor * var_sum = ggml_pool_1d(ctx0, weighted_var, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    var_sum = ggml_scale(ctx0, var_sum, (float) seq_len);
    var_sum = ggml_reshape_3d(ctx0, var_sum, 1, 1536, 1);
    var_sum = ggml_clamp(ctx0, var_sum, 1e-12f, 1e10f);
    struct ggml_tensor * weighted_std = ggml_sqrt(ctx0, var_sum);

    struct ggml_tensor * pooled = ggml_concat(ctx0, weighted_mean, weighted_std, 1);
    ggml_set_name(pooled, "asp_pooled");

    cur = apply_conv1d(ctx0, model.fc_w, model.fc_b, pooled, 1, 0, 1);
    ggml_set_name(cur, "fc_out");

    cur = ggml_reshape_1d(ctx0, cur, cfg.embedding_dim);
    ggml_set_name(cur, "embedding");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    return gf;
}

} // namespace qwen3_tts
