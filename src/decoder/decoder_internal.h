#pragma once

#include "audio_tokenizer_decoder.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <map>
#include <string>
#include <vector>

namespace qwen3_tts {

class AudioTokenizerDecoder;

inline constexpr int QWEN3_TTS_DEC_MAX_NODES = 32768;

// Pre-transformer layer weights
struct pre_tfm_layer {
    struct ggml_tensor * attn_norm_w = nullptr;
    struct ggml_tensor * attn_q_w = nullptr;
    struct ggml_tensor * attn_k_w = nullptr;
    struct ggml_tensor * attn_v_w = nullptr;
    struct ggml_tensor * attn_output_w = nullptr;
    struct ggml_tensor * attn_scale = nullptr;

    struct ggml_tensor * ffn_norm_w = nullptr;
    struct ggml_tensor * ffn_gate_w = nullptr;
    struct ggml_tensor * ffn_up_w = nullptr;
    struct ggml_tensor * ffn_down_w = nullptr;
    struct ggml_tensor * ffn_scale = nullptr;
};

// Residual block weights (Snake + Conv + Snake + Conv)
struct residual_block {
    int dilation = 1;
    struct ggml_tensor * act1_alpha = nullptr;
    struct ggml_tensor * act1_beta = nullptr;
    struct ggml_tensor * act1_alpha_exp = nullptr;
    struct ggml_tensor * act1_inv_beta_exp = nullptr;
    struct ggml_tensor * conv1_w = nullptr;
    struct ggml_tensor * conv1_b = nullptr;
    struct ggml_tensor * act2_alpha = nullptr;
    struct ggml_tensor * act2_beta = nullptr;
    struct ggml_tensor * act2_alpha_exp = nullptr;
    struct ggml_tensor * act2_inv_beta_exp = nullptr;
    struct ggml_tensor * conv2_w = nullptr;
    struct ggml_tensor * conv2_b = nullptr;
};

// Decoder block weights (Snake + ConvTranspose + Residual blocks)
struct decoder_block {
    struct ggml_tensor * snake_alpha = nullptr;
    struct ggml_tensor * snake_beta = nullptr;
    struct ggml_tensor * snake_alpha_exp = nullptr;
    struct ggml_tensor * snake_inv_beta_exp = nullptr;
    struct ggml_tensor * conv_t_w = nullptr;
    struct ggml_tensor * conv_t_w_perm = nullptr;
    struct ggml_tensor * conv_t_b = nullptr;
    residual_block res[3];
};

// Upsample block weights (ConvNeXt-style)
struct upsample_block {
    struct ggml_tensor * conv_w = nullptr;
    struct ggml_tensor * conv_w_perm = nullptr;
    struct ggml_tensor * conv_b = nullptr;
    struct ggml_tensor * dwconv_w = nullptr;
    struct ggml_tensor * dwconv_b = nullptr;
    struct ggml_tensor * norm_w = nullptr;
    struct ggml_tensor * norm_b = nullptr;
    struct ggml_tensor * pwconv1_w = nullptr;
    struct ggml_tensor * pwconv1_b = nullptr;
    struct ggml_tensor * pwconv2_w = nullptr;
    struct ggml_tensor * pwconv2_b = nullptr;
    struct ggml_tensor * gamma = nullptr;
};

// Audio tokenizer decoder model weights
struct audio_decoder_model {
    audio_decoder_config config;

    struct ggml_tensor * vq_first_input_proj = nullptr;
    struct ggml_tensor * vq_first_output_proj = nullptr;
    struct ggml_tensor * vq_first_codebook = nullptr;
    struct ggml_tensor * vq_first_usage = nullptr;

    struct ggml_tensor * vq_rest_input_proj = nullptr;
    struct ggml_tensor * vq_rest_output_proj = nullptr;
    struct ggml_tensor * vq_rest_codebook[15] = {nullptr};
    struct ggml_tensor * vq_rest_usage[15] = {nullptr};

    upsample_block upsample[2];

    struct ggml_tensor * pre_tfm_input_proj_w = nullptr;
    struct ggml_tensor * pre_tfm_input_proj_b = nullptr;
    pre_tfm_layer pre_tfm_layers[8];
    struct ggml_tensor * pre_tfm_norm_w = nullptr;
    struct ggml_tensor * pre_tfm_output_proj_w = nullptr;
    struct ggml_tensor * pre_tfm_output_proj_b = nullptr;

    struct ggml_tensor * pre_conv_w = nullptr;
    struct ggml_tensor * pre_conv_b = nullptr;

    struct ggml_tensor * dec0_conv_w = nullptr;
    struct ggml_tensor * dec0_conv_b = nullptr;

    decoder_block dec_blocks[4];

    struct ggml_tensor * dec5_snake_alpha = nullptr;
    struct ggml_tensor * dec5_snake_beta = nullptr;
    struct ggml_tensor * dec5_snake_alpha_exp = nullptr;
    struct ggml_tensor * dec5_snake_inv_beta_exp = nullptr;

    struct ggml_tensor * dec6_conv_w = nullptr;
    struct ggml_tensor * dec6_conv_b = nullptr;

    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct audio_decoder_state {
    ggml_backend_t backend = nullptr;
    bool backend_shared = true;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
    struct ggml_context * decode_graph_ctx = nullptr;
    struct ggml_cgraph * decode_graph = nullptr;
    struct ggml_tensor * decode_code_tensors[16] = {nullptr};
    struct ggml_tensor * decode_positions_tensor = nullptr;
    struct ggml_tensor * decode_mask_tensor = nullptr;
    struct ggml_tensor * decode_audio_tensor = nullptr;
    int32_t decode_graph_n_frames = 0;

    struct decoder_stream_graph {
        struct ggml_context * ctx = nullptr;
        struct ggml_cgraph * gf = nullptr;
        ggml_gallocr_t galloc = nullptr;
        struct ggml_tensor * codes = nullptr;
        struct ggml_tensor * positions = nullptr;
        struct ggml_tensor * rows = nullptr;
        struct ggml_tensor * mask = nullptr;
        struct ggml_tensor * audio = nullptr;
    };

    bool stream_ready = false;
    int32_t stream_pos = 0;
    int32_t stream_ring = 128;
    struct ggml_context * stream_ctx = nullptr;
    ggml_backend_buffer_t stream_buffer = nullptr;
    struct ggml_tensor * stream_pre_conv = nullptr;
    struct ggml_tensor * stream_upsample_dw[2] = {nullptr};
    struct ggml_tensor * stream_dec_pre = nullptr;
    struct ggml_tensor * stream_dec_carry[4] = {nullptr};
    struct ggml_tensor * stream_dec_res[4][3] = {{nullptr}};
    struct ggml_tensor * stream_dec_post = nullptr;
    std::vector<struct ggml_tensor *> stream_k;
    std::vector<struct ggml_tensor *> stream_v;
    std::map<int32_t, decoder_stream_graph> stream_graphs;
    struct ggml_context * stream_snapshot_ctx = nullptr;
    ggml_backend_buffer_t stream_snapshot_buffer = nullptr;
    uint64_t stream_snapshot_key = 0;
    int32_t stream_snapshot_pos = 0;
    bool stream_snapshot_valid = false;
};

namespace decoder_internal {

struct ops {
    static struct ggml_cgraph * build_graph(AudioTokenizerDecoder & self, int32_t n_frames);
    static struct ggml_cgraph * build_graph_impl(AudioTokenizerDecoder & self,
                                                 int32_t n_frames,
                                                 struct ggml_context ** graph_ctx_out);
    static void release_cached_decode_graph(AudioTokenizerDecoder & self);
    static bool ensure_cached_decode_graph(AudioTokenizerDecoder & self, int32_t n_frames);
    static void release_stream(AudioTokenizerDecoder & self);
    static void release_stream_graphs_above(AudioTokenizerDecoder & self, int32_t max_frames);
    static bool ensure_stream_state(AudioTokenizerDecoder & self);
    static bool ensure_stream_graph(AudioTokenizerDecoder & self, int32_t n_frames);
    static struct ggml_tensor * apply_snake(struct ggml_context * ctx,
                                            struct ggml_tensor * x,
                                            struct ggml_tensor * alpha,
                                            struct ggml_tensor * beta,
                                            struct ggml_tensor * alpha_exp = nullptr,
                                            struct ggml_tensor * inv_beta_exp = nullptr);
    static struct ggml_tensor * apply_rms_norm(struct ggml_context * ctx,
                                               struct ggml_tensor * x,
                                               struct ggml_tensor * w,
                                               float eps);
    static struct ggml_tensor * apply_pre_tfm_layer(struct ggml_context * ctx,
                                                    AudioTokenizerDecoder & self,
                                                    struct ggml_tensor * x,
                                                    const pre_tfm_layer & layer,
                                                    int32_t n_frames,
                                                    struct ggml_tensor * positions,
                                                    struct ggml_tensor * mask);
    static struct ggml_tensor * apply_pre_tfm_layer_stream(
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
                                                    int32_t ring);
    static struct ggml_tensor * apply_causal_conv_stream(
                                                    struct ggml_context * ctx,
                                                    struct ggml_cgraph * gf,
                                                    struct ggml_tensor * x,
                                                    struct ggml_tensor * weight,
                                                    struct ggml_tensor * bias,
                                                    int kernel,
                                                    int dilation,
                                                    struct ggml_tensor * stream_state);
    static struct ggml_tensor * apply_upsample_block(struct ggml_context * ctx,
                                                     struct ggml_tensor * x,
                                                     const upsample_block & block,
                                                     int block_idx);
    static struct ggml_tensor * apply_upsample_block_stream(
                                                     struct ggml_context * ctx,
                                                     struct ggml_cgraph * gf,
                                                     struct ggml_tensor * x,
                                                     const upsample_block & block,
                                                     struct ggml_tensor * dw_state,
                                                     int block_idx);
    static struct ggml_tensor * apply_residual_block(struct ggml_context * ctx,
                                                     struct ggml_tensor * x,
                                                     const residual_block & block);
    static struct ggml_tensor * apply_decoder_block(struct ggml_context * ctx,
                                                    AudioTokenizerDecoder & self,
                                                    struct ggml_tensor * x,
                                                    const decoder_block & block,
                                                    int upsample_rate,
                                                    int block_idx);
    static struct ggml_tensor * apply_decoder_block_stream(
                                                    struct ggml_context * ctx,
                                                    struct ggml_cgraph * gf,
                                                    AudioTokenizerDecoder & self,
                                                    struct ggml_tensor * x,
                                                    const decoder_block & block,
                                                    int upsample_rate,
                                                    struct ggml_tensor * carry,
                                                    struct ggml_tensor * const res_state[3],
                                                    int block_idx);
    static void normalize_codebooks(AudioTokenizerDecoder & self);
    static void prepare_snake_tensors(AudioTokenizerDecoder & self);
    static bool upload_snake_tensors(AudioTokenizerDecoder & self);
    static void prepare_transconv_tensors(AudioTokenizerDecoder & self);
    static bool upload_transconv_tensors(AudioTokenizerDecoder & self);
};

} // namespace decoder_internal

void free_audio_decoder_model(audio_decoder_model & model);

} // namespace qwen3_tts
