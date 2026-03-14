#pragma once

#include "tts_transformer.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "coreml_code_predictor.h"

#include <map>
#include <random>
#include <string>
#include <vector>
#ifdef QWEN3_TTS_TIMING
#include <chrono>
#endif

namespace qwen3_tts {

#ifdef QWEN3_TTS_TIMING
struct tts_timing {
    double t_prefill_build_ms = 0;
    double t_prefill_forward_ms = 0;
    double t_prefill_graph_build_ms = 0;
    double t_prefill_graph_alloc_ms = 0;
    double t_prefill_compute_ms = 0;
    double t_prefill_data_ms = 0;

    double t_talker_forward_ms = 0;
    double t_talker_graph_build_ms = 0;
    double t_talker_graph_alloc_ms = 0;
    double t_talker_compute_ms = 0;
    double t_talker_data_ms = 0;

    double t_code_pred_ms = 0;
    double t_code_pred_init_ms = 0;
    double t_code_pred_prefill_ms = 0;
    double t_code_pred_steps_ms = 0;
    double t_code_pred_graph_build_ms = 0;
    double t_code_pred_graph_alloc_ms = 0;
    double t_code_pred_compute_ms = 0;
    double t_code_pred_data_ms = 0;
    double t_code_pred_coreml_ms = 0;

    double t_embed_lookup_ms = 0;

    int32_t n_frames = 0;
    double t_generate_total_ms = 0;
};
#endif

#define QWEN3_TTS_MAX_NODES 16384

struct transformer_layer {
    struct ggml_tensor * attn_norm = nullptr;

    struct ggml_tensor * attn_q = nullptr;
    struct ggml_tensor * attn_k = nullptr;
    struct ggml_tensor * attn_v = nullptr;
    struct ggml_tensor * attn_output = nullptr;
    struct ggml_tensor * attn_q_norm = nullptr;
    struct ggml_tensor * attn_k_norm = nullptr;

    struct ggml_tensor * ffn_norm = nullptr;

    struct ggml_tensor * ffn_gate = nullptr;
    struct ggml_tensor * ffn_up = nullptr;
    struct ggml_tensor * ffn_down = nullptr;
};

struct tts_transformer_model {
    tts_transformer_config config;

    struct ggml_tensor * text_embd = nullptr;
    struct ggml_tensor * text_proj_fc1 = nullptr;
    struct ggml_tensor * text_proj_fc1_bias = nullptr;
    struct ggml_tensor * text_proj_fc2 = nullptr;
    struct ggml_tensor * text_proj_fc2_bias = nullptr;

    struct ggml_tensor * codec_embd = nullptr;

    std::vector<transformer_layer> layers;

    struct ggml_tensor * output_norm = nullptr;
    struct ggml_tensor * codec_head = nullptr;

    std::vector<transformer_layer> code_pred_layers;
    struct ggml_tensor * code_pred_output_norm = nullptr;

    struct ggml_tensor * code_pred_small_to_mtp_weight = nullptr;
    struct ggml_tensor * code_pred_small_to_mtp_bias = nullptr;

    std::vector<struct ggml_tensor *> code_pred_embd;
    std::vector<struct ggml_tensor *> code_pred_head;

    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    std::map<std::string, struct ggml_tensor *> tensors;
};

struct tts_kv_cache {
    std::vector<struct ggml_tensor *> k_cache;
    std::vector<struct ggml_tensor *> v_cache;

    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    int32_t n_ctx = 0;
    int32_t n_used = 0;
    int32_t head_dim = 128;
    int32_t n_kv_heads = 8;
    int32_t n_layers = 28;
};

struct tts_transformer_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    bool sched_reserved = false;
    bool sched_reserve_failed = false;
    int32_t sched_reserved_ctx = 0;
    int32_t sched_reserved_prefill_len = 0;

    std::vector<uint8_t> compute_meta;
    std::vector<std::vector<uint8_t>> code_pred_compute_meta;
    std::vector<ggml_fp16_t> code_pred_mask;

    tts_kv_cache cache;
    tts_kv_cache code_pred_cache;
};

struct tts_transformer_private {
    tts_transformer_model model;
    tts_transformer_state state;
    int32_t n_threads = 0;
    std::vector<float> codec_embd_host;
    std::vector<std::vector<float>> code_pred_embd_host;
    std::vector<ggml_fp16_t> embd_row_fp16_scratch;
    std::mt19937 rng{std::random_device{}()};
    CoreMLCodePredictor coreml_code_predictor;
    bool use_coreml_code_predictor = false;
    std::string coreml_code_predictor_path;
    bool skip_ggml_code_pred_layers = false;

#ifdef QWEN3_TTS_TIMING
    tts_timing * timing = nullptr;
#endif
};

void free_transformer_model(tts_transformer_model & model);
void free_tts_kv_cache(tts_kv_cache & cache);

} // namespace qwen3_tts
