#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"
#include "transformer/transformer_sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace qwen3_tts {

bool TTSTransformer::get_hidden_states(std::vector<float> & hidden) const {
    if (last_hidden_.empty()) {
        return false;
    }
    hidden = last_hidden_;
    return true;
}

bool TTSTransformer::predict_codes(const float * hidden, const int32_t * prev_codes,
                                   std::vector<float> & output) {
    if (!impl_->model.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }

    const auto & cfg = impl_->model.config;
    int n_prev = (prev_codes != nullptr) ? cfg.n_codebooks - 1 : 0;

    struct ggml_cgraph * gf = transformer_internal::ops::build_code_pred_graph(*this, n_prev);

    if (!ggml_backend_sched_alloc_graph(impl_->state.sched, gf)) {
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

    if (ggml_backend_sched_graph_compute(impl_->state.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute code predictor graph";
        ggml_backend_sched_reset(impl_->state.sched);
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

    ggml_backend_sched_reset(impl_->state.sched);

    return true;
}

bool transformer_internal::ops::predict_codes_autoregressive_coreml(TTSTransformer & self,
                                                                    const float * hidden,
                                                                    int32_t codebook_0_token,
                                                                    std::vector<int32_t> & output,
                                                                    float temperature,
                                                                    int32_t top_k,
                                                                    float top_p,
                                                                    int64_t seed,
                                                                    int64_t * sampling_subseq,
                                                                    int32_t trace_frame) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;
    if (!impl->use_coreml_code_predictor || !impl->coreml_code_predictor.is_loaded()) {
        error_msg = "CoreML code predictor is not loaded";
        return false;
    }

    const auto & cfg = impl->model.config;
    const int32_t n_steps = cfg.n_codebooks - 1;
    const auto & trace_cfg = transformer_internal::get_debug_trace_config();
    const bool trace_frame_enabled = transformer_internal::debug_trace_should_dump_frame(trace_cfg, trace_frame);

    output.resize(n_steps);
    std::vector<float> logits_data(cfg.code_pred_vocab_size);
    std::vector<float> seq_embd((size_t) 16 * cfg.hidden_size, 0.0f);
    int64_t local_subseq = sampling_subseq ? *sampling_subseq : 0;
    transformer_sampling_state sampling{resolve_sampling_seed(seed), local_subseq};

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
        return transformer_sample_top_k_p(logits_ptr, vocab_size,
                                          temperature, top_k, top_p,
                                          1.0f, nullptr, 0, sampling);
    };

    memcpy(seq_embd.data(), hidden, (size_t) cfg.hidden_size * sizeof(float));
    if (!lookup_single_embedding_row(self, impl->model.codec_embd, codebook_0_token,
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
    if (impl->timing) impl->timing->t_code_pred_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    for (int32_t step = 0; step < n_steps; ++step) {
        if (step > 0) {
            float * dst = seq_embd.data() + (size_t) (step + 1) * cfg.hidden_size;
            if (!lookup_single_embedding_row(self, impl->model.code_pred_embd[step - 1], output[step - 1], dst)) {
                return false;
            }
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!impl->coreml_code_predictor.predict_step(step, seq_embd.data(), step + 2, cfg.hidden_size, logits_data)) {
            error_msg = "CoreML predictor step failed: " + impl->coreml_code_predictor.get_error();
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (impl->timing) impl->timing->t_code_pred_compute_ms += dt_ms;
        if (impl->timing) impl->timing->t_code_pred_coreml_ms += dt_ms;
#endif

        if ((int32_t) logits_data.size() != cfg.code_pred_vocab_size) {
            error_msg = "CoreML predictor returned unexpected logits size";
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
        if (impl->timing) {
            if (step == 0) {
                impl->timing->t_code_pred_prefill_ms += dt_ms;
            } else {
                impl->timing->t_code_pred_steps_ms += dt_ms;
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

    if (sampling_subseq) {
        *sampling_subseq = sampling.subseq;
    }
    return true;
}

bool TTSTransformer::predict_codes_autoregressive(const float * hidden, int32_t codebook_0_token,
                                                  std::vector<int32_t> & output,
                                                  float temperature, int32_t top_k,
                                                  float top_p,
                                                  int64_t seed,
                                                  int64_t * sampling_subseq,
                                                  int32_t trace_frame) {
    if (!impl_->model.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }

    const auto & cfg = impl_->model.config;
    const auto & trace_cfg = transformer_internal::get_debug_trace_config();
    const bool trace_frame_enabled = transformer_internal::debug_trace_should_dump_frame(trace_cfg, trace_frame);

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    if (impl_->use_coreml_code_predictor && impl_->coreml_code_predictor.is_loaded()) {
        if (transformer_internal::ops::predict_codes_autoregressive_coreml(*this, hidden, codebook_0_token, output, temperature, top_k, top_p, seed, sampling_subseq, trace_frame)) {
            return true;
        }
        if (impl_->skip_ggml_code_pred_layers) {
            return false;
        }
        fprintf(stderr, "  CoreML code predictor failed, falling back to GGML: %s\n", error_msg_.c_str());
        impl_->use_coreml_code_predictor = false;
    }

    if (impl_->state.code_pred_cache.n_ctx < 16) {
        if (!init_code_pred_kv_cache(16)) {
            return false;
        }
    }
    clear_code_pred_kv_cache();

    output.resize(15);
    std::vector<float> logits_data(cfg.code_pred_vocab_size);
    int64_t local_subseq = sampling_subseq ? *sampling_subseq : 0;
    transformer_sampling_state sampling{resolve_sampling_seed(seed), local_subseq};

    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
        return transformer_sample_top_k_p(logits_ptr, vocab_size,
                                          temperature, top_k, top_p,
                                          1.0f, nullptr, 0, sampling);
    };

    std::vector<float> cb0_embd(cfg.hidden_size);
    if (!transformer_internal::ops::lookup_single_embedding_row(*this, impl_->model.codec_embd, codebook_0_token, cb0_embd.data())) {
        return false;
    }
    std::vector<float> prefill_input((size_t) 2 * cfg.hidden_size);
    memcpy(prefill_input.data(), hidden, (size_t) cfg.hidden_size * sizeof(float));
    memcpy(prefill_input.data() + cfg.hidden_size, cb0_embd.data(),
           (size_t) cfg.hidden_size * sizeof(float));
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

        char prefill_input_name[128];
        snprintf(prefill_input_name, sizeof(prefill_input_name),
                 "frame%03d_codepred_prefill_input.f32.bin", trace_frame);
        transformer_internal::debug_trace_write_bin(trace_cfg, prefill_input_name, prefill_input.data(),
                                                    prefill_input.size(), "f32",
                                                    {2, (int64_t) cfg.hidden_size});
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_code_pred_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    auto dump_graph_tensor_f32 = [&](struct ggml_cgraph * gf,
                                     const char * tensor_name,
                                     const char * file_suffix,
                                     const std::vector<int64_t> & shape) {
        struct ggml_tensor * tensor = ggml_graph_get_tensor(gf, tensor_name);
        if (!tensor || !trace_frame_enabled) {
            return;
        }
        const size_t count = (size_t) ggml_nelements(tensor);
        std::vector<float> data(count);
        if (tensor->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(tensor, data.data(), 0, count * sizeof(float));
        } else if (tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(count);
            ggml_backend_tensor_get(tensor, tmp.data(), 0, count * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(tmp.data(), data.data(), (int64_t) count);
        } else if (tensor->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp(count);
            ggml_backend_tensor_get(tensor, tmp.data(), 0, count * sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row(tmp.data(), data.data(), (int64_t) count);
        } else {
            return;
        }
        char out_name[128];
        snprintf(out_name, sizeof(out_name), "frame%03d_%s.f32.bin", trace_frame, file_suffix);
        transformer_internal::debug_trace_write_bin(trace_cfg, out_name, data.data(), data.size(),
                                                    "f32", shape);
    };

    {
#ifdef QWEN3_TTS_TIMING
        auto t_pf_start = clk::now();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_cgraph * gf = transformer_internal::ops::build_code_pred_prefill_graph(*this);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) impl_->timing->t_code_pred_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!ggml_backend_sched_alloc_graph(impl_->state.sched, gf)) {
            error_msg_ = "Failed to allocate code predictor prefill graph";
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) impl_->timing->t_code_pred_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_tensor * inp_prefill = ggml_graph_get_tensor(gf, "inp_prefill");
        if (inp_prefill) {
            ggml_backend_tensor_set(inp_prefill, prefill_input.data(), 0,
                                    prefill_input.size() * sizeof(float));
        }

        struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
        if (inp_pos) {
            int32_t positions[2] = {0, 1};
            ggml_backend_tensor_set(inp_pos, positions, 0, 2 * sizeof(int32_t));
            if (trace_frame_enabled) {
                char pos_name[128];
                snprintf(pos_name, sizeof(pos_name),
                         "frame%03d_codepred_prefill_pos.i32.bin", trace_frame);
                transformer_internal::debug_trace_write_bin(trace_cfg, pos_name, positions, 2,
                                                            "i32", {2});
            }
        }

        struct ggml_tensor * inp_mrope_pos = ggml_graph_get_tensor(gf, "inp_mrope_pos");
        if (inp_mrope_pos && impl_->model.config.use_mrope) {
            int32_t positions[8] = {0, 1, 0, 1, 0, 1, 0, 0};
            ggml_backend_tensor_set(inp_mrope_pos, positions, 0, 8 * sizeof(int32_t));
        }

        struct ggml_tensor * inp_mask = ggml_graph_get_tensor(gf, "inp_mask");
        if (inp_mask) {
            ggml_fp16_t mask[4];
            std::fill(mask, mask + 4, ggml_fp32_to_fp16(-INFINITY));
            const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
            mask[0] = zero;
            mask[2] = zero;
            mask[3] = zero;
            ggml_backend_tensor_set(inp_mask, mask, 0, sizeof(mask));
            if (trace_frame_enabled) {
                float mask_f32[4] = {0.0f, -INFINITY, 0.0f, 0.0f};
                char mask_name[128];
                snprintf(mask_name, sizeof(mask_name),
                         "frame%03d_codepred_prefill_mask.f32.bin", trace_frame);
                transformer_internal::debug_trace_write_bin(trace_cfg, mask_name, mask_f32, 4,
                                                            "f32", {2, 2});
            }
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) impl_->timing->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (ggml_backend_sched_graph_compute(impl_->state.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute code predictor prefill graph";
            ggml_backend_sched_reset(impl_->state.sched);
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) impl_->timing->t_code_pred_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor in prefill";
            ggml_backend_sched_reset(impl_->state.sched);
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        ggml_backend_tensor_get(logits, logits_data.data(), 0,
                                cfg.code_pred_vocab_size * sizeof(float));

        dump_graph_tensor_f32(gf, "inp_prefill", "codepred_prefill_concat",
                              {2, (int64_t) cfg.hidden_size});
        dump_graph_tensor_f32(gf, "codepred_prefill_projected", "codepred_prefill_projected",
                              {2, (int64_t) cfg.code_pred_hidden_size});
        const char * sublayer_suffixes[] = {"attn_norm", "attn_out", "ffn_norm", "ffn_out"};
        for (int il = 0; il < cfg.code_pred_layers; ++il) {
            for (const char * sublayer_suffix : sublayer_suffixes) {
                char tensor_name[96];
                char file_suffix[128];
                snprintf(tensor_name, sizeof(tensor_name), "codepred_prefill_layer%02d_%s",
                         il, sublayer_suffix);
                snprintf(file_suffix, sizeof(file_suffix), "codepred_prefill_layer%02d_%s",
                         il, sublayer_suffix);
                dump_graph_tensor_f32(gf, tensor_name, file_suffix,
                                      {2, (int64_t) cfg.code_pred_hidden_size});
            }
            char tensor_name[64];
            char file_suffix[96];
            snprintf(tensor_name, sizeof(tensor_name), "codepred_prefill_layer%02d_hidden", il);
            snprintf(file_suffix, sizeof(file_suffix), "codepred_prefill_layer%02d_hidden", il);
            dump_graph_tensor_f32(gf, tensor_name, file_suffix,
                                  {2, (int64_t) cfg.code_pred_hidden_size});
        }
        dump_graph_tensor_f32(gf, "codepred_prefill_final_hidden", "codepred_prefill_final_hidden",
                              {2, (int64_t) cfg.code_pred_hidden_size});

        if (trace_frame_enabled && 0 < trace_cfg.max_code_steps) {
            char logits_name[128];
            snprintf(logits_name, sizeof(logits_name),
                     "frame%03d_codepred_logits_step00.f32.bin", trace_frame);
            transformer_internal::debug_trace_write_bin(trace_cfg, logits_name, logits_data.data(),
                                                        (size_t) cfg.code_pred_vocab_size, "f32",
                                                        {(int64_t) cfg.code_pred_vocab_size});
        }

        output[0] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);

        ggml_backend_sched_reset(impl_->state.sched);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) impl_->timing->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (impl_->timing) impl_->timing->t_code_pred_prefill_ms += std::chrono::duration<double, std::milli>(t1 - t_pf_start).count();
#endif
    }

#ifdef QWEN3_TTS_TIMING
    auto t_steps_start = clk::now();
#endif
    if (impl_->state.code_pred_mask.size() != (size_t) impl_->state.code_pred_cache.n_ctx) {
        impl_->state.code_pred_mask.resize((size_t) impl_->state.code_pred_cache.n_ctx);
    }
    std::fill(impl_->state.code_pred_mask.begin(), impl_->state.code_pred_mask.end(), ggml_fp32_to_fp16(-INFINITY));
    const ggml_fp16_t zero_fp16 = ggml_fp32_to_fp16(0.0f);
    for (int i = 0; i <= 2 && i < impl_->state.code_pred_cache.n_ctx; ++i) {
        impl_->state.code_pred_mask[(size_t) i] = zero_fp16;
    }

    for (int step = 1; step < 15; ++step) {
        int32_t n_past = step + 1;
        if (n_past < impl_->state.code_pred_cache.n_ctx) {
            impl_->state.code_pred_mask[(size_t) n_past] = zero_fp16;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_cgraph * gf = transformer_internal::ops::build_code_pred_step_graph(*this, n_past, step);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) impl_->timing->t_code_pred_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!ggml_backend_sched_alloc_graph(impl_->state.sched, gf)) {
            error_msg_ = "Failed to allocate code predictor step graph";
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) impl_->timing->t_code_pred_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
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
        if (inp_mrope_pos && impl_->model.config.use_mrope) {
            int32_t positions[4] = {n_past, n_past, n_past, 0};
            ggml_backend_tensor_set(inp_mrope_pos, positions, 0, 4 * sizeof(int32_t));
        }

        struct ggml_tensor * inp_mask = ggml_graph_get_tensor(gf, "inp_mask");
        if (inp_mask) {
            const size_t mask_size = (size_t) n_past + 1;
            ggml_backend_tensor_set(inp_mask, impl_->state.code_pred_mask.data(), 0,
                                    mask_size * sizeof(ggml_fp16_t));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) impl_->timing->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (ggml_backend_sched_graph_compute(impl_->state.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute code predictor step graph";
            ggml_backend_sched_reset(impl_->state.sched);
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) impl_->timing->t_code_pred_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor";
            ggml_backend_sched_reset(impl_->state.sched);
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        ggml_backend_tensor_get(logits, logits_data.data(), 0,
                                cfg.code_pred_vocab_size * sizeof(float));

        if (trace_frame_enabled && step < trace_cfg.max_code_steps) {
            char projected_suffix[96];
            snprintf(projected_suffix, sizeof(projected_suffix), "codepred_step%02d_projected", step);
            dump_graph_tensor_f32(gf, "codepred_step_projected", projected_suffix,
                                  {1, (int64_t) cfg.code_pred_hidden_size});
            const char * sublayer_suffixes[] = {"attn_norm", "attn_out", "ffn_norm", "ffn_out"};
            for (int il = 0; il < cfg.code_pred_layers; ++il) {
                for (const char * sublayer_suffix : sublayer_suffixes) {
                    char tensor_name[96];
                    char file_suffix[128];
                    snprintf(tensor_name, sizeof(tensor_name), "codepred_step%02d_layer%02d_%s",
                             step, il, sublayer_suffix);
                    snprintf(file_suffix, sizeof(file_suffix), "codepred_step%02d_layer%02d_%s",
                             step, il, sublayer_suffix);
                    dump_graph_tensor_f32(gf, tensor_name, file_suffix,
                                          {1, (int64_t) cfg.code_pred_hidden_size});
                }
                char tensor_name[64];
                char file_suffix[96];
                snprintf(tensor_name, sizeof(tensor_name), "codepred_step%02d_layer%02d_hidden", step, il);
                snprintf(file_suffix, sizeof(file_suffix), "codepred_step%02d_layer%02d_hidden", step, il);
                dump_graph_tensor_f32(gf, tensor_name, file_suffix,
                                      {1, (int64_t) cfg.code_pred_hidden_size});
            }
            char final_tensor_name[64];
            char final_file_suffix[96];
            snprintf(final_tensor_name, sizeof(final_tensor_name), "codepred_step%02d_final_hidden", step);
            snprintf(final_file_suffix, sizeof(final_file_suffix), "codepred_step%02d_final_hidden", step);
            dump_graph_tensor_f32(gf, final_tensor_name, final_file_suffix,
                                  {1, (int64_t) cfg.code_pred_hidden_size});

            char logits_name[128];
            snprintf(logits_name, sizeof(logits_name),
                     "frame%03d_codepred_logits_step%02d.f32.bin", trace_frame, step);
            transformer_internal::debug_trace_write_bin(trace_cfg, logits_name, logits_data.data(),
                                                        (size_t) cfg.code_pred_vocab_size, "f32",
                                                        {(int64_t) cfg.code_pred_vocab_size});
        }

        output[step] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);

        ggml_backend_sched_reset(impl_->state.sched);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) impl_->timing->t_code_pred_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif
    }
#ifdef QWEN3_TTS_TIMING
    if (impl_->timing) impl_->timing->t_code_pred_steps_ms += std::chrono::duration<double, std::milli>(clk::now() - t_steps_start).count();
#endif

    if (trace_frame_enabled) {
        char tokens_name[128];
        snprintf(tokens_name, sizeof(tokens_name),
                 "frame%03d_codepred_tokens_cb1_15.i32.bin", trace_frame);
        transformer_internal::debug_trace_write_bin(trace_cfg, tokens_name, output.data(), output.size(),
                                                    "i32", {(int64_t) output.size()});
    }

    if (sampling_subseq) {
        *sampling_subseq = sampling.subseq;
    }
    return true;
}

} // namespace qwen3_tts
