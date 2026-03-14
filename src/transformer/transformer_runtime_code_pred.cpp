#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace qwen3_tts {

namespace {

int32_t argmax_code_pred(const float * data, int32_t n) {
    int32_t max_idx = 0;
    float max_val = data[0];
    for (int32_t i = 1; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

} // namespace

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
    std::vector<float> code_probs(cfg.code_pred_vocab_size);
    std::vector<float> seq_embd((size_t) 16 * cfg.hidden_size, 0.0f);

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
        if (temperature <= 0.0f) {
            return argmax_code_pred(logits_ptr, vocab_size);
        }

        for (int32_t i = 0; i < vocab_size; ++i) {
            logits_ptr[i] /= temperature;
        }

        if (top_k > 0 && top_k < vocab_size) {
            std::vector<std::pair<float, int32_t>> scored(vocab_size);
            for (int32_t i = 0; i < vocab_size; ++i) {
                scored[i] = {logits_ptr[i], i};
            }
            std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first;
                });
            float threshold = scored[top_k - 1].first;
            for (int32_t i = 0; i < vocab_size; ++i) {
                if (logits_ptr[i] < threshold) {
                    logits_ptr[i] = -INFINITY;
                }
            }
        }

        float max_logit = *std::max_element(logits_ptr, logits_ptr + vocab_size);
        double sum = 0.0;
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = expf(logits_ptr[i] - max_logit);
            sum += code_probs[i];
        }
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = (float) (code_probs[i] / sum);
        }

        std::discrete_distribution<int32_t> dist(code_probs.begin(), code_probs.begin() + vocab_size);
        return dist(impl->rng);
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

    return true;
}

bool TTSTransformer::predict_codes_autoregressive(const float * hidden, int32_t codebook_0_token,
                                                  std::vector<int32_t> & output,
                                                  float temperature, int32_t top_k,
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
        if (transformer_internal::ops::predict_codes_autoregressive_coreml(*this, hidden, codebook_0_token, output, temperature, top_k, trace_frame)) {
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
    std::vector<float> code_probs(cfg.code_pred_vocab_size);

    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
        if (temperature <= 0.0f) {
            return argmax_code_pred(logits_ptr, vocab_size);
        }
        for (int32_t i = 0; i < vocab_size; ++i) {
            logits_ptr[i] /= temperature;
        }
        if (top_k > 0 && top_k < vocab_size) {
            std::vector<std::pair<float, int32_t>> scored(vocab_size);
            for (int32_t i = 0; i < vocab_size; ++i) {
                scored[i] = {logits_ptr[i], i};
            }
            std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first;
                });
            float threshold = scored[top_k - 1].first;
            for (int32_t i = 0; i < vocab_size; ++i) {
                if (logits_ptr[i] < threshold) {
                    logits_ptr[i] = -INFINITY;
                }
            }
        }
        float max_logit = *std::max_element(logits_ptr, logits_ptr + vocab_size);
        double sum = 0.0;
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = expf(logits_ptr[i] - max_logit);
            sum += code_probs[i];
        }
        for (int32_t i = 0; i < vocab_size; ++i) {
            code_probs[i] = (float) (code_probs[i] / sum);
        }
        std::discrete_distribution<int32_t> dist(code_probs.begin(), code_probs.begin() + vocab_size);
        return dist(impl_->rng);
    };

    std::vector<float> cb0_embd(cfg.hidden_size);
    if (!transformer_internal::ops::lookup_single_embedding_row(*this, impl_->model.codec_embd, codebook_0_token, cb0_embd.data())) {
        return false;
    }
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
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_code_pred_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

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
        struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
        if (inp_hidden) {
            ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
        }

        struct ggml_tensor * inp_cb0_embd = ggml_graph_get_tensor(gf, "inp_cb0_embd");
        if (inp_cb0_embd) {
            ggml_backend_tensor_set(inp_cb0_embd, cb0_embd.data(), 0, cfg.hidden_size * sizeof(float));
        }

        struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
        if (inp_pos) {
            int32_t positions[2] = {0, 1};
            ggml_backend_tensor_set(inp_pos, positions, 0, 2 * sizeof(int32_t));
        }

        struct ggml_tensor * inp_mrope_pos = ggml_graph_get_tensor(gf, "inp_mrope_pos");
        if (inp_mrope_pos && impl_->model.config.use_mrope) {
            int32_t positions[8] = {0, 1, 0, 1, 0, 1, 0, 0};
            ggml_backend_tensor_set(inp_mrope_pos, positions, 0, 8 * sizeof(int32_t));
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
            ggml_backend_tensor_set(inp_mask, impl_->state.code_pred_mask.data(), 0,
                                    impl_->state.code_pred_cache.n_ctx * sizeof(ggml_fp16_t));
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

    return true;
}

} // namespace qwen3_tts
