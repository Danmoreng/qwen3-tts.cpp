#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"
#include "transformer/transformer_sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace qwen3_tts {

namespace {

bool ensure_code_pred_static_masks(tts_transformer_state & state) {
    const int32_t n_ctx = state.code_pred_cache.n_ctx;
    if (n_ctx <= 0) {
        return false;
    }

    const size_t prefill_size = (size_t) n_ctx * 2;
    const size_t step_size = (size_t) n_ctx * 15;
    if (state.code_pred_static_mask_n_ctx == n_ctx &&
        state.code_pred_prefill_mask.size() == prefill_size &&
        state.code_pred_step_masks.size() == step_size) {
        return true;
    }

    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);

    state.code_pred_prefill_mask.assign(prefill_size, neg_inf);
    state.code_pred_prefill_mask[0] = zero;
    state.code_pred_prefill_mask[(size_t) n_ctx] = zero;
    state.code_pred_prefill_mask[(size_t) n_ctx + 1] = zero;

    state.code_pred_step_masks.assign(step_size, neg_inf);
    for (int step = 1; step < 15; ++step) {
        ggml_fp16_t * row = state.code_pred_step_masks.data() + (size_t) step * n_ctx;
        const int32_t max_unmasked = std::min<int32_t>(n_ctx - 1, step + 1);
        for (int32_t i = 0; i <= max_unmasked; ++i) {
            row[i] = zero;
        }
    }

    state.code_pred_static_mask_n_ctx = n_ctx;
    return true;
}

bool env_flag_value(const char * name) {
#if defined(_MSC_VER)
    char * value = nullptr;
    size_t value_len = 0;
    if (_dupenv_s(&value, &value_len, name) != 0 || !value) {
        return false;
    }
    const bool enabled = value[0] != '\0' && value[0] != '0';
    std::free(value);
    return enabled;
#else
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && value[0] != '0';
#endif
}

bool env_flag_disabled(const char * name) {
#if defined(_MSC_VER)
    char * value = nullptr;
    size_t value_len = 0;
    if (_dupenv_s(&value, &value_len, name) != 0 || !value) {
        return false;
    }
    const bool disabled = value[0] == '0';
    std::free(value);
    return disabled;
#else
    const char * value = std::getenv(name);
    return value && value[0] == '0';
#endif
}

bool graph_stats_enabled() {
    static const bool enabled = env_flag_value("QWEN3_TTS_GRAPH_STATS");
    return enabled;
}

bool code_pred_replay_enabled() {
    static const bool enabled = !env_flag_disabled("QWEN3_TTS_CODE_PRED_REPLAY_GRAPHS");
    return enabled;
}

void maybe_log_code_pred_graph_stats(tts_transformer_private & impl, ggml_cgraph * gf,
                                     int32_t graph_index, const char * label) {
    if (!gf || graph_index < 0 || graph_index >= 15) {
        return;
    }
    if (!graph_stats_enabled()) {
        return;
    }

    if (impl.state.code_pred_graph_stats_logged.size() != 15) {
        impl.state.code_pred_graph_stats_logged.assign(15, 0);
    }
    if (impl.state.code_pred_graph_stats_logged[(size_t) graph_index]) {
        return;
    }
    impl.state.code_pred_graph_stats_logged[(size_t) graph_index] = 1;

    fprintf(stderr,
            "  CodePred graph stats [%s]: nodes=%d graph_size=%d capacity=%d\n",
            label, ggml_graph_n_nodes(gf), ggml_graph_size(gf), QWEN3_TTS_CODE_PRED_MAX_NODES);
}

bool reserve_code_pred_dedicated_schedulers(TTSTransformer & self, tts_transformer_private & impl,
                                            bool use_hidden_bridge) {
    auto & state = impl.state;
    if (!state.code_pred_prefill_sched || !state.code_pred_step_sched) {
        return false;
    }
    if (state.code_pred_sched_reserved) {
        return true;
    }
    if (state.code_pred_sched_reserve_failed) {
        return false;
    }

    struct ggml_cgraph * prefill_graph =
        transformer_internal::ops::build_code_pred_prefill_graph(self, use_hidden_bridge);
    const bool prefill_ok =
        prefill_graph && ggml_backend_sched_reserve(state.code_pred_prefill_sched, prefill_graph);
    ggml_backend_sched_reset(state.code_pred_prefill_sched);

    struct ggml_cgraph * step_graph =
        transformer_internal::ops::build_code_pred_step_graph(self, 15, 14);
    const bool step_ok =
        step_graph && ggml_backend_sched_reserve(state.code_pred_step_sched, step_graph);
    ggml_backend_sched_reset(state.code_pred_step_sched);

    if (!prefill_ok || !step_ok) {
        state.code_pred_sched_reserved = false;
        state.code_pred_sched_reserve_failed = true;
        fprintf(stderr,
                "  CodePred dedicated scheduler reserve failed; falling back to the main scheduler\n");
        return false;
    }

    state.code_pred_sched_reserved = true;
    fprintf(stderr, "  CodePred dedicated schedulers: reserved\n");
    return true;
}

bool ensure_code_pred_replay_graphs(TTSTransformer & self, tts_transformer_private & impl,
                                    bool use_hidden_bridge) {
    auto & state = impl.state;
    if (!code_pred_replay_enabled()) {
        return false;
    }
    if (!use_hidden_bridge) {
        return false;
    }
    if (state.code_pred_replay_ready) {
        const code_pred_graph_mode requested_mode = state.code_pred_device_chain_active
            ? code_pred_graph_mode::device_chain_replay
            : code_pred_graph_mode::legacy_replay;
        if (state.code_pred_mode == requested_mode) {
            return true;
        }

        // The host-token and device-token graphs have different inputs and
        // outputs. A resident process may cross the automatic dispatch
        // threshold between requests, so rebuild instead of replaying a graph
        // created for the other contract.
        for (ggml_backend_sched_t sched : state.code_pred_replay_scheds) {
            if (sched) {
                ggml_backend_sched_reset(sched);
            }
        }
        state.code_pred_replay_ready = false;
        state.code_pred_replay_graphs.clear();
        state.code_pred_mode = code_pred_graph_mode::none;
    }
    if (state.code_pred_replay_failed) {
        return false;
    }
    if (!state.backend) {
        return false;
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(state.backend);
    if (state.backend_cpu) {
        backends.push_back(state.backend_cpu);
    }

    if (state.code_pred_replay_scheds.empty()) {
        state.code_pred_replay_scheds.assign(15, nullptr);
        for (int i = 0; i < 15; ++i) {
            state.code_pred_replay_scheds[(size_t) i] = ggml_backend_sched_new(
                backends.data(), nullptr, (int) backends.size(), QWEN3_TTS_CODE_PRED_MAX_NODES, false, true);
            if (!state.code_pred_replay_scheds[(size_t) i]) {
                state.code_pred_replay_failed = true;
                fprintf(stderr,
                        "  CodePred replay graph schedulers failed to initialize; using dynamic graphs\n");
                return false;
            }
        }
    }

    state.code_pred_replay_graphs.assign(15, nullptr);
    state.code_pred_replay_graphs[0] =
        transformer_internal::ops::build_code_pred_prefill_graph(self, true);
    if (!state.code_pred_replay_graphs[0]) {
        state.code_pred_replay_failed = true;
        return false;
    }

    for (int step = 1; step < 15; ++step) {
        state.code_pred_replay_graphs[(size_t) step] =
            transformer_internal::ops::build_code_pred_step_graph(self, 15, step);
        if (!state.code_pred_replay_graphs[(size_t) step]) {
            state.code_pred_replay_failed = true;
            return false;
        }
    }

    for (int i = 0; i < 15; ++i) {
        if (!ggml_backend_sched_alloc_graph(state.code_pred_replay_scheds[(size_t) i],
                                            state.code_pred_replay_graphs[(size_t) i])) {
            for (ggml_backend_sched_t sched : state.code_pred_replay_scheds) {
                if (sched) {
                    ggml_backend_sched_reset(sched);
                }
            }
            state.code_pred_replay_failed = true;
            fprintf(stderr,
                    "  CodePred replay graph allocation failed; using dynamic graphs\n");
            return false;
        }
    }

    state.code_pred_replay_ready = true;
    state.code_pred_mode = state.code_pred_device_chain_active
        ? code_pred_graph_mode::device_chain_replay
        : code_pred_graph_mode::legacy_replay;
    fprintf(stderr, "  CodePred replay graphs: enabled (default)\n");
    return true;
}

bool ensure_code_pred_supergraph(TTSTransformer & self, tts_transformer_private & impl) {
    auto & state = impl.state;
    if (!code_pred_replay_enabled() || !state.code_pred_supergraph_requested ||
        state.code_pred_supergraph_failed || !state.backend ||
        !state.hidden_bridge || !state.code_pred_tokens_bridge) {
        return false;
    }
    if (state.code_pred_supergraph_ready) {
        state.code_pred_replay_ready = true;
        state.code_pred_mode = code_pred_graph_mode::supergraph;
        return true;
    }

    for (ggml_backend_sched_t sched : state.code_pred_replay_scheds) {
        if (sched) {
            ggml_backend_sched_reset(sched);
        }
    }
    state.code_pred_replay_ready = false;
    state.code_pred_replay_graphs.clear();
    state.code_pred_mode = code_pred_graph_mode::none;

    std::vector<ggml_backend_t> backends;
    backends.push_back(state.backend);
    if (state.backend_cpu) {
        backends.push_back(state.backend_cpu);
    }
    if (!state.code_pred_supergraph_sched) {
        state.code_pred_supergraph_sched = ggml_backend_sched_new(
            backends.data(), nullptr, (int) backends.size(),
            QWEN3_TTS_CODE_PRED_MAX_NODES, false, true);
        if (!state.code_pred_supergraph_sched) {
            state.code_pred_supergraph_failed = true;
            fprintf(stderr, "  CodePred supergraph scheduler initialization failed; using established path\n");
            return false;
        }
    }

    const auto build_start = std::chrono::high_resolution_clock::now();
    if (!state.code_pred_supergraph) {
        state.code_pred_supergraph = transformer_internal::ops::build_code_pred_supergraph(self);
    }
    const auto build_end = std::chrono::high_resolution_clock::now();
    if (!state.code_pred_supergraph) {
        state.code_pred_supergraph_failed = true;
        fprintf(stderr, "  CodePred supergraph construction failed; using established path\n");
        return false;
    }

    ggml_backend_sched_t sched = state.code_pred_supergraph_sched;
    const auto alloc_start = std::chrono::high_resolution_clock::now();
    const bool allocated = ggml_backend_sched_alloc_graph(sched, state.code_pred_supergraph);
    const auto alloc_end = std::chrono::high_resolution_clock::now();
    if (!allocated || ggml_backend_sched_get_n_splits(sched) != 1) {
        ggml_backend_sched_reset(sched);
        state.code_pred_supergraph_failed = true;
        fprintf(stderr, "  CodePred supergraph allocation/single-backend validation failed; using established path\n");
        return false;
    }

    state.code_pred_replay_ready = true;
    state.code_pred_supergraph_ready = true;
    state.code_pred_mode = code_pred_graph_mode::supergraph;
    const double build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    const double alloc_ms = std::chrono::duration<double, std::milli>(alloc_end - alloc_start).count();
    const double scratch_mib = (double) ggml_backend_sched_get_buffer_size(sched, state.backend) /
                               (1024.0 * 1024.0);
    fprintf(stderr,
            "  CodePred supergraph: ready (nodes=%d, build=%.2f ms, alloc=%.2f ms, scratch=%.1f MiB)\n",
            ggml_graph_n_nodes(state.code_pred_supergraph), build_ms, alloc_ms, scratch_mib);
    return true;
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
#ifdef QWEN3_TTS_TIMING
        auto t_sample_start = clk::now();
#endif
        const int32_t token = transformer_sample_top_k_p(logits_ptr, vocab_size,
                                                         temperature, top_k, top_p,
                                                         1.0f, nullptr, 0, sampling);
#ifdef QWEN3_TTS_TIMING
        auto t_sample_end = clk::now();
        if (impl->timing) {
            impl->timing->t_code_pred_sampling_ms +=
                std::chrono::duration<double, std::milli>(t_sample_end - t_sample_start).count();
        }
#endif
        return token;
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

    if (hidden && impl_->use_coreml_code_predictor && impl_->coreml_code_predictor.is_loaded()) {
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
    const bool use_hidden_bridge = hidden == nullptr && impl_->state.hidden_bridge != nullptr;
    if (!use_hidden_bridge && hidden == nullptr) {
        error_msg_ = "Code predictor requires hidden input or a device hidden bridge";
        return false;
    }

    const bool request_supergraph =
        impl_->state.code_pred_supergraph_requested &&
        temperature == 0.0f && use_hidden_bridge && !trace_frame_enabled;
    impl_->state.code_pred_device_chain_active = false;
    if (request_supergraph && ensure_code_pred_supergraph(*this, *impl_)) {
        impl_->state.code_pred_supergraph_active = true;
        if (!impl_->state.code_pred_supergraph_logged) {
            fprintf(stderr, "  CodePred supergraph: enabled (greedy CUDA path)\n");
            impl_->state.code_pred_supergraph_logged = true;
        }

        struct ggml_cgraph * gf = impl_->state.code_pred_supergraph;
        ggml_backend_sched_t sched = impl_->state.code_pred_supergraph_sched;
        struct ggml_tensor * inp_cb0_code = ggml_graph_get_tensor(gf, "inp_cb0_code");
        if (!inp_cb0_code) {
            error_msg_ = "Failed to find code predictor supergraph input";
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        ggml_backend_tensor_set(inp_cb0_code, &codebook_0_token, 0, sizeof(int32_t));
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_data_ms += dt_ms;
            impl_->timing->t_code_pred_input_upload_ms += dt_ms;
        }
        t0 = clk::now();
#endif
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            impl_->state.code_pred_replay_ready = false;
            impl_->state.code_pred_supergraph_failed = true;
            impl_->state.code_pred_supergraph_ready = false;
            impl_->state.code_pred_mode = code_pred_graph_mode::none;
            ggml_backend_sched_reset(sched);
            error_msg_ = "Failed to compute code predictor supergraph";
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) {
            impl_->timing->t_code_pred_compute_ms +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        t0 = clk::now();
#endif
        output.resize(15);
        ggml_backend_tensor_get(impl_->state.code_pred_tokens_bridge, output.data(), 0,
                                output.size() * sizeof(int32_t));
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_data_ms += dt_ms;
            impl_->timing->t_code_pred_logits_read_ms += dt_ms;
        }
#endif
        for (int32_t token : output) {
            if (token < 0 || token >= cfg.code_pred_vocab_size) {
                error_msg_ = "Code predictor supergraph returned an invalid token";
                return false;
            }
        }
        // Greedy decoding consumes no random subsequences.
        return true;
    }
    impl_->state.code_pred_supergraph_active = false;
    if (!request_supergraph) {
        impl_->state.code_pred_supergraph_logged = false;
    }

    const bool use_device_chain = impl_->state.code_pred_device_chain_requested &&
        temperature == 0.0f && impl_->state.code_pred_tokens_bridge;
    impl_->state.code_pred_device_chain_active = use_device_chain;
    if (use_device_chain && !impl_->state.code_pred_device_chain_logged) {
        fprintf(stderr, "  CodePred device chain: enabled (greedy CUDA path)\n");
        impl_->state.code_pred_device_chain_logged = true;
    } else if (!use_device_chain) {
        impl_->state.code_pred_device_chain_logged = false;
    }
    const bool use_replay_graphs =
        ensure_code_pred_replay_graphs(*this, *impl_, use_hidden_bridge);
    const bool use_dedicated_schedulers =
        !use_replay_graphs && reserve_code_pred_dedicated_schedulers(*this, *impl_, use_hidden_bridge);
    ggml_backend_sched_t code_pred_prefill_sched = use_replay_graphs
        ? impl_->state.code_pred_replay_scheds[0]
        : (use_dedicated_schedulers ? impl_->state.code_pred_prefill_sched : impl_->state.sched);
    ggml_backend_sched_t code_pred_step_sched = use_dedicated_schedulers
        ? impl_->state.code_pred_step_sched
        : impl_->state.sched;

    output.resize(15);
    std::vector<float> logits_data(cfg.code_pred_vocab_size);
    int64_t local_subseq = sampling_subseq ? *sampling_subseq : 0;
    transformer_sampling_state sampling{resolve_sampling_seed(seed), local_subseq};

    auto sample_or_argmax = [&](float * logits_ptr, int32_t vocab_size) -> int32_t {
#ifdef QWEN3_TTS_TIMING
        auto t_sample_start = clk::now();
#endif
        const int32_t token = transformer_sample_top_k_p(logits_ptr, vocab_size,
                                                         temperature, top_k, top_p,
                                                         1.0f, nullptr, 0, sampling);
#ifdef QWEN3_TTS_TIMING
        auto t_sample_end = clk::now();
        if (impl_->timing) {
            impl_->timing->t_code_pred_sampling_ms +=
                std::chrono::duration<double, std::milli>(t_sample_end - t_sample_start).count();
        }
#endif
        return token;
    };

    if (trace_frame_enabled && hidden) {
        char hidden_name[128];
        snprintf(hidden_name, sizeof(hidden_name),
                 "frame%03d_codepred_input_hidden.f32.bin", trace_frame);
        transformer_internal::debug_trace_write_bin(trace_cfg, hidden_name, hidden,
                                                    (size_t) cfg.hidden_size, "f32",
                                                    {(int64_t) cfg.hidden_size});

        std::vector<float> cb0_embd(cfg.hidden_size);
        if (!transformer_internal::ops::lookup_single_embedding_row(*this, impl_->model.codec_embd,
                                                                    codebook_0_token, cb0_embd.data())) {
            return false;
        }
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
        struct ggml_cgraph * gf = use_replay_graphs
            ? impl_->state.code_pred_replay_graphs[0]
            : transformer_internal::ops::build_code_pred_prefill_graph(*this, use_hidden_bridge);
        maybe_log_code_pred_graph_stats(*impl_, gf, 0, "prefill");
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing && !use_replay_graphs) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_graph_build_ms += dt_ms;
            impl_->timing->t_code_pred_prefill_graph_build_ms += dt_ms;
        }
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!use_replay_graphs) {
            if (!ggml_backend_sched_alloc_graph(code_pred_prefill_sched, gf)) {
                error_msg_ = "Failed to allocate code predictor prefill graph";
                return false;
            }
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing && !use_replay_graphs) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_graph_alloc_ms += dt_ms;
            impl_->timing->t_code_pred_prefill_graph_alloc_ms += dt_ms;
        }
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
        if (inp_hidden && hidden) {
            ggml_backend_tensor_set(inp_hidden, hidden, 0, cfg.hidden_size * sizeof(float));
        }

        struct ggml_tensor * inp_cb0_code = ggml_graph_get_tensor(gf, "inp_cb0_code");
        if (inp_cb0_code) {
            ggml_backend_tensor_set(inp_cb0_code, &codebook_0_token, 0, sizeof(int32_t));
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

        struct ggml_tensor * inp_mask = ggml_graph_get_tensor(gf, "inp_mask");
        if (inp_mask) {
            if (!ensure_code_pred_static_masks(impl_->state)) {
                error_msg_ = "Failed to initialize code predictor masks";
                if (!use_replay_graphs) {
                    ggml_backend_sched_reset(code_pred_prefill_sched);
                }
                return false;
            }
            ggml_backend_tensor_set(inp_mask, impl_->state.code_pred_prefill_mask.data(), 0,
                                    impl_->state.code_pred_prefill_mask.size() * sizeof(ggml_fp16_t));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_data_ms += dt_ms;
            impl_->timing->t_code_pred_prefill_data_ms += dt_ms;
            impl_->timing->t_code_pred_input_upload_ms += dt_ms;
        }
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (ggml_backend_sched_graph_compute(code_pred_prefill_sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute code predictor prefill graph";
            if (use_replay_graphs) {
                impl_->state.code_pred_replay_ready = false;
                impl_->state.code_pred_replay_failed = true;
            }
            ggml_backend_sched_reset(code_pred_prefill_sched);
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_compute_ms += dt_ms;
            impl_->timing->t_code_pred_prefill_compute_ms += dt_ms;
        }
#endif

        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor in prefill";
            if (!use_replay_graphs) {
                ggml_backend_sched_reset(code_pred_prefill_sched);
            }
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!use_device_chain || trace_frame_enabled) {
            ggml_backend_tensor_get(logits, logits_data.data(), 0,
                                    cfg.code_pred_vocab_size * sizeof(float));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing && (!use_device_chain || trace_frame_enabled)) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_data_ms += dt_ms;
            impl_->timing->t_code_pred_prefill_data_ms += dt_ms;
            impl_->timing->t_code_pred_logits_read_ms += dt_ms;
        }
#endif

        if (trace_frame_enabled && 0 < trace_cfg.max_code_steps) {
            char logits_name[128];
            snprintf(logits_name, sizeof(logits_name),
                     "frame%03d_codepred_logits_step00.f32.bin", trace_frame);
            transformer_internal::debug_trace_write_bin(trace_cfg, logits_name, logits_data.data(),
                                                        (size_t) cfg.code_pred_vocab_size, "f32",
                                                        {(int64_t) cfg.code_pred_vocab_size});
        }

        if (!use_device_chain) {
            output[0] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!use_replay_graphs) {
            ggml_backend_sched_reset(code_pred_prefill_sched);
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing && !use_replay_graphs) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_data_ms += dt_ms;
            impl_->timing->t_code_pred_prefill_data_ms += dt_ms;
            impl_->timing->t_code_pred_sched_reset_ms += dt_ms;
        }
        if (impl_->timing) impl_->timing->t_code_pred_prefill_ms += std::chrono::duration<double, std::milli>(t1 - t_pf_start).count();
#endif
    }

#ifdef QWEN3_TTS_TIMING
    auto t_steps_start = clk::now();
#endif
    for (int step = 1; step < 15; ++step) {
        int32_t n_past = step + 1;

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_cgraph * gf = use_replay_graphs
            ? impl_->state.code_pred_replay_graphs[(size_t) step]
            : transformer_internal::ops::build_code_pred_step_graph(*this, n_past, step);
        char graph_label[16];
        snprintf(graph_label, sizeof(graph_label), "step%02d", step);
        maybe_log_code_pred_graph_stats(*impl_, gf, step, graph_label);
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing && !use_replay_graphs) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_graph_build_ms += dt_ms;
            impl_->timing->t_code_pred_steps_graph_build_ms += dt_ms;
        }
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        ggml_backend_sched_t step_sched = use_replay_graphs
            ? impl_->state.code_pred_replay_scheds[(size_t) step]
            : code_pred_step_sched;
        if (!use_replay_graphs) {
            if (!ggml_backend_sched_alloc_graph(step_sched, gf)) {
                error_msg_ = "Failed to allocate code predictor step graph";
                return false;
            }
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing && !use_replay_graphs) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_graph_alloc_ms += dt_ms;
            impl_->timing->t_code_pred_steps_graph_alloc_ms += dt_ms;
        }
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        struct ggml_tensor * inp_hidden = ggml_graph_get_tensor(gf, "inp_hidden");
        if (inp_hidden && hidden) {
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
            const size_t n_ctx = (size_t) impl_->state.code_pred_cache.n_ctx;
            const ggml_fp16_t * mask = impl_->state.code_pred_step_masks.data() + (size_t) step * n_ctx;
            ggml_backend_tensor_set(inp_mask, mask, 0, n_ctx * sizeof(ggml_fp16_t));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_data_ms += dt_ms;
            impl_->timing->t_code_pred_steps_data_ms += dt_ms;
            impl_->timing->t_code_pred_input_upload_ms += dt_ms;
        }
#endif

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (ggml_backend_sched_graph_compute(step_sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute code predictor step graph";
            if (use_replay_graphs) {
                impl_->state.code_pred_replay_ready = false;
                impl_->state.code_pred_replay_failed = true;
            }
            ggml_backend_sched_reset(step_sched);
            return false;
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_compute_ms += dt_ms;
            impl_->timing->t_code_pred_steps_compute_ms += dt_ms;
        }
#endif

        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor";
            if (!use_replay_graphs) {
                ggml_backend_sched_reset(step_sched);
            }
            return false;
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!use_device_chain || trace_frame_enabled) {
            ggml_backend_tensor_get(logits, logits_data.data(), 0,
                                    cfg.code_pred_vocab_size * sizeof(float));
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing && (!use_device_chain || trace_frame_enabled)) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_data_ms += dt_ms;
            impl_->timing->t_code_pred_steps_data_ms += dt_ms;
            impl_->timing->t_code_pred_logits_read_ms += dt_ms;
        }
#endif

        if (trace_frame_enabled && step < trace_cfg.max_code_steps) {
            char logits_name[128];
            snprintf(logits_name, sizeof(logits_name),
                     "frame%03d_codepred_logits_step%02d.f32.bin", trace_frame, step);
            transformer_internal::debug_trace_write_bin(trace_cfg, logits_name, logits_data.data(),
                                                        (size_t) cfg.code_pred_vocab_size, "f32",
                                                        {(int64_t) cfg.code_pred_vocab_size});
        }

        if (!use_device_chain) {
            output[step] = sample_or_argmax(logits_data.data(), cfg.code_pred_vocab_size);
        }

#ifdef QWEN3_TTS_TIMING
        t0 = clk::now();
#endif
        if (!use_replay_graphs) {
            ggml_backend_sched_reset(step_sched);
        }
#ifdef QWEN3_TTS_TIMING
        t1 = clk::now();
        if (impl_->timing && !use_replay_graphs) {
            const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            impl_->timing->t_code_pred_data_ms += dt_ms;
            impl_->timing->t_code_pred_steps_data_ms += dt_ms;
            impl_->timing->t_code_pred_sched_reset_ms += dt_ms;
        }
#endif
    }

    if (use_device_chain) {
        ggml_backend_tensor_get(impl_->state.code_pred_tokens_bridge, output.data(), 0,
                                output.size() * sizeof(int32_t));
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
