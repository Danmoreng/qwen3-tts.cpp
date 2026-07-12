#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace qwen3_tts {

namespace {

bool talker_replay_enabled() {
    const char * value = std::getenv("QWEN3_TTS_TALKER_REPLAY_GRAPHS");
    return !value || value[0] != '0';
}

int32_t talker_step_n_kv_pad(const tts_transformer_state & state, int32_t n_past) {
    return std::min<int32_t>(state.cache.n_ctx, GGML_PAD(n_past + 1, 256));
}

bool ensure_talker_replay_graph(TTSTransformer & self, tts_transformer_private & impl,
                                int32_t n_past, bool use_frame_codes,
                                bool read_hidden, bool has_hidden_out) {
    auto & state = impl.state;
    if (!talker_replay_enabled()) {
        return false;
    }
    if (!use_frame_codes || read_hidden || has_hidden_out) {
        return false;
    }
    if (state.talker_replay_failed || !state.backend) {
        return false;
    }

    const int32_t n_kv_pad = talker_step_n_kv_pad(state, n_past);
    if (state.talker_replay_ready && state.talker_replay_n_kv_pad == n_kv_pad) {
        return true;
    }

    if (state.talker_replay_sched) {
        ggml_backend_sched_reset(state.talker_replay_sched);
    } else {
        std::vector<ggml_backend_t> backends;
        backends.push_back(state.backend);
        if (state.backend_cpu) {
            backends.push_back(state.backend_cpu);
        }
        state.talker_replay_sched = ggml_backend_sched_new(
            backends.data(), nullptr, (int) backends.size(), QWEN3_TTS_MAX_NODES, false, true);
        if (!state.talker_replay_sched) {
            state.talker_replay_failed = true;
            fprintf(stderr, "  Talker replay graph scheduler failed to initialize; using dynamic graphs\n");
            return false;
        }
    }

    state.talker_replay_graph = transformer_internal::ops::build_step_graph(
        self, n_past, true, &state.talker_replay_compute_meta);
    if (!state.talker_replay_graph) {
        state.talker_replay_failed = true;
        return false;
    }

    if (!ggml_backend_sched_alloc_graph(state.talker_replay_sched, state.talker_replay_graph)) {
        ggml_backend_sched_reset(state.talker_replay_sched);
        state.talker_replay_ready = false;
        state.talker_replay_failed = true;
        fprintf(stderr, "  Talker replay graph allocation failed; using dynamic graphs\n");
        return false;
    }

    state.talker_replay_ready = true;
    state.talker_replay_n_kv_pad = n_kv_pad;
    fprintf(stderr, "  Talker replay graph: enabled for %d-token bucket (experimental)\n", n_kv_pad);
    return true;
}

} // namespace

bool TTSTransformer::forward_prefill(const float * prefill_embd, int32_t n_tokens,
                                     int32_t n_past, std::vector<float> & output,
                                     std::vector<float> * logits_out) {
    if (!impl_->model.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!prefill_embd) {
        error_msg_ = "prefill_embd is null";
        return false;
    }
    if (n_tokens <= 0) {
        error_msg_ = "n_tokens must be > 0";
        return false;
    }

    if (impl_->state.cache.n_ctx == 0) {
        const int32_t min_ctx = std::max<int32_t>(256, n_past + n_tokens + 16);
        if (!init_kv_cache(min_ctx)) {
            return false;
        }
    }

    if (n_past + n_tokens > impl_->state.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_cgraph * gf = transformer_internal::ops::build_prefill_forward_graph(*this, n_tokens, n_past);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_prefill_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!ggml_backend_sched_alloc_graph(impl_->state.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_prefill_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_tensor * inp_prefill = ggml_graph_get_tensor(gf, "inp_prefill_embd");
    if (inp_prefill) {
        ggml_backend_tensor_set(inp_prefill, prefill_embd, 0,
                                (size_t) n_tokens * impl_->model.config.hidden_size * sizeof(float));
    }

    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    if (inp_pos) {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            positions[i] = n_past + i;
        }
        size_t write_size = positions.size() * sizeof(int32_t);
        if (write_size > ggml_nbytes(inp_pos)) {
            error_msg_ = "Talker position input is smaller than the requested prefill write";
            ggml_backend_sched_reset(impl_->state.sched);
            return false;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, write_size);
    }

    struct ggml_tensor * inp_mrope_pos = ggml_graph_get_tensor(gf, "inp_mrope_pos");
    if (inp_mrope_pos && impl_->model.config.use_mrope) {
        std::vector<int32_t> positions(n_tokens * 4);
        for (int i = 0; i < n_tokens; ++i) {
            int32_t p = n_past + i;
            positions[i + n_tokens * 0] = p;
            positions[i + n_tokens * 1] = p;
            positions[i + n_tokens * 2] = p;
            positions[i + n_tokens * 3] = 0;
        }
        size_t write_size = positions.size() * sizeof(int32_t);
        if (write_size > ggml_nbytes(inp_mrope_pos)) {
            error_msg_ = "Talker M-RoPE input is smaller than the requested prefill write";
            ggml_backend_sched_reset(impl_->state.sched);
            return false;
        }
        ggml_backend_tensor_set(inp_mrope_pos, positions.data(), 0, write_size);
    }

    struct ggml_tensor * inp_mask = ggml_graph_get_tensor(gf, "inp_mask");
    if (inp_mask) {
        const int32_t n_kv_pad = (int32_t) inp_mask->ne[0];
        const size_t mask_size = (size_t) n_tokens * (size_t) n_kv_pad;
        auto & mask = impl_->state.talker_mask;
        if (mask.size() != mask_size) {
            mask.resize(mask_size);
        }
        const ggml_fp16_t zero_fp16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf_fp16 = ggml_fp32_to_fp16(-INFINITY);
        for (size_t i = 0; i < mask.size(); ++i) {
            mask[i] = neg_inf_fp16;
        }
        for (int32_t q = 0; q < n_tokens; ++q) {
            const int32_t q_pos = n_past + q;
            const int32_t last_k = std::min<int32_t>(q_pos, n_kv_pad - 1);
            ggml_fp16_t * row = mask.data() + (size_t) q * (size_t) n_kv_pad;
            for (int32_t k = 0; k <= last_k; ++k) {
                row[k] = zero_fp16;
            }
        }
        ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_prefill_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (ggml_backend_sched_graph_compute(impl_->state.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(impl_->state.sched);
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_prefill_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");
    if (!hidden) {
        error_msg_ = "Failed to find hidden_states tensor";
        ggml_backend_sched_reset(impl_->state.sched);
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    output.resize(n_tokens * impl_->model.config.hidden_size);
    ggml_backend_tensor_get(hidden, output.data(), 0, output.size() * sizeof(float));

    last_hidden_.resize(impl_->model.config.hidden_size);
    ggml_backend_tensor_get(hidden, last_hidden_.data(),
                            (n_tokens - 1) * impl_->model.config.hidden_size * sizeof(float),
                            impl_->model.config.hidden_size * sizeof(float));

    if (logits_out) {
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        if (!logits) {
            error_msg_ = "Failed to find logits tensor";
            ggml_backend_sched_reset(impl_->state.sched);
            return false;
        }

        logits_out->resize(impl_->model.config.codec_vocab_size);
        ggml_backend_tensor_get(logits, logits_out->data(),
                                (n_tokens - 1) * impl_->model.config.codec_vocab_size * sizeof(float),
                                impl_->model.config.codec_vocab_size * sizeof(float));
    }

    impl_->state.cache.n_used = n_past + n_tokens;

    ggml_backend_sched_reset(impl_->state.sched);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_prefill_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    return true;
}

bool TTSTransformer::forward_text(const int32_t * text_tokens, int32_t n_tokens,
                                  const float * speaker_embd, int32_t n_past,
                                  std::vector<float> & output) {
    if (!text_tokens) {
        error_msg_ = "text_tokens is null";
        return false;
    }
    if (n_tokens <= 0) {
        error_msg_ = "n_tokens must be > 0";
        return false;
    }

    std::vector<float> projected;
    if (!transformer_internal::ops::project_text_tokens(*this, text_tokens, n_tokens, projected)) {
        return false;
    }

    if (speaker_embd) {
        const int32_t hidden_size = impl_->model.config.hidden_size;
        for (int32_t t = 0; t < n_tokens; ++t) {
            float * row = projected.data() + (size_t) t * hidden_size;
            for (int32_t h = 0; h < hidden_size; ++h) {
                row[h] += speaker_embd[h];
            }
        }
    }

    return forward_prefill(projected.data(), n_tokens, n_past, output, nullptr);
}

bool TTSTransformer::forward_step_internal(const float * step_embd,
                                           const int32_t * frame_codes,
                                           const float * overlay_embd,
                                           int32_t n_past,
                                           std::vector<float> & output,
                                           std::vector<float> * hidden_out,
                                           bool read_hidden) {
    if (!impl_->model.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    const bool use_frame_codes = frame_codes != nullptr && overlay_embd != nullptr;
    if (!step_embd && !use_frame_codes) {
        error_msg_ = "step_embd is null";
        return false;
    }

    if (impl_->state.cache.n_ctx == 0) {
        const int32_t min_ctx = std::max<int32_t>(256, n_past + 1 + 16);
        if (!init_kv_cache(min_ctx)) {
            return false;
        }
    }

    if (n_past + 1 > impl_->state.cache.n_ctx) {
        error_msg_ = "Context length exceeded";
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now(), t1 = t0;
#endif

    const bool use_talker_replay =
        ensure_talker_replay_graph(*this, *impl_, n_past, use_frame_codes,
                                   read_hidden, hidden_out != nullptr);
    ggml_backend_sched_t step_sched = use_talker_replay
        ? impl_->state.talker_replay_sched
        : impl_->state.sched;

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_cgraph * gf = use_talker_replay
        ? impl_->state.talker_replay_graph
        : transformer_internal::ops::build_step_graph(*this, n_past, use_frame_codes);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing && !use_talker_replay) {
        impl_->timing->t_talker_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!use_talker_replay) {
        if (!ggml_backend_sched_alloc_graph(step_sched, gf)) {
            error_msg_ = "Failed to allocate graph";
            return false;
        }
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing && !use_talker_replay) {
        impl_->timing->t_talker_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (use_frame_codes) {
        struct ggml_tensor * inp_frame_codes = ggml_graph_get_tensor(gf, "inp_frame_codes");
        if (inp_frame_codes) {
            ggml_backend_tensor_set(inp_frame_codes, frame_codes, 0,
                                    impl_->model.config.n_codebooks * sizeof(int32_t));
        }
        struct ggml_tensor * inp_overlay = ggml_graph_get_tensor(gf, "inp_overlay_embd");
        if (inp_overlay) {
            ggml_backend_tensor_set(inp_overlay, overlay_embd, 0,
                                    impl_->model.config.hidden_size * sizeof(float));
        }
    } else {
        struct ggml_tensor * inp_step = ggml_graph_get_tensor(gf, "inp_step_embd");
        if (!inp_step) {
            error_msg_ = "Failed to find inp_step_embd tensor";
            return false;
        }
        ggml_backend_tensor_set(inp_step, step_embd, 0,
                                impl_->model.config.hidden_size * sizeof(float));
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
        const int32_t n_kv_pad = std::min<int32_t>(impl_->state.cache.n_ctx, GGML_PAD(n_past + 1, 256));
        auto & mask = impl_->state.talker_mask;
        if (mask.size() != (size_t) n_kv_pad) {
            mask.resize((size_t) n_kv_pad);
        }
        const ggml_fp16_t zero_fp16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf_fp16 = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < n_kv_pad; ++i) {
            mask[(size_t) i] = (i <= n_past) ? zero_fp16 : neg_inf_fp16;
        }
        ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) {
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        impl_->timing->t_talker_data_ms += dt_ms;
        impl_->timing->t_talker_input_upload_ms += dt_ms;
    }
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (ggml_backend_sched_graph_compute(step_sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        if (use_talker_replay) {
            impl_->state.talker_replay_ready = false;
            impl_->state.talker_replay_failed = true;
        }
        ggml_backend_sched_reset(step_sched);
        return false;
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_talker_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");
    if (!hidden) {
        error_msg_ = "Failed to find hidden_states tensor";
        if (!use_talker_replay) {
            ggml_backend_sched_reset(step_sched);
        }
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (read_hidden || hidden_out) {
        last_hidden_.resize(impl_->model.config.hidden_size);
    }
    if (hidden_out) {
        hidden_out->resize(impl_->model.config.hidden_size);
        ggml_backend_tensor_get(hidden, hidden_out->data(), 0,
                                impl_->model.config.hidden_size * sizeof(float));
        last_hidden_ = *hidden_out;
    } else if (read_hidden) {
        ggml_backend_tensor_get(hidden, last_hidden_.data(), 0,
                                impl_->model.config.hidden_size * sizeof(float));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) {
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        impl_->timing->t_talker_data_ms += dt_ms;
        impl_->timing->t_talker_hidden_read_ms += dt_ms;
    }
    t0 = clk::now();
#endif

    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        if (!use_talker_replay) {
            ggml_backend_sched_reset(step_sched);
        }
        return false;
    }

    output.resize(impl_->model.config.codec_vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) {
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        impl_->timing->t_talker_data_ms += dt_ms;
        impl_->timing->t_talker_logits_read_ms += dt_ms;
    }
    t0 = clk::now();
#endif

    impl_->state.cache.n_used = n_past + 1;

    if (!use_talker_replay) {
        ggml_backend_sched_reset(step_sched);
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing && !use_talker_replay) {
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        impl_->timing->t_talker_data_ms += dt_ms;
        impl_->timing->t_talker_sched_reset_ms += dt_ms;
    }
#endif

    return true;
}

bool TTSTransformer::forward_step(const float * step_embd, int32_t n_past,
                                  std::vector<float> & output,
                                  std::vector<float> * hidden_out) {
    return forward_step_internal(step_embd, nullptr, nullptr, n_past, output, hidden_out, true);
}

bool TTSTransformer::forward_codec(int32_t codec_token, int32_t n_past,
                                   std::vector<float> & output) {
    std::vector<float> codec_row;
    if (!transformer_internal::ops::lookup_embedding_rows(*this, impl_->model.codec_embd, &codec_token, 1,
                               "inp_legacy_codec_token", "legacy_codec_row",
                               codec_row)) {
        return false;
    }

    return forward_step(codec_row.data(), n_past, output, nullptr);
}

} // namespace qwen3_tts
