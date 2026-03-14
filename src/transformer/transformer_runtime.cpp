#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace qwen3_tts {

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
            fprintf(stderr, "  ERROR: inp_pos write out of bounds! nbytes=%zu, write=%zu\n", ggml_nbytes(inp_pos), write_size);
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
            fprintf(stderr, "  ERROR: inp_mrope_pos write out of bounds! nbytes=%zu, write=%zu\n", ggml_nbytes(inp_mrope_pos), write_size);
        }
        ggml_backend_tensor_set(inp_mrope_pos, positions.data(), 0, write_size);
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

bool TTSTransformer::forward_step(const float * step_embd, int32_t n_past,
                                  std::vector<float> & output,
                                  std::vector<float> * hidden_out) {
    if (!impl_->model.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!step_embd) {
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

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    if (!transformer_internal::ops::ensure_cached_talker_step_graph(*this)) {
        return false;
    }
    struct ggml_cgraph * gf = impl_->state.talker_step_graph;
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_talker_graph_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
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
    if (impl_->timing) impl_->timing->t_talker_graph_alloc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    struct ggml_tensor * inp_step = impl_->state.talker_step_inp_step;
    if (inp_step) {
        ggml_backend_tensor_set(inp_step, step_embd, 0,
                                impl_->model.config.hidden_size * sizeof(float));
    }

    struct ggml_tensor * inp_pos = impl_->state.talker_step_inp_pos;
    if (inp_pos) {
        int32_t pos = n_past;
        ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
    }

    struct ggml_tensor * inp_mrope_pos = impl_->state.talker_step_inp_mrope_pos;
    if (inp_mrope_pos && impl_->model.config.use_mrope) {
        int32_t positions[4] = {n_past, n_past, n_past, 0};
        ggml_backend_tensor_set(inp_mrope_pos, positions, 0, 4 * sizeof(int32_t));
    }

    struct ggml_tensor * inp_mask = impl_->state.talker_step_inp_mask;
    if (inp_mask) {
        const ggml_fp16_t zero_fp16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf_fp16 = ggml_fp32_to_fp16(-INFINITY);
        auto & mask = impl_->state.talker_step_mask;
        if (mask.size() != (size_t) impl_->state.cache.n_ctx) {
            mask.assign((size_t) impl_->state.cache.n_ctx, neg_inf_fp16);
            impl_->state.talker_step_mask_n_past = -1;
        }
        if (n_past < impl_->state.talker_step_mask_n_past) {
            std::fill(mask.begin(), mask.end(), neg_inf_fp16);
            impl_->state.talker_step_mask_n_past = -1;
        }
        for (int32_t i = impl_->state.talker_step_mask_n_past + 1;
             i <= n_past && i < impl_->state.cache.n_ctx;
             ++i) {
            mask[(size_t) i] = zero_fp16;
        }
        impl_->state.talker_step_mask_n_past = n_past;
        ggml_backend_tensor_set(inp_mask, mask.data(), 0, impl_->state.cache.n_ctx * sizeof(ggml_fp16_t));
    }
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_talker_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
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
    if (impl_->timing) impl_->timing->t_talker_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    struct ggml_tensor * hidden = impl_->state.talker_step_hidden;
    if (!hidden) {
        error_msg_ = "Failed to find hidden_states tensor";
        ggml_backend_sched_reset(impl_->state.sched);
        return false;
    }

#ifdef QWEN3_TTS_TIMING
    t0 = clk::now();
#endif
    last_hidden_.resize(impl_->model.config.hidden_size);
    if (hidden_out) {
        hidden_out->resize(impl_->model.config.hidden_size);
        ggml_backend_tensor_get(hidden, hidden_out->data(), 0,
                                impl_->model.config.hidden_size * sizeof(float));
        last_hidden_ = *hidden_out;
    } else {
        ggml_backend_tensor_get(hidden, last_hidden_.data(), 0,
                                impl_->model.config.hidden_size * sizeof(float));
    }

    struct ggml_tensor * logits = impl_->state.talker_step_logits;
    if (!logits) {
        error_msg_ = "Failed to find logits tensor";
        ggml_backend_sched_reset(impl_->state.sched);
        return false;
    }

    output.resize(impl_->model.config.codec_vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));

    impl_->state.cache.n_used = n_past + 1;

    ggml_backend_sched_reset(impl_->state.sched);
#ifdef QWEN3_TTS_TIMING
    t1 = clk::now();
    if (impl_->timing) impl_->timing->t_talker_data_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    return true;
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
