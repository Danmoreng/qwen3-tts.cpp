#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cmath>
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

    const auto & trace_cfg = transformer_internal::get_debug_trace_config();
    if (trace_cfg.enabled) {
        auto dump_last_token_f32 = [&](const char * tensor_name, const char * out_name) {
            struct ggml_tensor * tensor = ggml_graph_get_tensor(gf, tensor_name);
            if (!tensor) {
                return;
            }
            const size_t count = (size_t) ggml_nelements(tensor);
            const int32_t hidden_size = impl_->model.config.hidden_size;
            if (count < (size_t) hidden_size || (count % (size_t) hidden_size) != 0) {
                return;
            }
            std::vector<float> all(count);
            if (tensor->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(tensor, all.data(), 0, count * sizeof(float));
            } else if (tensor->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> tmp(count);
                ggml_backend_tensor_get(tensor, tmp.data(), 0, count * sizeof(ggml_fp16_t));
                ggml_fp16_to_fp32_row(tmp.data(), all.data(), (int64_t) count);
            } else if (tensor->type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> tmp(count);
                ggml_backend_tensor_get(tensor, tmp.data(), 0, count * sizeof(ggml_bf16_t));
                ggml_bf16_to_fp32_row(tmp.data(), all.data(), (int64_t) count);
            } else {
                return;
            }
            const size_t row_offset = count - (size_t) hidden_size;
            transformer_internal::debug_trace_write_bin(trace_cfg, out_name, all.data() + row_offset,
                                                        (size_t) hidden_size, "f32",
                                                        {(int64_t) hidden_size});
        };

        const int32_t n_layer = impl_->model.config.n_layers;
        for (int32_t il = 0; il < n_layer; ++il) {
            char tensor_name[96];
            char out_name[128];
            snprintf(tensor_name, sizeof(tensor_name), "talker_prefill_layer%02d_hidden", il);
            snprintf(out_name, sizeof(out_name), "talker_prefill_layer%02d_hidden.f32.bin", il);
            dump_last_token_f32(tensor_name, out_name);
            snprintf(out_name, sizeof(out_name), "frame000_talker_layer%02d_hidden.f32.bin", il);
            dump_last_token_f32(tensor_name, out_name);
        }
        dump_last_token_f32("hidden_states", "talker_prefill_final_hidden.f32.bin");
        dump_last_token_f32("hidden_states", "frame000_talker_final_hidden.f32.bin");
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
                                  std::vector<float> * hidden_out,
                                  int32_t trace_frame) {
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
    struct ggml_cgraph * gf = transformer_internal::ops::build_step_graph(*this, n_past);
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
    struct ggml_tensor * inp_step = ggml_graph_get_tensor(gf, "inp_step_embd");
    if (inp_step) {
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
        std::vector<ggml_fp16_t> mask(impl_->state.cache.n_ctx);
        const ggml_fp16_t zero_fp16 = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf_fp16 = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < impl_->state.cache.n_ctx; ++i) {
            mask[(size_t) i] = (i <= n_past) ? zero_fp16 : neg_inf_fp16;
        }
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

    struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden_states");
    if (!hidden) {
        error_msg_ = "Failed to find hidden_states tensor";
        ggml_backend_sched_reset(impl_->state.sched);
        return false;
    }

    const auto & trace_cfg = transformer_internal::get_debug_trace_config();
    if (transformer_internal::debug_trace_should_dump_frame(trace_cfg, trace_frame)) {
        auto dump_tensor_f32 = [&](const char * tensor_name, const char * out_name) {
            struct ggml_tensor * tensor = ggml_graph_get_tensor(gf, tensor_name);
            if (!tensor) {
                return;
            }
            const size_t count = (size_t) ggml_nelements(tensor);
            std::vector<float> all(count);
            if (tensor->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(tensor, all.data(), 0, count * sizeof(float));
            } else if (tensor->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> tmp(count);
                ggml_backend_tensor_get(tensor, tmp.data(), 0, count * sizeof(ggml_fp16_t));
                ggml_fp16_to_fp32_row(tmp.data(), all.data(), (int64_t) count);
            } else if (tensor->type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> tmp(count);
                ggml_backend_tensor_get(tensor, tmp.data(), 0, count * sizeof(ggml_bf16_t));
                ggml_bf16_to_fp32_row(tmp.data(), all.data(), (int64_t) count);
            } else {
                return;
            }
            transformer_internal::debug_trace_write_bin(trace_cfg, out_name, all.data(), count, "f32",
                                                        {(int64_t) count});
        };

        const int32_t n_layer = impl_->model.config.n_layers;
        for (int32_t il = 0; il < n_layer; ++il) {
            char tensor_name[96];
            char out_name[128];
            snprintf(tensor_name, sizeof(tensor_name), "talker_step_layer%02d_hidden", il);
            snprintf(out_name, sizeof(out_name), "frame%03d_talker_layer%02d_hidden.f32.bin", trace_frame, il);
            dump_tensor_f32(tensor_name, out_name);
        }
        char final_name[128];
        snprintf(final_name, sizeof(final_name), "frame%03d_talker_final_hidden.f32.bin", trace_frame);
        dump_tensor_f32("hidden_states", final_name);
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

    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
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

    return forward_step(codec_row.data(), n_past, output, nullptr, -1);
}

} // namespace qwen3_tts
