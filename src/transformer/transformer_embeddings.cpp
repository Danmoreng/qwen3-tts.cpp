#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace qwen3_tts {

bool transformer_internal::ops::lookup_embedding_rows(TTSTransformer & self,
                                                       struct ggml_tensor * embedding,
                                                       const int32_t * token_ids,
                                                       int32_t n_tokens,
                                                       const char * input_name,
                                                       const char * output_name,
                                                       std::vector<float> & output) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;

    if (!impl->model.ctx) {
        error_msg = "Model not loaded";
        return false;
    }
    if (!embedding) {
        error_msg = "Embedding tensor not found";
        return false;
    }
    if (n_tokens <= 0) {
        output.clear();
        return true;
    }

    const int32_t embd_dim = (int32_t) embedding->ne[0];
    if (n_tokens <= 32 &&
        (embedding->type == GGML_TYPE_F16 || embedding->type == GGML_TYPE_F32)) {
        output.resize((size_t) embd_dim * n_tokens);
        for (int32_t t = 0; t < n_tokens; ++t) {
            if (!lookup_single_embedding_row(self, embedding, token_ids[t],
                                             output.data() + (size_t) t * embd_dim)) {
                return false;
            }
        }
        return true;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ impl->state.compute_meta.size(),
        /*.mem_buffer =*/ impl->state.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, input_name);
    ggml_set_input(inp_tokens);

    struct ggml_tensor * rows = ggml_get_rows(ctx0, embedding, inp_tokens);
    rows = ggml_cast(ctx0, rows, GGML_TYPE_F32);
    ggml_set_name(rows, output_name);
    ggml_set_output(rows);

    ggml_build_forward_expand(gf, rows);

    if (!ggml_backend_sched_alloc_graph(impl->state.sched, gf)) {
        error_msg = "Failed to allocate embedding lookup graph";
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * inp = ggml_graph_get_tensor(gf, input_name);
    ggml_backend_tensor_set(inp, token_ids, 0, n_tokens * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(impl->state.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg = "Failed to compute embedding lookup graph";
        ggml_backend_sched_reset(impl->state.sched);
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * out = ggml_graph_get_tensor(gf, output_name);
    if (!out) {
        error_msg = "Failed to find embedding lookup output tensor";
        ggml_backend_sched_reset(impl->state.sched);
        ggml_free(ctx0);
        return false;
    }

    output.resize((size_t) embedding->ne[0] * n_tokens);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_sched_reset(impl->state.sched);
    ggml_free(ctx0);
    return true;
}

bool transformer_internal::ops::lookup_single_embedding_row(TTSTransformer & self,
                                                            struct ggml_tensor * embedding,
                                                            int32_t token_id,
                                                            float * out_row) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;

    if (!embedding) {
        error_msg = "Embedding tensor not found";
        return false;
    }
    if (!out_row) {
        error_msg = "Embedding output row is null";
        return false;
    }

    const int64_t embd_dim = embedding->ne[0];
    const int64_t vocab_size = embedding->ne[1];
    if (token_id < 0 || token_id >= vocab_size) {
        error_msg = "Embedding token ID out of range";
        return false;
    }

    const size_t row_offset = (size_t) token_id * embedding->nb[1];
    if (embedding->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(embedding, out_row, row_offset, (size_t) embd_dim * sizeof(float));
        return true;
    }
    if (embedding->type == GGML_TYPE_F16) {
        impl->embd_row_fp16_scratch.resize((size_t) embd_dim);
        ggml_backend_tensor_get(embedding, impl->embd_row_fp16_scratch.data(),
                                row_offset, (size_t) embd_dim * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < embd_dim; ++i) {
            out_row[i] = ggml_fp16_to_fp32(impl->embd_row_fp16_scratch[i]);
        }
        return true;
    }

    std::vector<int32_t> single_token = { token_id };
    std::vector<float> single_out;
    if (!lookup_embedding_rows(self, embedding, single_token.data(), 1,
                               "inp_compat_embed", "out_compat_embed", single_out)) {
        return false;
    }
    memcpy(out_row, single_out.data(), (size_t) embd_dim * sizeof(float));
    return true;
}

bool transformer_internal::ops::project_text_tokens(TTSTransformer & self,
                                                    const int32_t * text_tokens,
                                                    int32_t n_tokens,
                                                    std::vector<float> & output) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;

    if (!impl->model.ctx) {
        error_msg = "Model not loaded";
        return false;
    }
    if (n_tokens <= 0) {
        output.clear();
        return true;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ impl->state.compute_meta.size(),
        /*.mem_buffer =*/ impl->state.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);

    struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_text_tokens");
    ggml_set_input(inp_tokens);

    struct ggml_tensor * cur = ggml_get_rows(ctx0, impl->model.text_embd, inp_tokens);
    cur = ggml_mul_mat(ctx0, impl->model.text_proj_fc1, cur);
    cur = ggml_add(ctx0, cur, impl->model.text_proj_fc1_bias);
    cur = ggml_silu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, impl->model.text_proj_fc2, cur);
    cur = ggml_add(ctx0, cur, impl->model.text_proj_fc2_bias);

    ggml_set_name(cur, "text_proj_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (!ggml_backend_sched_alloc_graph(impl->state.sched, gf)) {
        error_msg = "Failed to allocate text projection graph";
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "inp_text_tokens");
    ggml_backend_tensor_set(inp, text_tokens, 0, n_tokens * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(impl->state.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg = "Failed to compute text projection graph";
        ggml_backend_sched_reset(impl->state.sched);
        ggml_free(ctx0);
        return false;
    }

    struct ggml_tensor * out = ggml_graph_get_tensor(gf, "text_proj_out");
    if (!out) {
        error_msg = "Failed to find text projection output tensor";
        ggml_backend_sched_reset(impl->state.sched);
        ggml_free(ctx0);
        return false;
    }

    output.resize((size_t) impl->model.config.hidden_size * n_tokens);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_sched_reset(impl->state.sched);
    ggml_free(ctx0);
    return true;
}

bool transformer_internal::ops::build_prefill_graph(TTSTransformer & self,
                                                    const int32_t * text_tokens, int32_t n_tokens,
                                                    const float * speaker_embd, int32_t language_id,
                                                    std::vector<float> & prefill_embd,
                                                    std::vector<float> & trailing_text_hidden,
                                                    std::vector<float> & tts_pad_embed,
                                                    const int32_t * instruct_tokens,
                                                    int32_t n_instruct_tokens,
                                                    const int32_t * reference_tokens,
                                                    int32_t n_reference_tokens,
                                                    const int32_t * reference_codes,
                                                    int32_t n_reference_frames,
                                                    int32_t n_reference_codebooks) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;

    if (!text_tokens) {
        error_msg = "text_tokens is null";
        return false;
    }
    if (n_tokens < 4) {
        error_msg = "Need at least 4 text tokens for prefill";
        return false;
    }

    const bool has_reference = reference_tokens || reference_codes ||
                               n_reference_tokens > 0 || n_reference_frames > 0;
    fprintf(stderr, "  build_prefill_graph: n_tokens=%d, n_instruct=%d, has_speaker=%s, has_reference=%s\n",
            n_tokens, n_instruct_tokens, speaker_embd ? "yes" : "no",
            has_reference ? "yes" : "no");

    const auto & cfg = impl->model.config;
    const int32_t hidden_size = cfg.hidden_size;

    int32_t special_tokens[3] = {
        cfg.tts_bos_token_id,
        cfg.tts_eos_token_id,
        cfg.tts_pad_token_id,
    };

    std::vector<float> special_proj;
    if (!project_text_tokens(self, special_tokens, 3, special_proj)) {
        return false;
    }

    std::vector<float> tts_bos_embed(hidden_size);
    std::vector<float> tts_eos_embed(hidden_size);
    tts_pad_embed.resize(hidden_size);
    memcpy(tts_bos_embed.data(), special_proj.data() + 0 * hidden_size, hidden_size * sizeof(float));
    memcpy(tts_eos_embed.data(), special_proj.data() + 1 * hidden_size, hidden_size * sizeof(float));
    memcpy(tts_pad_embed.data(), special_proj.data() + 2 * hidden_size, hidden_size * sizeof(float));

    std::vector<float> instruct_embed;
    if (n_instruct_tokens > 0 && instruct_tokens) {
        if (!project_text_tokens(self, instruct_tokens, n_instruct_tokens, instruct_embed)) {
            return false;
        }
    }

    std::vector<float> role_embed;
    if (!project_text_tokens(self, text_tokens, 3, role_embed)) {
        return false;
    }

    std::vector<int32_t> codec_prefill_tokens;
    if (language_id < 0) {
        codec_prefill_tokens = {
            cfg.codec_nothink_id,
            cfg.codec_think_bos_id,
            cfg.codec_think_eos_id,
        };
    } else {
        codec_prefill_tokens = {
            cfg.codec_think_id,
            cfg.codec_think_bos_id,
            language_id,
            cfg.codec_think_eos_id,
        };
    }

    std::vector<float> codec_prefill_embed;
    if (!lookup_embedding_rows(self, impl->model.codec_embd, codec_prefill_tokens.data(),
                               (int32_t) codec_prefill_tokens.size(),
                               "inp_codec_prefill_tokens", "codec_prefill_rows",
                               codec_prefill_embed)) {
        return false;
    }

    int32_t codec_tail_tokens[2] = { cfg.codec_pad_id, cfg.codec_bos_id };
    std::vector<float> codec_tail_embed;
    if (!lookup_embedding_rows(self, impl->model.codec_embd, codec_tail_tokens, 2,
                               "inp_codec_tail_tokens", "codec_tail_rows",
                               codec_tail_embed)) {
        return false;
    }

    if (has_reference) {
        if (!speaker_embd) {
            error_msg = "ICL reference prefill requires a speaker embedding";
            return false;
        }
        if (!reference_tokens || n_reference_tokens < 5) {
            error_msg = "ICL reference prefill requires reference text tokens";
            return false;
        }
        if (!reference_codes || n_reference_frames <= 0) {
            error_msg = "ICL reference prefill requires reference speech codes";
            return false;
        }
        if (n_reference_codebooks != cfg.n_codebooks) {
            error_msg = "ICL reference codebook count does not match model";
            return false;
        }
        if (n_tokens < 8) {
            error_msg = "Need at least 8 text tokens for ICL prefill";
            return false;
        }

        auto append_row = [&](std::vector<float> & dst, const float * row) {
            dst.insert(dst.end(), row, row + hidden_size);
        };
        auto append_rows = [&](std::vector<float> & dst, const std::vector<float> & rows) {
            dst.insert(dst.end(), rows.begin(), rows.end());
        };
        auto append_added_row = [&](std::vector<float> & dst, const float * a, const float * b) {
            const size_t base = dst.size();
            dst.resize(base + (size_t) hidden_size);
            float * out = dst.data() + base;
            for (int32_t h = 0; h < hidden_size; ++h) {
                out[h] = a[h] + b[h];
            }
        };

        std::vector<float> prompt;
        if (n_instruct_tokens > 0 && instruct_tokens) {
            append_rows(prompt, instruct_embed);
        }
        append_rows(prompt, role_embed);

        const int32_t codec_input_len = (int32_t) codec_prefill_tokens.size() + 1 + 2;
        std::vector<float> codec_input_embedding((size_t) codec_input_len * hidden_size);
        int32_t dst_token = 0;
        memcpy(codec_input_embedding.data(), codec_prefill_embed.data(),
               codec_prefill_embed.size() * sizeof(float));
        dst_token += (int32_t) codec_prefill_tokens.size();
        memcpy(codec_input_embedding.data() + (size_t) dst_token * hidden_size,
               speaker_embd, hidden_size * sizeof(float));
        ++dst_token;
        memcpy(codec_input_embedding.data() + (size_t) dst_token * hidden_size,
               codec_tail_embed.data(), codec_tail_embed.size() * sizeof(float));

        const int32_t codec_plus_overlay_len = codec_input_len - 1;
        for (int32_t row = 0; row < codec_plus_overlay_len; ++row) {
            const float * overlay = (row == codec_plus_overlay_len - 1)
                ? tts_bos_embed.data()
                : tts_pad_embed.data();
            append_added_row(prompt, overlay,
                             codec_input_embedding.data() + (size_t) row * hidden_size);
        }

        const int32_t target_text_count = n_tokens - 3 - 5;
        const int32_t reference_text_count = n_reference_tokens - 3 - 2;
        if (target_text_count < 0 || reference_text_count < 0) {
            error_msg = "ICL text token layout is invalid";
            return false;
        }

        std::vector<int32_t> combined_text;
        combined_text.reserve((size_t) reference_text_count + target_text_count);
        combined_text.insert(combined_text.end(),
                             reference_tokens + 3,
                             reference_tokens + 3 + reference_text_count);
        combined_text.insert(combined_text.end(),
                             text_tokens + 3,
                             text_tokens + 3 + target_text_count);

        std::vector<float> text_embed;
        if (!combined_text.empty()) {
            if (!project_text_tokens(self, combined_text.data(), (int32_t) combined_text.size(), text_embed)) {
                return false;
            }
        }
        append_row(text_embed, tts_eos_embed.data());

        std::vector<float> ref_codec;
        ref_codec.reserve((size_t) (n_reference_frames + 1) * hidden_size);
        append_row(ref_codec, codec_tail_embed.data() + (size_t) hidden_size);
        std::vector<float> code_row(hidden_size);
        std::vector<float> summed(hidden_size);
        for (int32_t frame = 0; frame < n_reference_frames; ++frame) {
            std::fill(summed.begin(), summed.end(), 0.0f);
            for (int32_t cb = 0; cb < n_reference_codebooks; ++cb) {
                const int32_t code = reference_codes[(size_t) frame * n_reference_codebooks + cb];
                struct ggml_tensor * table = cb == 0
                    ? impl->model.codec_embd
                    : impl->model.code_pred_embd[(size_t) cb - 1];
                if (!lookup_single_embedding_row(self, table, code, code_row.data())) {
                    return false;
                }
                for (int32_t h = 0; h < hidden_size; ++h) {
                    summed[h] += code_row[h];
                }
            }
            append_row(ref_codec, summed.data());
        }

        const float * codec_pad_embed = codec_tail_embed.data();
        const int32_t text_rows = (int32_t) (text_embed.size() / hidden_size);
        for (int32_t row = 0; row < text_rows; ++row) {
            append_added_row(prompt,
                             text_embed.data() + (size_t) row * hidden_size,
                             codec_pad_embed);
        }

        const int32_t ref_codec_rows = (int32_t) (ref_codec.size() / hidden_size);
        for (int32_t row = 0; row < ref_codec_rows; ++row) {
            append_added_row(prompt,
                             tts_pad_embed.data(),
                             ref_codec.data() + (size_t) row * hidden_size);
        }

        trailing_text_hidden = tts_pad_embed;

        prefill_embd = std::move(prompt);
        return true;
    }

    const bool has_speaker = (speaker_embd != nullptr);
    const int32_t codec_input_len = (int32_t) codec_prefill_tokens.size() + (has_speaker ? 1 : 0) + 2;
    std::vector<float> codec_input_embedding((size_t) codec_input_len * hidden_size);

    int32_t dst_token = 0;
    memcpy(codec_input_embedding.data(), codec_prefill_embed.data(), codec_prefill_embed.size() * sizeof(float));
    dst_token += (int32_t) codec_prefill_tokens.size();

    if (has_speaker) {
        memcpy(codec_input_embedding.data() + (size_t) dst_token * hidden_size,
               speaker_embd, hidden_size * sizeof(float));
        ++dst_token;
    }

    memcpy(codec_input_embedding.data() + (size_t) dst_token * hidden_size,
           codec_tail_embed.data(), codec_tail_embed.size() * sizeof(float));

    const int32_t codec_plus_overlay_len = codec_input_len - 1;
    std::vector<float> codec_plus_overlay((size_t) codec_plus_overlay_len * hidden_size);
    for (int32_t t = 0; t < codec_plus_overlay_len; ++t) {
        const float * overlay = (t == codec_plus_overlay_len - 1)
            ? tts_bos_embed.data()
            : tts_pad_embed.data();
        const float * codec_row = codec_input_embedding.data() + (size_t) t * hidden_size;
        float * out_row = codec_plus_overlay.data() + (size_t) t * hidden_size;
        for (int32_t h = 0; h < hidden_size; ++h) {
            out_row[h] = overlay[h] + codec_row[h];
        }
    }

    const int32_t suffix_len = 5;
    const int32_t text_body_count = n_tokens - 3 - suffix_len;
    if (text_body_count < 0) {
        error_msg = "Text token layout is invalid";
        return false;
    }

    std::vector<float> text_body_proj;
    if (text_body_count > 0) {
        if (!project_text_tokens(self, text_tokens + 3, text_body_count, text_body_proj)) {
            return false;
        }
    }

    const float * codec_pad_embed = codec_tail_embed.data();
    const float * codec_bos_embed = codec_tail_embed.data() + (size_t) hidden_size;

    const int32_t prefill_len = n_instruct_tokens + 3 + codec_plus_overlay_len + text_body_count + 2;
    prefill_embd.resize((size_t) prefill_len * hidden_size);

    int32_t offset = 0;
    if (n_instruct_tokens > 0 && instruct_tokens) {
        memcpy(prefill_embd.data() + (size_t) offset * hidden_size,
               instruct_embed.data(), instruct_embed.size() * sizeof(float));
        offset += n_instruct_tokens;
    }

    memcpy(prefill_embd.data() + (size_t) offset * hidden_size,
           role_embed.data(), role_embed.size() * sizeof(float));
    offset += 3;
    memcpy(prefill_embd.data() + (size_t) offset * hidden_size,
           codec_plus_overlay.data(), codec_plus_overlay.size() * sizeof(float));
    offset += codec_plus_overlay_len;

    for (int32_t t = 0; t < text_body_count; ++t) {
        float * dst = prefill_embd.data() + (size_t) offset * hidden_size;
        const float * text_row = text_body_proj.data() + (size_t) t * hidden_size;
        for (int32_t h = 0; h < hidden_size; ++h) {
            dst[h] = text_row[h] + codec_pad_embed[h];
        }
        ++offset;
    }

    float * eos_row = prefill_embd.data() + (size_t) offset * hidden_size;
    for (int32_t h = 0; h < hidden_size; ++h) {
        eos_row[h] = tts_eos_embed[h] + codec_pad_embed[h];
    }
    ++offset;

    float * bos_row = prefill_embd.data() + (size_t) offset * hidden_size;
    for (int32_t h = 0; h < hidden_size; ++h) {
        bos_row[h] = tts_pad_embed[h] + codec_bos_embed[h];
    }
    ++offset;

    if (offset != prefill_len) {
        error_msg = "Internal error: prefill layout length mismatch";
        return false;
    }

    trailing_text_hidden = tts_pad_embed;

    return true;
}

bool TTSTransformer::get_named_speaker_embedding(const std::string & speaker_name,
                                                 std::vector<float> & speaker_embedding) {
    if (!impl_->model.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (!impl_->model.codec_embd) {
        error_msg_ = "Model missing codec embedding tensor";
        return false;
    }
    if (impl_->model.config.speaker_id_map.empty()) {
        error_msg_ = "No speaker map found in model metadata";
        return false;
    }

    const std::string key = transformer_internal::normalize_speaker_name(speaker_name);
    auto it = impl_->model.config.speaker_id_map.find(key);
    if (it == impl_->model.config.speaker_id_map.end()) {
        error_msg_ = "Unknown speaker: " + speaker_name;
        return false;
    }

    speaker_embedding.resize(impl_->model.config.hidden_size);
    if (!transformer_internal::ops::lookup_single_embedding_row(*this, impl_->model.codec_embd,
                                                                it->second, speaker_embedding.data())) {
        if (error_msg_.empty()) {
            error_msg_ = "Failed to lookup speaker embedding for speaker: " + speaker_name;
        }
        return false;
    }

    return true;
}

} // namespace qwen3_tts
