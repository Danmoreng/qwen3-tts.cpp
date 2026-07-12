#include "audio_tokenizer_decoder.h"
#include "decoder/decoder_state_internal.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace qwen3_tts {

namespace {

ggml_backend_t init_dedicated_decoder_backend(std::string & error_msg) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, nullptr);
    }
    if (!backend) {
        backend = init_cpu_backend("AudioTokenizerDecoder", &error_msg);
    }
    if (!backend) {
        error_msg = "Failed to initialize dedicated backend for AudioTokenizerDecoder";
    }
    return backend;
}

} // namespace

AudioTokenizerDecoder::AudioTokenizerDecoder()
    : impl_(std::make_unique<audio_decoder_private>()) {
}

AudioTokenizerDecoder::~AudioTokenizerDecoder() {
    unload_model();
}

const audio_decoder_config & AudioTokenizerDecoder::get_config() const {
    return impl_->model.config;
}

const std::string & AudioTokenizerDecoder::get_error() const {
    return impl_->error_msg;
}

const audio_decoder_timing & AudioTokenizerDecoder::get_last_timing() const {
    return impl_->last_timing;
}

void AudioTokenizerDecoder::unload_model() {
    auto & model = impl_->model;
    auto & state = impl_->state;
    auto & codes_buf = impl_->codes_buf;
    auto & codebook_input_bufs = impl_->codebook_input_bufs;
    auto & positions_buf = impl_->positions_buf;
    auto & mask_buf = impl_->mask_buf;

    decoder_internal::ops::release_cached_decode_graph(*this);
    decoder_internal::ops::release_stream(*this);
    free_audio_decoder_model(model);

    if (state.sched) {
        ggml_backend_sched_free(state.sched);
        state.sched = nullptr;
    }
    if (state.backend) {
        if (state.backend_shared) {
            release_preferred_backend(state.backend);
        } else {
            ggml_backend_free(state.backend);
        }
        state.backend = nullptr;
        state.backend_shared = true;
    }
    if (state.backend_cpu) {
        ggml_backend_free(state.backend_cpu);
        state.backend_cpu = nullptr;
    }

    state.compute_meta.clear();
    codes_buf.clear();
    codebook_input_bufs.clear();
    positions_buf.clear();
    mask_buf.clear();
    impl_->stream_codes_buf.clear();
    impl_->stream_positions_buf.clear();
    impl_->stream_rows_buf.clear();
    impl_->stream_mask_buf.clear();
}

void decoder_internal::ops::normalize_codebooks(AudioTokenizerDecoder & self) {
    auto & model = self.impl_->model;
    const float epsilon = 1e-5f;

    auto normalize_codebook = [epsilon](struct ggml_tensor * codebook, struct ggml_tensor * usage, const char *) {
        if (!codebook || !usage || !codebook->data || !usage->data) {
            return;
        }

        const int64_t codebook_dim = codebook->ne[0];
        const int64_t codebook_size = codebook->ne[1];

        ggml_fp16_t * cb_data = (ggml_fp16_t *) codebook->data;
        float * usage_data = (float *) usage->data;

        for (int64_t emb_idx = 0; emb_idx < codebook_size; ++emb_idx) {
            float u = usage_data[emb_idx];
            if (u < epsilon) {
                u = epsilon;
            }
            const float inv_u = 1.0f / u;

            for (int64_t dim_idx = 0; dim_idx < codebook_dim; ++dim_idx) {
                const int64_t mem_idx = dim_idx + emb_idx * codebook_dim;
                const float val = ggml_fp16_to_fp32(cb_data[mem_idx]);
                cb_data[mem_idx] = ggml_fp32_to_fp16(val * inv_u);
            }
        }
    };

    normalize_codebook(model.vq_first_codebook, model.vq_first_usage, "first");

    for (int i = 0; i < 15; ++i) {
        char name[16];
        snprintf(name, sizeof(name), "rest%d", i);
        normalize_codebook(model.vq_rest_codebook[i], model.vq_rest_usage[i], name);
    }
}

bool AudioTokenizerDecoder::load_model_impl(const std::string & model_path,
                                            bool shared_backend) {
    auto & model = impl_->model;
    auto & state = impl_->state;
    auto & error_msg = impl_->error_msg;

    unload_model();

    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg = loader.get_error();
        return false;
    }

    model.config.sample_rate = loader.get_u32(
        "qwen3-tts-tokenizer.sample_rate",
        loader.get_u32("qwen3-tts-tokenizer.output_sample_rate",
                       loader.get_u32("qwen3-tts.tokenizer.sample_rate", 24000)));
    model.config.n_codebooks = loader.get_u32(
        "qwen3-tts-tokenizer.num_codebooks",
        loader.get_u32("qwen3-tts-tokenizer.decoder.num_quantizers",
                       loader.get_u32("qwen3-tts.tokenizer.num_codebooks", 16)));
    model.config.codebook_size = loader.get_u32(
        "qwen3-tts-tokenizer.codebook_size",
        loader.get_u32("qwen3-tts-tokenizer.decoder.codebook_size",
                       loader.get_u32("qwen3-tts.tokenizer.codebook_size", 2048)));
    model.config.sliding_window = loader.get_u32(
        "qwen3-tts-tokenizer.decoder.sliding_window",
        loader.get_u32("qwen3-tts.tokenizer.decoder.sliding_window", 72));

    const int64_t n_tensors = loader.get_n_tensors();
    int dec_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "tok_dec.", 8) == 0) {
            dec_tensor_count++;
        }
    }

    if (dec_tensor_count == 0) {
        error_msg = "No decoder tensors found in model";
        return false;
    }

    const size_t ctx_size = ggml_tensor_overhead() * (dec_tensor_count + 72);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    model.ctx = ggml_init(params);
    if (!model.ctx) {
        error_msg = "Failed to initialize GGML context";
        return false;
    }

    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();

    bool zero_based_decoder_residuals = false;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name) {
            continue;
        }
        int blk_idx = 0;
        int res_idx = 0;
        if (sscanf(name, "tok_dec.dec.%d.res.%d.", &blk_idx, &res_idx) == 2 &&
            blk_idx >= 1 && blk_idx <= 4 && res_idx >= 0 && res_idx <= 1) {
            zero_based_decoder_residuals = true;
            break;
        }
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "tok_dec.", 8) != 0) {
            continue;
        }

        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta_ctx, name);
        if (!meta_tensor) {
            continue;
        }

        struct ggml_tensor * tensor = ggml_dup_tensor(model.ctx, meta_tensor);
        ggml_set_name(tensor, name);

        model.tensors[name] = tensor;

        std::string sname(name);

        if (sname == "tok_dec.vq_first.input_proj.weight") model.vq_first_input_proj = tensor;
        else if (sname == "tok_dec.vq_first.output_proj.weight") model.vq_first_output_proj = tensor;
        else if (sname == "tok_dec.vq_first.0.codebook") model.vq_first_codebook = tensor;
        else if (sname == "tok_dec.vq_first.0.usage") model.vq_first_usage = tensor;
        else if (sname == "tok_dec.vq_rest.input_proj.weight") model.vq_rest_input_proj = tensor;
        else if (sname == "tok_dec.vq_rest.output_proj.weight") model.vq_rest_output_proj = tensor;
        else if (sname == "tok_dec.pre_conv.weight") model.pre_conv_w = tensor;
        else if (sname == "tok_dec.pre_conv.bias") model.pre_conv_b = tensor;
        else if (sname == "tok_dec.pre_tfm.input_proj.weight") model.pre_tfm_input_proj_w = tensor;
        else if (sname == "tok_dec.pre_tfm.input_proj.bias") model.pre_tfm_input_proj_b = tensor;
        else if (sname == "tok_dec.pre_tfm.norm.weight") model.pre_tfm_norm_w = tensor;
        else if (sname == "tok_dec.pre_tfm.output_proj.weight") model.pre_tfm_output_proj_w = tensor;
        else if (sname == "tok_dec.pre_tfm.output_proj.bias") model.pre_tfm_output_proj_b = tensor;
        else if (sname == "tok_dec.dec.0.conv.weight") model.dec0_conv_w = tensor;
        else if (sname == "tok_dec.dec.0.conv.bias") model.dec0_conv_b = tensor;
        else if (sname == "tok_dec.dec.5.snake.alpha") model.dec5_snake_alpha = tensor;
        else if (sname == "tok_dec.dec.5.snake.beta") model.dec5_snake_beta = tensor;
        else if (sname == "tok_dec.dec.6.conv.weight") model.dec6_conv_w = tensor;
        else if (sname == "tok_dec.dec.6.conv.bias") model.dec6_conv_b = tensor;
        else if (sname.find("pre_tfm.blk.") != std::string::npos) {
            int blk_idx;
            if (sscanf(name, "tok_dec.pre_tfm.blk.%d.", &blk_idx) == 1 && blk_idx >= 0 && blk_idx < 8) {
                if (sname.find(".attn_v.weight") != std::string::npos) model.pre_tfm_layers[blk_idx].attn_v_w = tensor;
                else if (sname.find(".ffn_gate.weight") != std::string::npos) model.pre_tfm_layers[blk_idx].ffn_gate_w = tensor;
                else if (sname.find(".attn_norm.weight") != std::string::npos) model.pre_tfm_layers[blk_idx].attn_norm_w = tensor;
                else if (sname.find(".attn_q.weight") != std::string::npos) model.pre_tfm_layers[blk_idx].attn_q_w = tensor;
                else if (sname.find(".attn_k.weight") != std::string::npos) model.pre_tfm_layers[blk_idx].attn_k_w = tensor;
                else if (sname.find(".attn_output.weight") != std::string::npos) model.pre_tfm_layers[blk_idx].attn_output_w = tensor;
                else if (sname.find(".attn_scale") != std::string::npos) model.pre_tfm_layers[blk_idx].attn_scale = tensor;
                else if (sname.find(".ffn_norm.weight") != std::string::npos) model.pre_tfm_layers[blk_idx].ffn_norm_w = tensor;
                else if (sname.find(".ffn_up.weight") != std::string::npos) model.pre_tfm_layers[blk_idx].ffn_up_w = tensor;
                else if (sname.find(".ffn_down.weight") != std::string::npos) model.pre_tfm_layers[blk_idx].ffn_down_w = tensor;
                else if (sname.find(".ffn_scale") != std::string::npos) model.pre_tfm_layers[blk_idx].ffn_scale = tensor;
            }
        } else {
            int blk_idx, res_idx, cb_idx, n = 0;
            char suffix[64];
            const size_t name_len = strlen(name);
            const auto map_res_idx = [zero_based_decoder_residuals](int idx) -> int {
                if (zero_based_decoder_residuals) {
                    return (idx >= 0 && idx <= 2) ? idx : -1;
                }
                if (idx >= 2 && idx <= 4) return idx - 2;
                return -1;
            };

            #define MATCH1(fmt, var) (sscanf(name, fmt "%n", &var, &n) == 1 && (size_t) n == name_len)
            #define MATCH2(fmt, v1, v2) (sscanf(name, fmt "%n", &v1, &v2, &n) == 2 && (size_t) n == name_len)
            #define MATCH1S(fmt, var, suf) (sscanf(name, fmt, &var, suf) == 2)

            if (MATCH1("tok_dec.vq_rest.%d.codebook", cb_idx)) {
                if (cb_idx >= 0 && cb_idx < 15) {
                    model.vq_rest_codebook[cb_idx] = tensor;
                }
            } else if (MATCH1("tok_dec.vq_rest.%d.usage", cb_idx)) {
                if (cb_idx >= 0 && cb_idx < 15) {
                    model.vq_rest_usage[cb_idx] = tensor;
                }
            } else if (MATCH1S("tok_dec.upsample.%d.conv.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model.upsample[blk_idx].conv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model.upsample[blk_idx].conv_b = tensor;
                }
            } else if (MATCH1S("tok_dec.upsample.%d.dwconv.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model.upsample[blk_idx].dwconv_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model.upsample[blk_idx].dwconv_b = tensor;
                }
            } else if (MATCH1S("tok_dec.upsample.%d.norm.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model.upsample[blk_idx].norm_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model.upsample[blk_idx].norm_b = tensor;
                }
            } else if (MATCH1S("tok_dec.upsample.%d.pwconv1.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model.upsample[blk_idx].pwconv1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model.upsample[blk_idx].pwconv1_b = tensor;
                }
            } else if (MATCH1S("tok_dec.upsample.%d.pwconv2.%63s", blk_idx, suffix)) {
                if (blk_idx >= 0 && blk_idx < 2) {
                    if (strcmp(suffix, "weight") == 0) model.upsample[blk_idx].pwconv2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model.upsample[blk_idx].pwconv2_b = tensor;
                }
            } else if (MATCH1("tok_dec.upsample.%d.gamma", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 2) model.upsample[blk_idx].gamma = tensor;
            } else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_norm.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model.pre_tfm_layers[blk_idx].attn_norm_w = tensor;
            } else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_q.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model.pre_tfm_layers[blk_idx].attn_q_w = tensor;
            } else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_k.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model.pre_tfm_layers[blk_idx].attn_k_w = tensor;
            } else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_v.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model.pre_tfm_layers[blk_idx].attn_v_w = tensor;
            } else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_output.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model.pre_tfm_layers[blk_idx].attn_output_w = tensor;
            } else if (MATCH1("tok_dec.pre_tfm.blk.%d.attn_scale", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model.pre_tfm_layers[blk_idx].attn_scale = tensor;
            } else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_norm.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model.pre_tfm_layers[blk_idx].ffn_norm_w = tensor;
            } else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_gate.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model.pre_tfm_layers[blk_idx].ffn_gate_w = tensor;
            } else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_up.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model.pre_tfm_layers[blk_idx].ffn_up_w = tensor;
            } else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_down.weight", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model.pre_tfm_layers[blk_idx].ffn_down_w = tensor;
            } else if (MATCH1("tok_dec.pre_tfm.blk.%d.ffn_scale", blk_idx)) {
                if (blk_idx >= 0 && blk_idx < 8) model.pre_tfm_layers[blk_idx].ffn_scale = tensor;
            } else if (MATCH1("tok_dec.dec.%d.snake.alpha", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model.dec_blocks[blk_idx - 1].snake_alpha = tensor;
            } else if (MATCH1("tok_dec.dec.%d.snake.beta", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model.dec_blocks[blk_idx - 1].snake_beta = tensor;
            } else if (MATCH1("tok_dec.dec.%d.conv_t.weight", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model.dec_blocks[blk_idx - 1].conv_t_w = tensor;
            } else if (MATCH1("tok_dec.dec.%d.conv_t.bias", blk_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4) model.dec_blocks[blk_idx - 1].conv_t_b = tensor;
            } else if (MATCH2("tok_dec.dec.%d.res.%d.act1.alpha", blk_idx, res_idx)) {
                const int mapped_res_idx = map_res_idx(res_idx);
                if (blk_idx >= 1 && blk_idx <= 4 && mapped_res_idx >= 0) {
                    model.dec_blocks[blk_idx - 1].res[mapped_res_idx].act1_alpha = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.act1.beta", blk_idx, res_idx)) {
                const int mapped_res_idx = map_res_idx(res_idx);
                if (blk_idx >= 1 && blk_idx <= 4 && mapped_res_idx >= 0) {
                    model.dec_blocks[blk_idx - 1].res[mapped_res_idx].act1_beta = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.weight", blk_idx, res_idx)) {
                const int mapped_res_idx = map_res_idx(res_idx);
                if (blk_idx >= 1 && blk_idx <= 4 && mapped_res_idx >= 0) {
                    model.dec_blocks[blk_idx - 1].res[mapped_res_idx].conv1_w = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.bias", blk_idx, res_idx)) {
                const int mapped_res_idx = map_res_idx(res_idx);
                if (blk_idx >= 1 && blk_idx <= 4 && mapped_res_idx >= 0) {
                    model.dec_blocks[blk_idx - 1].res[mapped_res_idx].conv1_b = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.act2.alpha", blk_idx, res_idx)) {
                const int mapped_res_idx = map_res_idx(res_idx);
                if (blk_idx >= 1 && blk_idx <= 4 && mapped_res_idx >= 0) {
                    model.dec_blocks[blk_idx - 1].res[mapped_res_idx].act2_alpha = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.act2.beta", blk_idx, res_idx)) {
                const int mapped_res_idx = map_res_idx(res_idx);
                if (blk_idx >= 1 && blk_idx <= 4 && mapped_res_idx >= 0) {
                    model.dec_blocks[blk_idx - 1].res[mapped_res_idx].act2_beta = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.weight", blk_idx, res_idx)) {
                const int mapped_res_idx = map_res_idx(res_idx);
                if (blk_idx >= 1 && blk_idx <= 4 && mapped_res_idx >= 0) {
                    model.dec_blocks[blk_idx - 1].res[mapped_res_idx].conv2_w = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.bias", blk_idx, res_idx)) {
                const int mapped_res_idx = map_res_idx(res_idx);
                if (blk_idx >= 1 && blk_idx <= 4 && mapped_res_idx >= 0) {
                    model.dec_blocks[blk_idx - 1].res[mapped_res_idx].conv2_b = tensor;
                }
            }

            #undef MATCH1
            #undef MATCH2
            #undef MATCH1S
        }
    }

    decoder_internal::ops::prepare_snake_tensors(*this);
    decoder_internal::ops::prepare_transconv_tensors(*this);

    if (!load_tensor_data_from_file(model_path, gguf_ctx, model.ctx,
                                    model.tensors, model.buffer, error_msg,
                                    get_preferred_backend_type())) {
        return false;
    }

    if (!decoder_internal::ops::upload_snake_tensors(*this)) {
        return false;
    }
    if (!decoder_internal::ops::upload_transconv_tensors(*this)) {
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        model.dec_blocks[i].res[0].dilation = 1;
        model.dec_blocks[i].res[1].dilation = 3;
        model.dec_blocks[i].res[2].dilation = 9;
    }

    decoder_internal::ops::normalize_codebooks(*this);
    auto upload_if_present = [](struct ggml_tensor * t) {
        if (t && t->data) {
            ggml_backend_tensor_set(t, t->data, 0, ggml_nbytes(t));
        }
    };
    upload_if_present(model.vq_first_codebook);
    for (int i = 0; i < 15; ++i) {
        upload_if_present(model.vq_rest_codebook[i]);
    }

    state.backend_shared = shared_backend;
    state.backend = shared_backend
        ? init_preferred_backend("AudioTokenizerDecoder", &error_msg)
        : init_dedicated_decoder_backend(error_msg);
    if (!state.backend) {
        return false;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(state.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  AudioTokenizerDecoder backend: %s\n", device_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state.backend_cpu = init_cpu_backend("AudioTokenizerDecoder fallback", &error_msg);
        if (!state.backend_cpu) {
            error_msg = "Failed to initialize CPU fallback backend for AudioTokenizerDecoder";
            return false;
        }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(state.backend);
    if (state.backend_cpu) {
        backends.push_back(state.backend_cpu);
    }
    state.sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), QWEN3_TTS_DEC_MAX_NODES, false, true);
    if (!state.sched) {
        error_msg = "Failed to create backend scheduler";
        return false;
    }

    state.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_DEC_MAX_NODES + ggml_graph_overhead());

    return true;
}

bool AudioTokenizerDecoder::load_model(const std::string & model_path) {
    return load_model_impl(model_path, true);
}

bool AudioTokenizerDecoder::load_model_dedicated(const std::string & model_path) {
    return load_model_impl(model_path, false);
}

void free_audio_decoder_model(audio_decoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
}

void decoder_internal::ops::prepare_snake_tensors(AudioTokenizerDecoder & self) {
    auto & model = self.impl_->model;

    auto prepare_pair = [&](struct ggml_tensor * alpha,
                            struct ggml_tensor * beta,
                            struct ggml_tensor *& alpha_exp,
                            struct ggml_tensor *& inv_beta_exp,
                            const char * name_prefix) {
        if (!alpha || !beta) {
            return;
        }

        const int64_t channels = alpha->ne[0];
        alpha_exp = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 1, channels);
        inv_beta_exp = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 1, channels);

        char name[128];
        snprintf(name, sizeof(name), "%s.alpha_exp", name_prefix);
        ggml_set_name(alpha_exp, name);
        snprintf(name, sizeof(name), "%s.inv_beta_exp", name_prefix);
        ggml_set_name(inv_beta_exp, name);
    };

    for (int i = 0; i < 4; ++i) {
        decoder_block & block = model.dec_blocks[i];
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "tok_dec.dec.%d.snake", i + 1);
        prepare_pair(block.snake_alpha, block.snake_beta,
                     block.snake_alpha_exp, block.snake_inv_beta_exp, prefix);

        for (int r = 0; r < 3; ++r) {
            residual_block & res = block.res[r];
            snprintf(prefix, sizeof(prefix), "tok_dec.dec.%d.res.%d.act1", i + 1, r + 2);
            prepare_pair(res.act1_alpha, res.act1_beta,
                         res.act1_alpha_exp, res.act1_inv_beta_exp, prefix);
            snprintf(prefix, sizeof(prefix), "tok_dec.dec.%d.res.%d.act2", i + 1, r + 2);
            prepare_pair(res.act2_alpha, res.act2_beta,
                         res.act2_alpha_exp, res.act2_inv_beta_exp, prefix);
        }
    }

    prepare_pair(model.dec5_snake_alpha, model.dec5_snake_beta,
                 model.dec5_snake_alpha_exp, model.dec5_snake_inv_beta_exp,
                 "tok_dec.dec.5.snake");
}

bool decoder_internal::ops::upload_snake_tensors(AudioTokenizerDecoder & self) {
    auto & model = self.impl_->model;
    auto & error_msg = self.impl_->error_msg;

    auto upload_pair = [&](struct ggml_tensor * alpha,
                           struct ggml_tensor * beta,
                           struct ggml_tensor * alpha_exp,
                           struct ggml_tensor * inv_beta_exp,
                           const char * label) -> bool {
        if (!alpha || !beta || !alpha_exp || !inv_beta_exp) {
            return true;
        }
        if (alpha->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32) {
            error_msg = std::string("Snake tensor must be F32: ") + label;
            return false;
        }
        if (alpha->ne[0] != beta->ne[0]) {
            error_msg = std::string("Snake alpha/beta size mismatch: ") + label;
            return false;
        }

        const int64_t channels = alpha->ne[0];
        std::vector<float> alpha_host((size_t) channels);
        std::vector<float> beta_host((size_t) channels);
        std::vector<float> alpha_exp_host((size_t) channels);
        std::vector<float> inv_beta_exp_host((size_t) channels);

        ggml_backend_tensor_get(alpha, alpha_host.data(), 0, (size_t) channels * sizeof(float));
        ggml_backend_tensor_get(beta, beta_host.data(), 0, (size_t) channels * sizeof(float));

        for (int64_t i = 0; i < channels; ++i) {
            alpha_exp_host[(size_t) i] = expf(alpha_host[(size_t) i]);
            inv_beta_exp_host[(size_t) i] = 1.0f / (expf(beta_host[(size_t) i]) + 1e-9f);
        }

        ggml_backend_tensor_set(alpha_exp, alpha_exp_host.data(), 0,
                                (size_t) channels * sizeof(float));
        ggml_backend_tensor_set(inv_beta_exp, inv_beta_exp_host.data(), 0,
                                (size_t) channels * sizeof(float));
        return true;
    };

    for (int i = 0; i < 4; ++i) {
        decoder_block & block = model.dec_blocks[i];
        char label[64];
        snprintf(label, sizeof(label), "tok_dec.dec.%d.snake", i + 1);
        if (!upload_pair(block.snake_alpha, block.snake_beta,
                         block.snake_alpha_exp, block.snake_inv_beta_exp, label)) {
            return false;
        }

        for (int r = 0; r < 3; ++r) {
            residual_block & res = block.res[r];
            snprintf(label, sizeof(label), "tok_dec.dec.%d.res.%d.act1", i + 1, r + 2);
            if (!upload_pair(res.act1_alpha, res.act1_beta,
                             res.act1_alpha_exp, res.act1_inv_beta_exp, label)) {
                return false;
            }
            snprintf(label, sizeof(label), "tok_dec.dec.%d.res.%d.act2", i + 1, r + 2);
            if (!upload_pair(res.act2_alpha, res.act2_beta,
                             res.act2_alpha_exp, res.act2_inv_beta_exp, label)) {
                return false;
            }
        }
    }

    return upload_pair(model.dec5_snake_alpha, model.dec5_snake_beta,
                       model.dec5_snake_alpha_exp, model.dec5_snake_inv_beta_exp,
                       "tok_dec.dec.5.snake");
}

void decoder_internal::ops::prepare_transconv_tensors(AudioTokenizerDecoder & self) {
    auto & model = self.impl_->model;

    auto prepare_perm = [&](struct ggml_tensor * src,
                            struct ggml_tensor *& dst,
                            const char * name_prefix) {
        if (!src) {
            return;
        }

        const int64_t kernel = src->ne[0];
        const int64_t out_channels = src->ne[1];
        const int64_t in_channels = src->ne[2];
        dst = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32,
                                 in_channels, kernel * out_channels);

        char name[128];
        snprintf(name, sizeof(name), "%s.perm_f32", name_prefix);
        ggml_set_name(dst, name);
    };

    for (int i = 0; i < 2; ++i) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "tok_dec.upsample.%d.conv.weight", i);
        prepare_perm(model.upsample[i].conv_w, model.upsample[i].conv_w_perm, prefix);
    }

    for (int i = 0; i < 4; ++i) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "tok_dec.dec.%d.conv_t.weight", i + 1);
        prepare_perm(model.dec_blocks[i].conv_t_w, model.dec_blocks[i].conv_t_w_perm, prefix);
    }
}

bool decoder_internal::ops::upload_transconv_tensors(AudioTokenizerDecoder & self) {
    auto & model = self.impl_->model;
    auto & error_msg = self.impl_->error_msg;

    auto load_value = [](const void * data, enum ggml_type type, size_t idx) -> float {
        if (type == GGML_TYPE_F32) {
            return static_cast<const float *>(data)[idx];
        }
        if (type == GGML_TYPE_F16) {
            return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(data)[idx]);
        }
        return ggml_bf16_to_fp32(static_cast<const ggml_bf16_t *>(data)[idx]);
    };

    auto upload_perm = [&](struct ggml_tensor * src,
                           struct ggml_tensor * dst,
                           const char * label) -> bool {
        if (!src || !dst) {
            return true;
        }
        if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16) {
            error_msg = std::string("ConvTranspose weight cannot be pre-permuted: ") + label;
            return false;
        }

        const int64_t kernel = src->ne[0];
        const int64_t out_channels = src->ne[1];
        const int64_t in_channels = src->ne[2];
        const int64_t n = kernel * out_channels * in_channels;

        std::vector<uint8_t> src_host(ggml_nbytes(src));
        std::vector<float> dst_host((size_t) n);
        ggml_backend_tensor_get(src, src_host.data(), 0, src_host.size());

        for (int64_t ic = 0; ic < in_channels; ++ic) {
            for (int64_t oc = 0; oc < out_channels; ++oc) {
                for (int64_t k = 0; k < kernel; ++k) {
                    const size_t src_idx = (size_t) ic * (size_t) out_channels * (size_t) kernel
                                         + (size_t) oc * (size_t) kernel
                                         + (size_t) k;
                    const size_t dst_idx = ((size_t) oc * (size_t) kernel + (size_t) k)
                                         * (size_t) in_channels
                                         + (size_t) ic;
                    dst_host[dst_idx] = load_value(src_host.data(), src->type, src_idx);
                }
            }
        }

        ggml_backend_tensor_set(dst, dst_host.data(), 0, dst_host.size() * sizeof(float));
        return true;
    };

    for (int i = 0; i < 2; ++i) {
        char label[64];
        snprintf(label, sizeof(label), "tok_dec.upsample.%d.conv.weight", i);
        if (!upload_perm(model.upsample[i].conv_w, model.upsample[i].conv_w_perm, label)) {
            return false;
        }
    }

    for (int i = 0; i < 4; ++i) {
        char label[64];
        snprintf(label, sizeof(label), "tok_dec.dec.%d.conv_t.weight", i + 1);
        if (!upload_perm(model.dec_blocks[i].conv_t_w, model.dec_blocks[i].conv_t_w_perm, label)) {
            return false;
        }
    }

    return true;
}

} // namespace qwen3_tts
