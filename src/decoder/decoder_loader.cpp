#include "audio_tokenizer_decoder.h"
#include "decoder/decoder_state_internal.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>

namespace qwen3_tts {

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

void AudioTokenizerDecoder::unload_model() {
    auto & model = impl_->model;
    auto & state = impl_->state;
    auto & codes_buf = impl_->codes_buf;
    auto & codebook_input_bufs = impl_->codebook_input_bufs;
    auto & positions_buf = impl_->positions_buf;

    decoder_internal::ops::release_cached_decode_graph(*this);
    free_audio_decoder_model(model);

    if (state.sched) {
        ggml_backend_sched_free(state.sched);
        state.sched = nullptr;
    }
    if (state.backend) {
        release_preferred_backend(state.backend);
        state.backend = nullptr;
    }
    if (state.backend_cpu) {
        ggml_backend_free(state.backend_cpu);
        state.backend_cpu = nullptr;
    }

    state.compute_meta.clear();
    codes_buf.clear();
    codebook_input_bufs.clear();
    positions_buf.clear();
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

bool AudioTokenizerDecoder::load_model(const std::string & model_path) {
    auto & model = impl_->model;
    auto & state = impl_->state;
    auto & error_msg = impl_->error_msg;

    unload_model();

    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg = loader.get_error();
        return false;
    }

    model.config.sample_rate = loader.get_u32("qwen3-tts.tokenizer.sample_rate", 24000);
    model.config.n_codebooks = loader.get_u32("qwen3-tts.tokenizer.num_codebooks", 16);
    model.config.codebook_size = loader.get_u32("qwen3-tts.tokenizer.codebook_size", 2048);

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

    const size_t ctx_size = ggml_tensor_overhead() * dec_tensor_count;
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
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model.dec_blocks[blk_idx - 1].res[res_idx - 2].act1_alpha = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.act1.beta", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model.dec_blocks[blk_idx - 1].res[res_idx - 2].act1_beta = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.weight", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model.dec_blocks[blk_idx - 1].res[res_idx - 2].conv1_w = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.conv1.bias", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model.dec_blocks[blk_idx - 1].res[res_idx - 2].conv1_b = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.act2.alpha", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model.dec_blocks[blk_idx - 1].res[res_idx - 2].act2_alpha = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.act2.beta", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model.dec_blocks[blk_idx - 1].res[res_idx - 2].act2_beta = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.weight", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model.dec_blocks[blk_idx - 1].res[res_idx - 2].conv2_w = tensor;
                }
            } else if (MATCH2("tok_dec.dec.%d.res.%d.conv2.bias", blk_idx, res_idx)) {
                if (blk_idx >= 1 && blk_idx <= 4 && res_idx >= 2 && res_idx <= 4) {
                    model.dec_blocks[blk_idx - 1].res[res_idx - 2].conv2_b = tensor;
                }
            }

            #undef MATCH1
            #undef MATCH2
            #undef MATCH1S
        }
    }

    if (!load_tensor_data_from_file(model_path, gguf_ctx, model.ctx,
                                    model.tensors, model.buffer, error_msg,
                                    GGML_BACKEND_DEVICE_TYPE_IGPU)) {
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

    state.backend = init_preferred_backend("AudioTokenizerDecoder", &error_msg);
    if (!state.backend) {
        return false;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(state.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  AudioTokenizerDecoder backend: %s\n", device_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
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

} // namespace qwen3_tts
