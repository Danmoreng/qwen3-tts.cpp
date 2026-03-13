#include "audio_tokenizer_encoder.h"
#include "encoder/encoder_state_internal.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>

namespace qwen3_tts {

bool AudioTokenizerEncoder::load_model(const std::string & model_path) {
    auto & model = impl_->model;
    auto & state = impl_->state;

    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }

    model.config.sample_rate = loader.get_u32("qwen3-tts.speaker_encoder.sample_rate", 24000);
    model.config.embedding_dim = loader.get_u32("qwen3-tts.speaker_encoder.embedding_length", 1024);

    const int64_t n_tensors = loader.get_n_tensors();
    int spk_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "spk_enc.", 8) == 0) {
            spk_tensor_count++;
        }
    }

    if (spk_tensor_count == 0) {
        error_msg_ = "No speaker encoder tensors found in model";
        return false;
    }

    const size_t ctx_size = ggml_tensor_overhead() * spk_tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    model.ctx = ggml_init(params);
    if (!model.ctx) {
        error_msg_ = "Failed to initialize GGML context";
        return false;
    }

    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "spk_enc.", 8) != 0) {
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
        if (sname == "spk_enc.conv0.weight") {
            model.conv0_w = tensor;
        } else if (sname == "spk_enc.conv0.bias") {
            model.conv0_b = tensor;
        } else if (sname == "spk_enc.mfa.weight") {
            model.mfa_w = tensor;
        } else if (sname == "spk_enc.mfa.bias") {
            model.mfa_b = tensor;
        } else if (sname == "spk_enc.asp.conv.weight") {
            model.asp_conv_w = tensor;
        } else if (sname == "spk_enc.asp.conv.bias") {
            model.asp_conv_b = tensor;
        } else if (sname == "spk_enc.asp.tdnn.weight") {
            model.asp_tdnn_w = tensor;
        } else if (sname == "spk_enc.asp.tdnn.bias") {
            model.asp_tdnn_b = tensor;
        } else if (sname == "spk_enc.fc.weight") {
            model.fc_w = tensor;
        } else if (sname == "spk_enc.fc.bias") {
            model.fc_b = tensor;
        } else {
            int blk_idx = 0;
            int res_idx = 0;
            char suffix[64];

            if (sscanf(name, "spk_enc.blk.%d.tdnn1.%63s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model.blocks[blk_idx - 1].tdnn1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model.blocks[blk_idx - 1].tdnn1_b = tensor;
                }
            } else if (sscanf(name, "spk_enc.blk.%d.tdnn2.%63s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model.blocks[blk_idx - 1].tdnn2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model.blocks[blk_idx - 1].tdnn2_b = tensor;
                }
            } else if (sscanf(name, "spk_enc.blk.%d.res2net.%d.%63s", &blk_idx, &res_idx, suffix) == 3) {
                if (blk_idx >= 1 && blk_idx <= 3 && res_idx >= 0 && res_idx < 7) {
                    if (strcmp(suffix, "weight") == 0) model.blocks[blk_idx - 1].res2net_w[res_idx] = tensor;
                    else if (strcmp(suffix, "bias") == 0) model.blocks[blk_idx - 1].res2net_b[res_idx] = tensor;
                }
            } else if (sscanf(name, "spk_enc.blk.%d.se.conv1.%63s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model.blocks[blk_idx - 1].se_conv1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model.blocks[blk_idx - 1].se_conv1_b = tensor;
                }
            } else if (sscanf(name, "spk_enc.blk.%d.se.conv2.%63s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model.blocks[blk_idx - 1].se_conv2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model.blocks[blk_idx - 1].se_conv2_b = tensor;
                }
            }
        }
    }

    if (!load_tensor_data_from_file(model_path, gguf_ctx, model.ctx,
                                    model.tensors, model.buffer, error_msg_)) {
        return false;
    }

    state.backend = init_preferred_backend("AudioTokenizerEncoder", &error_msg_);
    if (!state.backend) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(state.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  AudioTokenizerEncoder backend: %s\n", device_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for AudioTokenizerEncoder";
            return false;
        }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(state.backend);
    if (state.backend_cpu) {
        backends.push_back(state.backend_cpu);
    }
    state.sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(),
                                         encoder_internal::QWEN3_TTS_ENC_MAX_NODES, false, true);
    if (!state.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }

    state.compute_meta.resize(ggml_tensor_overhead() * encoder_internal::QWEN3_TTS_ENC_MAX_NODES +
                              ggml_graph_overhead());
    encoder_internal::ops::init_frontend_cache(*this);

    return true;
}

} // namespace qwen3_tts
