#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "gguf_loader.h"
#include "transformer/transformer_internal.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace qwen3_tts {

bool transformer_internal::ops::create_tensors(TTSTransformer & self, struct gguf_context * ctx) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const auto & cfg = impl->model.config;

    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    impl->model.ctx = ggml_init(params);
    if (!impl->model.ctx) {
        error_msg = "Failed to create GGML context";
        return false;
    }

    impl->model.layers.resize(cfg.n_layers);
    impl->model.code_pred_layers.resize(cfg.code_pred_layers);
    impl->model.code_pred_embd.resize(cfg.n_codebooks - 1);
    impl->model.code_pred_head.resize(cfg.n_codebooks - 1);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);

        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        int n_dims = 0;

        if (strstr(name, "spk_enc.") || strstr(name, "tok_")) {
            continue;
        }

        if (strstr(name, "talker.text_embd.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.text_embd_dim;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            ne[0] = cfg.text_embd_dim;
            n_dims = 1;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            ne[0] = cfg.text_embd_dim;
            ne[1] = cfg.hidden_size;
            n_dims = 2;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.codec_head.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.codec_vocab_size;
            n_dims = 2;
        } else if (strstr(name, "talker.output_norm.weight")) {
            ne[0] = cfg.hidden_size;
            n_dims = 1;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            if (sscanf(name, "talker.blk.%d.", &layer_idx) == 1 &&
                layer_idx >= 0 && layer_idx < cfg.n_layers) {
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_attention_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.n_key_value_heads * cfg.head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.n_attention_heads * cfg.head_dim;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.hidden_size;
                    ne[1] = cfg.intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.intermediate_size;
                    ne[1] = cfg.hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.small_to_mtp.weight")) {
            ne[0] = cfg.hidden_size;
            ne[1] = cfg.code_pred_hidden_size;
            n_dims = 2;
        } else if (strstr(name, "code_pred.small_to_mtp.bias")) {
            ne[0] = cfg.code_pred_hidden_size;
            n_dims = 1;
        } else if (strstr(name, "code_pred.blk.")) {
            if (impl->skip_ggml_code_pred_layers) {
                continue;
            }
            int layer_idx = -1;
            if (sscanf(name, "code_pred.blk.%d.", &layer_idx) == 1 &&
                layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                if (strstr(name, "attn_norm.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "attn_q_norm.weight")) {
                    ne[0] = cfg.code_pred_head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_k_norm.weight")) {
                    ne[0] = cfg.code_pred_head_dim;
                    n_dims = 1;
                } else if (strstr(name, "attn_q.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_attention_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_k.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_key_value_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_v.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_n_key_value_heads * cfg.code_pred_head_dim;
                    n_dims = 2;
                } else if (strstr(name, "attn_output.weight")) {
                    ne[0] = cfg.code_pred_n_attention_heads * cfg.code_pred_head_dim;
                    ne[1] = cfg.code_pred_hidden_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_norm.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    n_dims = 1;
                } else if (strstr(name, "ffn_gate.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_up.weight")) {
                    ne[0] = cfg.code_pred_hidden_size;
                    ne[1] = cfg.code_pred_intermediate_size;
                    n_dims = 2;
                } else if (strstr(name, "ffn_down.weight")) {
                    ne[0] = cfg.code_pred_intermediate_size;
                    ne[1] = cfg.code_pred_hidden_size;
                    n_dims = 2;
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            if (sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx) == 1 &&
                cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                ne[0] = cfg.hidden_size;
                ne[1] = cfg.code_pred_vocab_size;
                n_dims = 2;
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.lm_head.")) {
            if (impl->skip_ggml_code_pred_layers) {
                continue;
            }
            int cb_idx = -1;
            if (sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx) == 1 &&
                cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                ne[0] = cfg.code_pred_hidden_size;
                ne[1] = cfg.code_pred_vocab_size;
                n_dims = 2;
            } else {
                continue;
            }
        } else if (strstr(name, "code_pred.output_norm.weight")) {
            if (impl->skip_ggml_code_pred_layers) {
                continue;
            }
            ne[0] = cfg.code_pred_hidden_size;
            n_dims = 1;
        } else {
            continue;
        }

        struct ggml_tensor * tensor = ggml_new_tensor(impl->model.ctx, type, n_dims, ne);
        if (!tensor) {
            error_msg = "Failed to create tensor: " + std::string(name);
            return false;
        }
        ggml_set_name(tensor, name);
        impl->model.tensors[name] = tensor;

        if (strstr(name, "talker.text_embd.weight")) {
            impl->model.text_embd = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.weight")) {
            impl->model.text_proj_fc1 = tensor;
        } else if (strstr(name, "talker.text_proj.fc1.bias")) {
            impl->model.text_proj_fc1_bias = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.weight")) {
            impl->model.text_proj_fc2 = tensor;
        } else if (strstr(name, "talker.text_proj.fc2.bias")) {
            impl->model.text_proj_fc2_bias = tensor;
        } else if (strstr(name, "talker.codec_embd.weight")) {
            impl->model.codec_embd = tensor;
        } else if (strstr(name, "talker.codec_head.weight")) {
            impl->model.codec_head = tensor;
        } else if (strstr(name, "talker.output_norm.weight")) {
            impl->model.output_norm = tensor;
        } else if (strstr(name, "talker.blk.")) {
            int layer_idx = -1;
            sscanf(name, "talker.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.n_layers) {
                auto & layer = impl->model.layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.small_to_mtp.weight")) {
            impl->model.code_pred_small_to_mtp_weight = tensor;
        } else if (strstr(name, "code_pred.small_to_mtp.bias")) {
            impl->model.code_pred_small_to_mtp_bias = tensor;
        } else if (strstr(name, "code_pred.blk.")) {
            int layer_idx = -1;
            sscanf(name, "code_pred.blk.%d.", &layer_idx);
            if (layer_idx >= 0 && layer_idx < cfg.code_pred_layers) {
                auto & layer = impl->model.code_pred_layers[layer_idx];
                if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                else if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
            }
        } else if (strstr(name, "code_pred.codec_embd.")) {
            int cb_idx = -1;
            sscanf(name, "code_pred.codec_embd.%d.weight", &cb_idx);
            if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                impl->model.code_pred_embd[cb_idx] = tensor;
            }
        } else if (strstr(name, "code_pred.lm_head.")) {
            int cb_idx = -1;
            sscanf(name, "code_pred.lm_head.%d.weight", &cb_idx);
            if (cb_idx >= 0 && cb_idx < cfg.n_codebooks - 1) {
                impl->model.code_pred_head[cb_idx] = tensor;
            }
        } else if (strstr(name, "code_pred.output_norm.weight")) {
            impl->model.code_pred_output_norm = tensor;
        }
    }

    return true;
}

bool transformer_internal::ops::load_tensor_data(TTSTransformer & self, const std::string & path, struct gguf_context * ctx) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;
    ggml_backend_t backend = init_preferred_backend("TTSTransformer", &error_msg);
    if (!backend) {
        return false;
    }

    impl->model.buffer = ggml_backend_alloc_ctx_tensors(impl->model.ctx, backend);
    if (!impl->model.buffer) {
        error_msg = "Failed to allocate tensor buffer";
        release_preferred_backend(backend);
        return false;
    }

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg = "Failed to open file for reading: " + path;
        release_preferred_backend(backend);
        return false;
    }

    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);

        auto it = impl->model.tensors.find(name);
        if (it == impl->model.tensors.end()) {
            continue;
        }

        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);

        read_buf.resize(nbytes);

#ifdef _WIN32
        if (_fseeki64(f, (int64_t) data_offset + (int64_t) offset, SEEK_SET) != 0) {
#else
        if (fseek(f, data_offset + offset, SEEK_SET) != 0) {
#endif
            error_msg = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            release_preferred_backend(backend);
            return false;
        }

        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            release_preferred_backend(backend);
            return false;
        }

        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }

    fclose(f);
    release_preferred_backend(backend);

    return true;
}

} // namespace qwen3_tts
