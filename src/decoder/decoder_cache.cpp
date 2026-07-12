#include "audio_tokenizer_decoder.h"
#include "decoder/decoder_state_internal.h"

#include <chrono>
#include <cstdio>

namespace qwen3_tts {

namespace {

int64_t now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        clock::now().time_since_epoch()).count();
}

} // namespace

void decoder_internal::ops::release_stream(AudioTokenizerDecoder & self) {
    auto & state = self.impl_->state;
    for (auto & entry : state.stream_graphs) {
        auto & graph = entry.second;
        if (graph.galloc) {
            ggml_gallocr_free(graph.galloc);
            graph.galloc = nullptr;
        }
        if (graph.ctx) {
            ggml_free(graph.ctx);
            graph.ctx = nullptr;
        }
    }
    state.stream_graphs.clear();
    if (state.stream_snapshot_buffer) {
        ggml_backend_buffer_free(state.stream_snapshot_buffer);
        state.stream_snapshot_buffer = nullptr;
    }
    if (state.stream_snapshot_ctx) {
        ggml_free(state.stream_snapshot_ctx);
        state.stream_snapshot_ctx = nullptr;
    }
    if (state.stream_buffer) {
        ggml_backend_buffer_free(state.stream_buffer);
        state.stream_buffer = nullptr;
    }
    if (state.stream_ctx) {
        ggml_free(state.stream_ctx);
        state.stream_ctx = nullptr;
    }
    state.stream_k.clear();
    state.stream_v.clear();
    state.stream_pre_conv = nullptr;
    state.stream_dec_pre = nullptr;
    state.stream_dec_post = nullptr;
    for (int i = 0; i < 2; ++i) {
        state.stream_upsample_dw[i] = nullptr;
    }
    for (int i = 0; i < 4; ++i) {
        state.stream_dec_carry[i] = nullptr;
        for (int j = 0; j < 3; ++j) {
            state.stream_dec_res[i][j] = nullptr;
        }
    }
    state.stream_pos = 0;
    state.stream_snapshot_key = 0;
    state.stream_snapshot_pos = 0;
    state.stream_snapshot_valid = false;
    state.stream_ready = false;
}

void decoder_internal::ops::release_stream_graphs_above(AudioTokenizerDecoder & self,
                                                         int32_t max_frames) {
    auto & graphs = self.impl_->state.stream_graphs;
    for (auto it = graphs.begin(); it != graphs.end();) {
        if (it->first <= max_frames) {
            ++it;
            continue;
        }
        if (it->second.galloc) {
            ggml_gallocr_free(it->second.galloc);
        }
        if (it->second.ctx) {
            ggml_free(it->second.ctx);
        }
        it = graphs.erase(it);
    }
}

bool decoder_internal::ops::ensure_stream_state(AudioTokenizerDecoder & self) {
    auto & impl = self.impl_;
    auto & state = impl->state;
    auto & model = impl->model;
    auto & error_msg = impl->error_msg;
    const auto & cfg = model.config;
    if (state.stream_ready) {
        return true;
    }
    if (!state.backend || cfg.sliding_window + 32 > state.stream_ring) {
        error_msg = "Streaming decoder KV ring is too small";
        return false;
    }

    const int n_state_tensors = 1 + 2 + 1 + 4 + 12 + 1 + 2 * cfg.n_pre_tfm_layers;
    struct ggml_init_params params = {
        ggml_tensor_overhead() * (size_t) (n_state_tensors + 8), nullptr, true,
    };
    state.stream_ctx = ggml_init(params);
    if (!state.stream_ctx) {
        error_msg = "Failed to initialize streaming decoder state";
        return false;
    }
    auto tensor2d = [&](int64_t frames, int64_t channels, const char * name) {
        struct ggml_tensor * tensor = ggml_new_tensor_2d(
            state.stream_ctx, GGML_TYPE_F32, frames, channels);
        ggml_set_name(tensor, name);
        return tensor;
    };

    state.stream_pre_conv = tensor2d(2, cfg.hidden_dim, "stream_pre_conv");
    for (int i = 0; i < 2; ++i) {
        char name[48];
        snprintf(name, sizeof(name), "stream_upsample_dw_%d", i);
        state.stream_upsample_dw[i] = tensor2d(6, cfg.latent_dim, name);
    }
    state.stream_dec_pre = tensor2d(6, cfg.latent_dim, "stream_dec_pre");
    int final_channels = cfg.decoder_dim;
    for (int i = 0; i < 4; ++i) {
        const decoder_block & block = model.dec_blocks[i];
        const int out_channels = (int) block.conv_t_w->ne[1];
        const int kernel = (int) block.conv_t_w->ne[0];
        const int carry = kernel - cfg.upsample_rates[i];
        char name[48];
        snprintf(name, sizeof(name), "stream_dec_carry_%d", i);
        state.stream_dec_carry[i] = tensor2d(carry, out_channels, name);
        for (int r = 0; r < 3; ++r) {
            snprintf(name, sizeof(name), "stream_dec_res_%d_%d", i, r);
            state.stream_dec_res[i][r] = tensor2d(
                6 * block.res[r].dilation, out_channels, name);
        }
        final_channels = out_channels;
    }
    state.stream_dec_post = tensor2d(6, final_channels, "stream_dec_post");

    const int head_dim = cfg.latent_dim / cfg.n_heads;
    state.stream_k.resize((size_t) cfg.n_pre_tfm_layers);
    state.stream_v.resize((size_t) cfg.n_pre_tfm_layers);
    for (int i = 0; i < cfg.n_pre_tfm_layers; ++i) {
        state.stream_k[(size_t) i] = ggml_new_tensor_3d(
            state.stream_ctx, GGML_TYPE_F32, head_dim, state.stream_ring, cfg.n_heads);
        state.stream_v[(size_t) i] = ggml_new_tensor_3d(
            state.stream_ctx, GGML_TYPE_F32, head_dim, state.stream_ring, cfg.n_heads);
        char name[48];
        snprintf(name, sizeof(name), "stream_k_%d", i);
        ggml_set_name(state.stream_k[(size_t) i], name);
        snprintf(name, sizeof(name), "stream_v_%d", i);
        ggml_set_name(state.stream_v[(size_t) i], name);
    }

    state.stream_buffer = ggml_backend_alloc_ctx_tensors(state.stream_ctx, state.backend);
    if (!state.stream_buffer) {
        error_msg = "Failed to allocate streaming decoder state";
        release_stream(self);
        return false;
    }
    ggml_backend_buffer_clear(state.stream_buffer, 0);
    fprintf(stderr, "  Stateful decoder state: %.2f MiB\n",
            (double) ggml_backend_buffer_get_size(state.stream_buffer) / (1024.0 * 1024.0));
    state.stream_pos = 0;
    state.stream_ready = true;
    return true;
}

void decoder_internal::ops::release_cached_decode_graph(AudioTokenizerDecoder & self) {
    auto & state = self.impl_->state;

    state.decode_graph = nullptr;
    state.decode_positions_tensor = nullptr;
    state.decode_mask_tensor = nullptr;
    state.decode_audio_tensor = nullptr;
    state.decode_graph_n_frames = 0;
    for (int i = 0; i < 16; ++i) {
        state.decode_code_tensors[i] = nullptr;
    }
    if (state.decode_graph_ctx) {
        ggml_free(state.decode_graph_ctx);
        state.decode_graph_ctx = nullptr;
    }
}

bool decoder_internal::ops::ensure_cached_decode_graph(AudioTokenizerDecoder & self, int32_t n_frames) {
    auto & state = self.impl_->state;
    auto & error_msg = self.impl_->error_msg;

    if (state.decode_graph && state.decode_graph_n_frames == n_frames) {
        return true;
    }

    release_cached_decode_graph(self);

    const int64_t t_build_start = now_ms();
    state.decode_graph = build_graph_impl(self, n_frames, &state.decode_graph_ctx);
    self.impl_->last_timing.graph_build_ms += now_ms() - t_build_start;
    self.impl_->last_timing.graph_rebuilt = 1;
    if (!state.decode_graph || !state.decode_graph_ctx) {
        error_msg = "Failed to build cached decoder graph";
        release_cached_decode_graph(self);
        return false;
    }

    for (int cb = 0; cb < 16; ++cb) {
        char name[32];
        snprintf(name, sizeof(name), "codes_cb%d", cb);
        state.decode_code_tensors[cb] = ggml_graph_get_tensor(state.decode_graph, name);
        if (!state.decode_code_tensors[cb]) {
            error_msg = "Failed to find cached decoder input tensor for codebook " + std::to_string(cb);
            release_cached_decode_graph(self);
            return false;
        }
    }

    state.decode_positions_tensor = ggml_graph_get_tensor(state.decode_graph, "positions");
    state.decode_mask_tensor = ggml_graph_get_tensor(state.decode_graph, "mask");
    state.decode_audio_tensor = ggml_graph_get_tensor(state.decode_graph, "audio");
    if (!state.decode_positions_tensor || !state.decode_mask_tensor) {
        error_msg = "Failed to find cached decoder position/mask input tensor";
        release_cached_decode_graph(self);
        return false;
    }
    if (!state.decode_audio_tensor) {
        error_msg = "Failed to find cached decoder output tensor";
        release_cached_decode_graph(self);
        return false;
    }

    state.decode_graph_n_frames = n_frames;
    return true;
}

} // namespace qwen3_tts
