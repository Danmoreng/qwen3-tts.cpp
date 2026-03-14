#include "audio_tokenizer_decoder.h"
#include "decoder/decoder_state_internal.h"

#include <cstdio>

namespace qwen3_tts {

void decoder_internal::ops::release_cached_decode_graph(AudioTokenizerDecoder & self) {
    auto & state = self.impl_->state;

    state.decode_graph = nullptr;
    state.decode_positions_tensor = nullptr;
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

    state.decode_graph = build_graph_impl(self, n_frames, &state.decode_graph_ctx);
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
    state.decode_audio_tensor = ggml_graph_get_tensor(state.decode_graph, "audio");
    if (!state.decode_audio_tensor) {
        error_msg = "Failed to find cached decoder output tensor";
        release_cached_decode_graph(self);
        return false;
    }

    state.decode_graph_n_frames = n_frames;
    return true;
}

} // namespace qwen3_tts
