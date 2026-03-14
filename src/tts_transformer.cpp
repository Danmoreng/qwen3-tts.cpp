#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "gguf_loader.h"

namespace qwen3_tts {

TTSTransformer::TTSTransformer()
    : impl_(std::make_unique<tts_transformer_private>()) {
}

TTSTransformer::~TTSTransformer() {
    unload_model();
}

const tts_transformer_config & TTSTransformer::get_config() const {
    return impl_->model.config;
}

void TTSTransformer::set_n_threads(int32_t n_threads) {
    impl_->n_threads = n_threads > 0 ? n_threads : get_default_thread_count();
    apply_backend_thread_count(impl_->state.backend, impl_->n_threads);
    apply_backend_thread_count(impl_->state.backend_cpu, impl_->n_threads);
}

bool TTSTransformer::forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                             std::vector<float> & output) {
    return forward_text(tokens, n_tokens, nullptr, n_past, output);
}

bool TTSTransformer::forward_with_audio(const int32_t * tokens, int32_t n_tokens,
                                        const float * audio_embd, int32_t n_audio,
                                        int32_t audio_start_pos, int32_t n_past,
                                        std::vector<float> & output) {
    (void) audio_embd;
    (void) n_audio;
    (void) audio_start_pos;
    return forward_text(tokens, n_tokens, nullptr, n_past, output);
}

void free_transformer_model(tts_transformer_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
    model.layers.clear();
    model.code_pred_layers.clear();
    model.code_pred_small_to_mtp_weight = nullptr;
    model.code_pred_small_to_mtp_bias = nullptr;
    model.code_pred_output_norm = nullptr;
    model.code_pred_embd.clear();
    model.code_pred_head.clear();
}

void free_tts_kv_cache(tts_kv_cache & cache) {
    if (cache.buffer) {
        ggml_backend_buffer_free(cache.buffer);
        cache.buffer = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.n_ctx = 0;
    cache.n_used = 0;
}

} // namespace qwen3_tts
