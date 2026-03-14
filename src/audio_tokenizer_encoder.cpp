#include "audio_tokenizer_encoder.h"
#include "encoder/encoder_state_internal.h"
#include "gguf_loader.h"

namespace qwen3_tts {

AudioTokenizerEncoder::AudioTokenizerEncoder()
    : impl_(std::make_unique<speaker_encoder_private>()) {
}

AudioTokenizerEncoder::~AudioTokenizerEncoder() {
    auto & state = impl_->state;

    free_speaker_encoder_model(impl_->model);

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
    state.mel_filterbank.clear();
    state.stft_window.clear();
}

const speaker_encoder_config & AudioTokenizerEncoder::get_config() const {
    return impl_->model.config;
}

void AudioTokenizerEncoder::set_n_threads(int32_t n_threads) {
    impl_->n_threads = n_threads > 0 ? n_threads : get_default_thread_count();
    apply_backend_thread_count(impl_->state.backend, impl_->n_threads);
    apply_backend_thread_count(impl_->state.backend_cpu, impl_->n_threads);
}

void free_speaker_encoder_model(speaker_encoder_model & model) {
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
