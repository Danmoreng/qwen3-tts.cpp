#include "qwen3_tts.h"
#include "gguf_loader.h"

namespace qwen3_tts {

Qwen3TTS::Qwen3TTS() = default;

Qwen3TTS::~Qwen3TTS() = default;

void Qwen3TTS::apply_n_threads(int32_t n_threads) {
    n_threads_ = n_threads > 0 ? n_threads : get_default_thread_count();
    transformer_.set_n_threads(n_threads_);
    audio_encoder_.set_n_threads(n_threads_);
    audio_decoder_.set_n_threads(n_threads_);
}

void Qwen3TTS::set_progress_callback(tts_progress_callback_t callback) {
    progress_callback_ = callback;
}

} // namespace qwen3_tts
