#include "audio_tokenizer_encoder.h"
#include "encoder/encoder_state_internal.h"

namespace qwen3_tts {

bool AudioTokenizerEncoder::encode(const float * samples, int32_t n_samples,
                                   std::vector<float> & embedding) {
    if (!impl_->model.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }

    std::vector<float> mel;
    int32_t n_frames = 0;
    if (!encoder_internal::ops::compute_mel_spectrogram(*this, samples, n_samples, mel, n_frames)) {
        return false;
    }

    struct ggml_cgraph * gf = encoder_internal::ops::build_graph(*this, n_frames);
    if (!gf) {
        return false;
    }

    auto & state = impl_->state;
    if (!ggml_backend_sched_alloc_graph(state.sched, gf)) {
        error_msg_ = "Failed to allocate graph";
        return false;
    }

    struct ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf, "mel");
    if (!mel_tensor) {
        error_msg_ = "Failed to find mel tensor";
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    ggml_backend_tensor_set(mel_tensor, mel.data(), 0, mel.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(state.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute graph";
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    struct ggml_tensor * emb_tensor = ggml_graph_get_tensor(gf, "embedding");
    if (!emb_tensor) {
        error_msg_ = "Failed to find embedding tensor";
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    embedding.resize(impl_->model.config.embedding_dim);
    ggml_backend_tensor_get(emb_tensor, embedding.data(), 0, embedding.size() * sizeof(float));

    ggml_backend_sched_reset(state.sched);
    return true;
}

} // namespace qwen3_tts
