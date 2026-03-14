#include "audio_tokenizer_decoder.h"
#include "decoder/decoder_state_internal.h"

namespace qwen3_tts {

bool AudioTokenizerDecoder::decode(const int32_t * codes, int32_t n_frames,
                                    std::vector<float> & samples) {
    auto & model = impl_->model;
    auto & state = impl_->state;
    auto & error_msg = impl_->error_msg;
    auto & codebook_input_bufs = impl_->codebook_input_bufs;
    auto & positions_buf = impl_->positions_buf;

    if (!model.ctx) {
        error_msg = "Model not loaded";
        return false;
    }

    const auto & cfg = model.config;

    if (!decoder_internal::ops::ensure_cached_decode_graph(*this, n_frames)) {
        return false;
    }

    struct ggml_cgraph * gf = state.decode_graph;

    if (!ggml_backend_sched_alloc_graph(state.sched, gf)) {
        error_msg = "Failed to allocate graph";
        return false;
    }

    if ((int32_t) codebook_input_bufs.size() != cfg.n_codebooks) {
        codebook_input_bufs.assign(cfg.n_codebooks, {});
    }
    for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
        codebook_input_bufs[cb].resize(n_frames);
    }

    for (int f = 0; f < n_frames; ++f) {
        const int32_t * frame_codes = codes + (size_t) f * cfg.n_codebooks;
        for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
            codebook_input_bufs[cb][f] = frame_codes[cb];
        }
    }

    for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
        ggml_backend_tensor_set(state.decode_code_tensors[cb], codebook_input_bufs[cb].data(), 0,
                                (size_t) n_frames * sizeof(int32_t));
    }

    if ((int32_t) positions_buf.size() != n_frames) {
        positions_buf.resize(n_frames);
        for (int i = 0; i < n_frames; ++i) {
            positions_buf[i] = i;
        }
    }
    if (state.decode_positions_tensor) {
        ggml_backend_tensor_set(state.decode_positions_tensor, positions_buf.data(), 0,
                                (size_t) n_frames * sizeof(int32_t));
    }

    if (ggml_backend_sched_graph_compute(state.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg = "Failed to compute graph";
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    struct ggml_tensor * audio_tensor = state.decode_audio_tensor;
    if (!audio_tensor) {
        error_msg = "Failed to find audio tensor";
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    int64_t n_samples = audio_tensor->ne[0];
    samples.resize(n_samples);
    ggml_backend_tensor_get(audio_tensor, samples.data(), 0, n_samples * sizeof(float));

    ggml_backend_sched_reset(state.sched);

    return true;
}

} // namespace qwen3_tts
