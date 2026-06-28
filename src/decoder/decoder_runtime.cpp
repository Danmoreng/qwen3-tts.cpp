#include "audio_tokenizer_decoder.h"
#include "decoder/decoder_state_internal.h"

#include <chrono>

namespace qwen3_tts {

namespace {

int64_t now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        clock::now().time_since_epoch()).count();
}

bool backend_requires_decode_graph_rebuild(ggml_backend_t backend) {
    ggml_backend_dev_t device = backend ? ggml_backend_get_device(backend) : nullptr;
    return device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU;
}

} // namespace

bool AudioTokenizerDecoder::decode(const int32_t * codes, int32_t n_frames,
                                    std::vector<float> & samples) {
    auto & model = impl_->model;
    auto & state = impl_->state;
    auto & error_msg = impl_->error_msg;
    auto & codebook_input_bufs = impl_->codebook_input_bufs;
    auto & positions_buf = impl_->positions_buf;
    auto & timing = impl_->last_timing;

    timing = {};
    timing.n_frames = n_frames;
    const int64_t t_total_start = now_ms();

    if (!model.ctx) {
        error_msg = "Model not loaded";
        return false;
    }

    const auto & cfg = model.config;

    if (!decoder_internal::ops::ensure_cached_decode_graph(*this, n_frames)) {
        return false;
    }

    struct ggml_cgraph * gf = state.decode_graph;

    const int64_t t_alloc_start = now_ms();
    if (!ggml_backend_sched_alloc_graph(state.sched, gf)) {
        error_msg = "Failed to allocate graph";
        return false;
    }
    timing.graph_alloc_ms = now_ms() - t_alloc_start;

    const int64_t t_upload_start = now_ms();
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
    timing.input_upload_ms = now_ms() - t_upload_start;

    const int64_t t_compute_start = now_ms();
    if (ggml_backend_sched_graph_compute(state.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg = "Failed to compute graph";
        ggml_backend_sched_reset(state.sched);
        return false;
    }
    timing.graph_compute_ms = now_ms() - t_compute_start;

    struct ggml_tensor * audio_tensor = state.decode_audio_tensor;
    if (!audio_tensor) {
        error_msg = "Failed to find audio tensor";
        ggml_backend_sched_reset(state.sched);
        return false;
    }

    int64_t n_samples = audio_tensor->ne[0];
    samples.resize(n_samples);
    const int64_t t_read_start = now_ms();
    ggml_backend_tensor_get(audio_tensor, samples.data(), 0, n_samples * sizeof(float));
    timing.output_read_ms = now_ms() - t_read_start;
    timing.n_samples = n_samples;

    ggml_backend_sched_reset(state.sched);
    timing.total_ms = now_ms() - t_total_start;

    if (backend_requires_decode_graph_rebuild(state.backend)) {
        decoder_internal::ops::release_cached_decode_graph(*this);
    }

    return true;
}

void AudioTokenizerDecoder::clear_decode_cache() {
    decoder_internal::ops::release_cached_decode_graph(*this);
}

} // namespace qwen3_tts
