#include "audio_tokenizer_decoder.h"
#include "decoder/decoder_state_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>

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
    auto & mask_buf = impl_->mask_buf;
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
    if ((int32_t) mask_buf.size() != n_frames * n_frames) {
        mask_buf.assign((size_t) n_frames * (size_t) n_frames, -INFINITY);
        const int32_t window = cfg.sliding_window > 0 ? cfg.sliding_window : n_frames;
        for (int32_t q = 0; q < n_frames; ++q) {
            int32_t k_min = q - window + 1;
            if (k_min < 0) {
                k_min = 0;
            }
            for (int32_t k = k_min; k <= q; ++k) {
                mask_buf[(size_t) q * (size_t) n_frames + (size_t) k] = 0.0f;
            }
        }
    }
    if (state.decode_mask_tensor) {
        ggml_backend_tensor_set(state.decode_mask_tensor, mask_buf.data(), 0,
                                (size_t) n_frames * (size_t) n_frames * sizeof(float));
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

bool AudioTokenizerDecoder::reset_stream() {
    auto & state = impl_->state;
    if (!decoder_internal::ops::ensure_stream_state(*this)) {
        return false;
    }
    ggml_backend_buffer_clear(state.stream_buffer, 0);
    state.stream_pos = 0;
    return true;
}

bool AudioTokenizerDecoder::prime_stream(const int32_t * codes, int32_t n_frames,
                                         std::vector<float> & scratch_samples) {
    auto & state = impl_->state;
    if (!codes || n_frames <= 0) {
        return true;
    }
    if (!decoder_internal::ops::ensure_stream_state(*this)) {
        return false;
    }
    uint64_t key = 1469598103934665603ULL;
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(codes);
    const size_t byte_count = (size_t) n_frames * impl_->model.config.n_codebooks * sizeof(int32_t);
    for (size_t i = 0; i < byte_count; ++i) {
        key = (key ^ bytes[i]) * 1099511628211ULL;
    }
    key = (key ^ (uint64_t) n_frames) * 1099511628211ULL;

    auto copy_state = [](struct ggml_context * src_ctx, struct ggml_context * dst_ctx) {
        struct ggml_tensor * dst = ggml_get_first_tensor(dst_ctx);
        for (struct ggml_tensor * src = ggml_get_first_tensor(src_ctx);
             src && dst;
             src = ggml_get_next_tensor(src_ctx, src), dst = ggml_get_next_tensor(dst_ctx, dst)) {
            ggml_backend_tensor_copy(src, dst);
        }
    };
    if (state.stream_snapshot_valid && state.stream_snapshot_key == key) {
        const int64_t restore_start = now_ms();
        copy_state(state.stream_snapshot_ctx, state.stream_ctx);
        state.stream_pos = state.stream_snapshot_pos;
        scratch_samples.clear();
        impl_->last_timing = {};
        impl_->last_timing.n_frames = n_frames;
        impl_->last_timing.total_ms = now_ms() - restore_start;
        return true;
    }
    if (!decode_stream(codes, n_frames, scratch_samples)) {
        return false;
    }

    if (!state.stream_snapshot_ctx) {
        int tensor_count = 0;
        for (struct ggml_tensor * tensor = ggml_get_first_tensor(state.stream_ctx);
             tensor;
             tensor = ggml_get_next_tensor(state.stream_ctx, tensor)) {
            ++tensor_count;
        }
        struct ggml_init_params params = {
            ggml_tensor_overhead() * (size_t) (tensor_count + 4), nullptr, true,
        };
        state.stream_snapshot_ctx = ggml_init(params);
        if (state.stream_snapshot_ctx) {
            for (struct ggml_tensor * tensor = ggml_get_first_tensor(state.stream_ctx);
                 tensor;
                 tensor = ggml_get_next_tensor(state.stream_ctx, tensor)) {
                ggml_dup_tensor(state.stream_snapshot_ctx, tensor);
            }
            state.stream_snapshot_buffer = ggml_backend_alloc_ctx_tensors(
                state.stream_snapshot_ctx, state.backend);
            if (!state.stream_snapshot_buffer) {
                ggml_free(state.stream_snapshot_ctx);
                state.stream_snapshot_ctx = nullptr;
            } else {
                fprintf(stderr, "  Stateful decoder reference snapshot: %.2f MiB\n",
                        (double) ggml_backend_buffer_get_size(state.stream_snapshot_buffer) /
                            (1024.0 * 1024.0));
            }
        }
    }
    if (state.stream_snapshot_ctx && state.stream_snapshot_buffer) {
        copy_state(state.stream_ctx, state.stream_snapshot_ctx);
        state.stream_snapshot_key = key;
        state.stream_snapshot_pos = state.stream_pos;
        state.stream_snapshot_valid = true;
        // Reference priming favors wide graphs, but normal low-latency streaming
        // only uses widths up to four. Do not retain the one-shot wide scratch
        // allocations after the resident snapshot has been captured.
        decoder_internal::ops::release_stream_graphs_above(*this, 4);
    }
    return true;
}

bool AudioTokenizerDecoder::decode_stream(const int32_t * codes, int32_t n_frames,
                                          std::vector<float> & samples) {
    auto & state = impl_->state;
    auto & error_msg = impl_->error_msg;
    auto & timing = impl_->last_timing;
    const auto & cfg = impl_->model.config;
    if (!codes || n_frames <= 0) {
        error_msg = "Streaming decoder requires at least one codec frame";
        return false;
    }
    if (!decoder_internal::ops::ensure_stream_state(*this)) {
        return false;
    }
    timing = {};
    timing.n_frames = n_frames;
    const int64_t total_start = now_ms();
    samples.clear();
    samples.reserve((size_t) n_frames * 1920);

    int32_t frame_offset = 0;
    while (frame_offset < n_frames) {
        const int32_t width = std::min<int32_t>(32, n_frames - frame_offset);
        const bool graph_missing = state.stream_graphs.find(width) == state.stream_graphs.end();
        const int64_t build_start = now_ms();
        if (!decoder_internal::ops::ensure_stream_graph(*this, width)) {
            return false;
        }
        timing.graph_build_ms += now_ms() - build_start;
        timing.graph_rebuilt += graph_missing ? 1 : 0;
        auto & graph = state.stream_graphs[width];

        const int64_t upload_start = now_ms();
        impl_->stream_codes_buf.resize((size_t) width * cfg.n_codebooks);
        for (int cb = 0; cb < cfg.n_codebooks; ++cb) {
            for (int32_t t = 0; t < width; ++t) {
                impl_->stream_codes_buf[(size_t) cb * width + t] =
                    codes[(size_t) (frame_offset + t) * cfg.n_codebooks + cb];
            }
        }
        impl_->stream_positions_buf.resize((size_t) width);
        impl_->stream_rows_buf.resize((size_t) width);
        for (int32_t t = 0; t < width; ++t) {
            impl_->stream_positions_buf[(size_t) t] = state.stream_pos + t;
            impl_->stream_rows_buf[(size_t) t] = (state.stream_pos + t) % state.stream_ring;
        }
        impl_->stream_mask_buf.assign((size_t) state.stream_ring * width, -INFINITY);
        for (int32_t q = 0; q < width; ++q) {
            const int32_t pos = state.stream_pos + q;
            const int32_t first = std::max<int32_t>(0, pos - cfg.sliding_window + 1);
            for (int32_t p = first; p <= pos; ++p) {
                impl_->stream_mask_buf[(size_t) q * state.stream_ring +
                                       (size_t) (p % state.stream_ring)] = 0.0f;
            }
        }
        ggml_backend_tensor_set(graph.codes, impl_->stream_codes_buf.data(), 0,
                                impl_->stream_codes_buf.size() * sizeof(int32_t));
        ggml_backend_tensor_set(graph.positions, impl_->stream_positions_buf.data(), 0,
                                impl_->stream_positions_buf.size() * sizeof(int32_t));
        ggml_backend_tensor_set(graph.rows, impl_->stream_rows_buf.data(), 0,
                                impl_->stream_rows_buf.size() * sizeof(int64_t));
        ggml_backend_tensor_set(graph.mask, impl_->stream_mask_buf.data(), 0,
                                impl_->stream_mask_buf.size() * sizeof(float));
        timing.input_upload_ms += now_ms() - upload_start;

        const int64_t compute_start = now_ms();
        if (ggml_backend_graph_compute(state.backend, graph.gf) != GGML_STATUS_SUCCESS) {
            error_msg = "Failed to compute stateful streaming decoder graph";
            return false;
        }
        timing.graph_compute_ms += now_ms() - compute_start;

        const size_t old_size = samples.size();
        const size_t chunk_samples = (size_t) ggml_nelements(graph.audio);
        samples.resize(old_size + chunk_samples);
        const int64_t read_start = now_ms();
        ggml_backend_tensor_get(graph.audio, samples.data() + old_size, 0,
                                chunk_samples * sizeof(float));
        timing.output_read_ms += now_ms() - read_start;
        state.stream_pos += width;
        frame_offset += width;
    }
    timing.n_samples = (int64_t) samples.size();
    timing.total_ms = now_ms() - total_start;
    return true;
}

void AudioTokenizerDecoder::clear_decode_cache() {
    decoder_internal::ops::release_cached_decode_graph(*this);
}

} // namespace qwen3_tts
