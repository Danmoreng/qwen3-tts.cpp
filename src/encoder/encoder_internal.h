#pragma once

#include <cstdint>
#include <vector>

struct ggml_cgraph;

namespace qwen3_tts {

class AudioTokenizerEncoder;

namespace encoder_internal {

inline constexpr int QWEN3_TTS_ENC_MAX_NODES = 16384;

struct ops {
    static void init_frontend_cache(AudioTokenizerEncoder & self);
    static bool compute_mel_spectrogram(AudioTokenizerEncoder & self,
                                        const float * samples,
                                        int32_t n_samples,
                                        std::vector<float> & mel,
                                        int32_t & n_frames);
    static struct ggml_cgraph * build_graph(AudioTokenizerEncoder & self, int32_t n_frames);
};

} // namespace encoder_internal
} // namespace qwen3_tts
