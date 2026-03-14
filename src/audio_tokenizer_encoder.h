#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qwen3_tts {

struct speaker_encoder_model;
struct speaker_encoder_private;
namespace encoder_internal {
struct ops;
}

// Speaker encoder configuration (ECAPA-TDNN)
// Mel parameters MUST match extract_speaker_embedding() in modeling_qwen3_tts.py
struct speaker_encoder_config {
    int32_t sample_rate = 24000;
    int32_t n_mels = 128;
    int32_t n_fft = 1024;
    int32_t hop_length = 256;
    int32_t win_length = 1024;
    int32_t embedding_dim = 1024;
    int32_t hidden_dim = 512;
    int32_t n_res2net_blocks = 3;
    int32_t res2net_scale = 8;
    float f_min = 0.0f;
    float f_max = 12000.0f;
};

// Speaker encoder class (ECAPA-TDNN)
// Extracts speaker embedding from audio waveform
class AudioTokenizerEncoder {
public:
    AudioTokenizerEncoder();
    ~AudioTokenizerEncoder();

    // Load model from GGUF file (main TTS model, not tokenizer)
    bool load_model(const std::string & model_path);

    // Configure runtime thread count for CPU-capable backends.
    void set_n_threads(int32_t n_threads);

    // Encode audio samples to speaker embedding
    // samples: audio samples normalized to [-1, 1], 24kHz
    // n_samples: number of samples
    // embedding: output speaker embedding [1024]
    bool encode(const float * samples, int32_t n_samples,
                std::vector<float> & embedding);

    // Legacy interface for compatibility (not used for speaker encoding)
    bool encode(const float * samples, int32_t n_samples,
                std::vector<int32_t> & codes, int32_t & n_frames) {
        (void) samples;
        (void) n_samples;
        (void) codes;
        (void) n_frames;
        error_msg_ = "Use encode(samples, n_samples, embedding) for speaker encoding";
        return false;
    }

    // Legacy interface (not used)
    bool get_embeddings(const int32_t * codes, int32_t n_frames,
                        std::vector<float> & embeddings) {
        (void) codes;
        (void) n_frames;
        (void) embeddings;
        error_msg_ = "Use encode() for speaker embedding extraction";
        return false;
    }

    const speaker_encoder_config & get_config() const;

    const std::string & get_error() const { return error_msg_; }

private:
    friend struct encoder_internal::ops;

    std::unique_ptr<speaker_encoder_private> impl_;
    std::string error_msg_;
};

// Free model resources
void free_speaker_encoder_model(speaker_encoder_model & model);

// Backward compatibility aliases
using audio_encoder_config = speaker_encoder_config;
using audio_encoder_model = speaker_encoder_model;
inline void free_audio_encoder_model(audio_encoder_model & model) {
    free_speaker_encoder_model(model);
}

} // namespace qwen3_tts
