#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qwen3_tts {

struct speech_codes;
struct speech_tokenizer_encoder_private;

struct speech_tokenizer_encoder_config {
    int32_t sample_rate = 24000;
    float frame_rate = 12.5f;
    int32_t hidden_size = 512;
    int32_t n_layers = 8;
    int32_t n_heads = 8;
    int32_t head_dim = 64;
    int32_t codebook_dim = 256;
    int32_t codebook_size = 2048;
    int32_t n_quantizers = 32;
    int32_t n_valid_quantizers = 16;
    int32_t sliding_window = 250;
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
};

class SpeechTokenizerEncoder {
public:
    SpeechTokenizerEncoder();
    ~SpeechTokenizerEncoder();

    bool load_model(const std::string & tokenizer_model_path);
    void unload_model();

    bool encode(const float * samples, int32_t n_samples, speech_codes & codes);
    bool project(const float * samples,
                 int32_t n_samples,
                 std::vector<float> & semantic_features,
                 std::vector<float> & acoustic_features,
                 int32_t & n_frames);
    bool quantize_projected(const float * semantic_features,
                            const float * acoustic_features,
                            int32_t n_frames,
                            speech_codes & codes);

    const speech_tokenizer_encoder_config & get_config() const;
    const std::string & get_error() const;

private:
    std::unique_ptr<speech_tokenizer_encoder_private> impl_;
    std::string error_msg_;
};

} // namespace qwen3_tts
