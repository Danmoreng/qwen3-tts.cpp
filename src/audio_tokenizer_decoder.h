#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace qwen3_tts {

struct audio_decoder_private;
namespace decoder_internal {
struct ops;
}

// Audio tokenizer decoder (vocoder) configuration
struct audio_decoder_config {
    int32_t sample_rate = 24000;
    int32_t n_codebooks = 16;           // Total codebooks (1 first + 15 rest)
    int32_t codebook_size = 2048;       // Entries per codebook
    int32_t codebook_dim = 256;         // Embedding dimension per codebook
    int32_t latent_dim = 1024;          // Latent dimension after VQ
    int32_t hidden_dim = 512;           // Pre-transformer hidden dimension
    int32_t n_pre_tfm_layers = 8;       // Pre-transformer layers
    int32_t n_heads = 16;               // Attention heads in pre-transformer
    int32_t ffn_dim = 1024;             // FFN intermediate dimension
    int32_t decoder_dim = 1536;         // Initial decoder dimension
    int32_t upsample_rates[4] = {8, 5, 4, 3};  // Total: 480x upsampling
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
};

// Audio tokenizer decoder (vocoder) class
// Decodes discrete audio codes to waveform
class AudioTokenizerDecoder {
public:
    AudioTokenizerDecoder();
    ~AudioTokenizerDecoder();
    
    // Load model from GGUF file (tokenizer model)
    bool load_model(const std::string & model_path);

    // Release all model/runtime resources
    void unload_model();
    
    // Decode audio codes to waveform
    // codes: audio codes [n_frames, n_codebooks] as int32_t (row-major)
    // n_frames: number of frames
    // Returns: audio samples normalized to [-1, 1] at 24kHz
    bool decode(const int32_t * codes, int32_t n_frames,
                std::vector<float> & samples);
    
    const audio_decoder_config & get_config() const;
    
    const std::string & get_error() const;
    
private:
    friend struct decoder_internal::ops;

    std::unique_ptr<audio_decoder_private> impl_;
};

} // namespace qwen3_tts
