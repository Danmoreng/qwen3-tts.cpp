#pragma once

#include "text_tokenizer.h"
#include "tts_transformer.h"
#include "audio_tokenizer_encoder.h"
#include "audio_tokenizer_decoder.h"

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

namespace qwen3_tts {
namespace pipeline_internal {
struct ops;
}

enum class voice_clone_prompt_mode {
    audio_only = 0,
    reference_aware = 1,
};

struct voice_clone_prompt_asset {
    int32_t format_version = 1;
    voice_clone_prompt_mode prompt_mode = voice_clone_prompt_mode::audio_only;
    std::string model_kind;
    std::string model_name;
    int32_t speaker_embedding_dim = 0;
    std::vector<float> speaker_embedding;
    std::string reference_text;
    int32_t reference_codebooks = 0;
    int32_t reference_frames = 0;
    std::vector<int32_t> reference_codes;
};

struct voice_clone_prompt_validation {
    bool valid = false;
    bool model_compatible = false;
    std::string error_msg;
};

// TTS generation parameters
struct tts_params {
    // Maximum number of audio tokens to generate
    int32_t max_audio_tokens = 4096;
    
    // Temperature for sampling (0 = greedy)
    float temperature = 0.9f;
    
    // Top-p sampling
    float top_p = 1.0f;
    
    // Top-k sampling (0 = disabled)
    int32_t top_k = 50;
    
    // Number of threads
    int32_t n_threads = 4;
    
    // Print progress during generation
    bool print_progress = false;
    
    // Print timing information
    bool print_timing = true;
    
    // Repetition penalty for CB0 token generation (HuggingFace style)
    float repetition_penalty = 1.05f;

    // Language ID for codec (2050=en, 2069=ru, 2055=zh, 2058=ja, 2064=ko, 2053=de, 2061=fr, 2054=es)
    int32_t language_id = 2050;

    // Optional style/voice instruction
    std::string instruction;

    // Optional named speaker (for CustomVoice models)
    std::string speaker;
};

// TTS generation result
struct tts_result {
    // Generated audio samples (24kHz, mono)
    std::vector<float> audio;
    
    // Sample rate
    int32_t sample_rate = 24000;
    
    // Success flag
    bool success = false;
    
    // Error message if failed
    std::string error_msg;
    
    // Timing info (in milliseconds)
    int64_t t_load_ms = 0;
    int64_t t_tokenize_ms = 0;
    int64_t t_encode_ms = 0;
    int64_t t_generate_ms = 0;
    int64_t t_decode_ms = 0;
    int64_t t_total_ms = 0;

    // Process memory snapshots (bytes)
    uint64_t mem_rss_start_bytes = 0;
    uint64_t mem_rss_end_bytes = 0;
    uint64_t mem_rss_peak_bytes = 0;
    uint64_t mem_phys_start_bytes = 0;
    uint64_t mem_phys_end_bytes = 0;
    uint64_t mem_phys_peak_bytes = 0;
    
};

// Model capabilities inferred from loaded GGUF metadata.
struct tts_model_capabilities {
    bool loaded = false;
    bool supports_voice_clone = false;
    bool supports_named_speakers = false;
    bool supports_instruction = false;
    int32_t speaker_embedding_dim = 0;
    int32_t speaker_count = 0;
    std::string model_type;
};

// Progress callback type
using tts_progress_callback_t = std::function<void(int tokens_generated, int max_tokens)>;

// Main TTS class that orchestrates the full pipeline
class Qwen3TTS {
public:
    Qwen3TTS();
    ~Qwen3TTS();
    
    // Load all models from directory
    // model_dir should contain: transformer.gguf, tokenizer.gguf, vocoder.gguf
    bool load_models(const std::string & model_dir, const std::string & model_name = "");
    
    // Generate speech from text
    // text: input text to synthesize
    // params: generation parameters
    tts_result synthesize(const std::string & text,
                          const tts_params & params = tts_params());
    
    // Generate speech with voice cloning
    // text: input text to synthesize
    // reference_audio: path to reference audio file (WAV, 24kHz)
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const std::string & reference_audio,
                                      const tts_params & params = tts_params());
    
    // Generate speech with voice cloning from samples
    // text: input text to synthesize
    // ref_samples: reference audio samples (24kHz, mono, normalized to [-1, 1])
    // n_ref_samples: number of reference samples
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const float * ref_samples, int32_t n_ref_samples,
                                      const tts_params & params = tts_params());

    // Generate speech from a precomputed speaker embedding vector
    tts_result synthesize_with_speaker_embedding(const std::string & text,
                                                 const std::vector<float> & speaker_embedding,
                                                 const tts_params & params = tts_params());

    // Generate speech from a reusable voice clone prompt asset.
    tts_result synthesize_with_voice_clone_prompt(const std::string & text,
                                                  const voice_clone_prompt_asset & asset,
                                                  const tts_params & params = tts_params());

    // Create a reusable voice clone prompt asset. Audio-only mode stays native.
    // Reference-aware mode requires the helper script and upstream Python stack.
    bool create_voice_clone_prompt(const std::string & reference_audio,
                                   const std::string & reference_text,
                                   voice_clone_prompt_asset & asset);

    // Extract speaker embedding from reference audio file (WAV)
    bool extract_speaker_embedding(const std::string & reference_audio,
                                   std::vector<float> & speaker_embedding,
                                   int64_t * encode_time_ms = nullptr);

    bool validate_voice_clone_prompt(const voice_clone_prompt_asset & asset,
                                     voice_clone_prompt_validation * out = nullptr) const;
    
    // Set progress callback
    void set_progress_callback(tts_progress_callback_t callback);
    
    // Get error message
    const std::string & get_error() const { return error_msg_; }
    
    // Check if models are loaded
    bool is_loaded() const { return models_loaded_; }

    // List named speakers exposed by the currently loaded model metadata.
    // Returns normalized (lowercase) speaker keys; empty for non-CustomVoice models.
    std::vector<std::string> get_available_speakers() const;

    // Return feature flags for the currently loaded model.
    tts_model_capabilities get_model_capabilities() const;
    
private:
    friend struct pipeline_internal::ops;

    TextTokenizer tokenizer_;
    TTSTransformer transformer_;
    AudioTokenizerEncoder audio_encoder_;
    AudioTokenizerDecoder audio_decoder_;
    
    bool models_loaded_ = false;
    bool encoder_loaded_ = false;
    bool transformer_loaded_ = false;
    bool decoder_loaded_ = false;
    bool low_mem_mode_ = false;
    std::string error_msg_;
    std::string tts_model_path_;
    std::string decoder_model_path_;
    tts_progress_callback_t progress_callback_;
};

// Utility: Load audio file (WAV format)
bool load_audio_file(const std::string & path, std::vector<float> & samples, 
                     int & sample_rate);

// Utility: Save audio file (WAV format)
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate);

// Utility: Load speaker embedding from JSON or float32 binary
bool load_speaker_embedding_file(const std::string & path,
                                 std::vector<float> & embedding);

// Utility: Save speaker embedding as JSON (.json) or float32 binary
bool save_speaker_embedding_file(const std::string & path,
                                 const std::vector<float> & embedding);

bool load_voice_clone_prompt_file(const std::string & path,
                                  voice_clone_prompt_asset & asset,
                                  std::string * error_msg = nullptr);

bool save_voice_clone_prompt_file(const std::string & path,
                                  const voice_clone_prompt_asset & asset,
                                  std::string * error_msg = nullptr);

} // namespace qwen3_tts
