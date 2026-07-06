#pragma once

#include "text_tokenizer.h"
#include "tts_transformer.h"
#include "audio_tokenizer_encoder.h"
#include "speech_tokenizer_encoder.h"
#include "audio_tokenizer_decoder.h"

#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include <optional>

namespace qwen3_tts {
namespace pipeline_internal {
struct ops;
}

struct speech_codes {
    std::vector<int32_t> codes;
    int32_t n_frames = 0;
    int32_t n_codebooks = 0;
};

struct icl_prompt {
    std::vector<float> speaker_embedding;
    std::string reference_text;
    std::vector<int32_t> reference_token_ids;
    speech_codes reference_codes;
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

    // RNG seed for stochastic sampling (-1 = random per synthesis)
    int64_t seed = -1;
    
    // Number of threads
    int32_t n_threads = 4;
    
    // Print progress during generation
    bool print_progress = false;
    
    // Print timing information
    bool print_timing = true;
    
    // Repetition penalty for CB0 token generation (HuggingFace style)
    float repetition_penalty = 1.05f;

    // Language ID for codec (-1=auto, 2050=en, 2069=ru, 2055=zh, 2058=ja, 2064=ko, 2053=de, 2061=fr, 2054=es)
    int32_t language_id = -1;

    // Optional style/voice instruction
    std::string instruction;

    // Optional named speaker (for CustomVoice models)
    std::string speaker;

    // Optional ICL voice-clone prompt. reference_codes may omit n_codebooks; the
    // loaded talker codebook count is used when inferring frames.
    std::string reference_text;
    std::vector<int32_t> reference_token_ids;
    std::optional<speech_codes> reference_codes;

    // Optional debug dumps for generated and decoder-input speech codes.
    std::string dump_generated_codes_path;
    std::string dump_decoder_codes_path;

    // ICL vocoder left context in seconds. When reference speech codes are
    // present, the decoder is seeded with the last N reference frames and those
    // samples are trimmed from the output. Set high enough to cover the full
    // reference to reproduce full reference+generated decode.
    float vocoder_left_context_sec = 2.0f;
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
    // Time from code-generation start to the first complete generated audio
    // code frame. This matches qwentts.cpp's "first frame codes" TTFA metric.
    int64_t t_ttfa_ms = -1;

    // Reference preparation timing breakdown (milliseconds)
    int64_t t_reference_speaker_load_ms = 0;
    int64_t t_reference_speaker_encode_ms = 0;
    int64_t t_reference_speech_load_ms = 0;
    int64_t t_reference_speech_encode_ms = 0;
    int64_t t_reference_speech_project_ms = 0;
    int64_t t_reference_speech_graph_build_ms = 0;
    int64_t t_reference_speech_graph_alloc_ms = 0;
    int64_t t_reference_speech_input_upload_ms = 0;
    int64_t t_reference_speech_mask_prepare_ms = 0;
    int64_t t_reference_speech_graph_compute_ms = 0;
    int64_t t_reference_speech_output_read_ms = 0;
    int64_t t_reference_speech_quantize_ms = 0;
    int64_t t_reference_speech_quantize_semantic_ms = 0;
    int64_t t_reference_speech_quantize_acoustic_ms = 0;

    // Decoder timing breakdown (milliseconds)
    int64_t t_decode_graph_build_ms = 0;
    int64_t t_decode_graph_alloc_ms = 0;
    int64_t t_decode_input_upload_ms = 0;
    int64_t t_decode_graph_compute_ms = 0;
    int64_t t_decode_output_read_ms = 0;
    int32_t decode_graph_rebuilt = 0;
    int32_t decode_frames = 0;
    int64_t decode_samples = 0;

    // Streaming decoder counters. For non-streaming synthesis these stay zero.
    int32_t streaming_decode_chunks = 0;
    int32_t streaming_decode_input_frames = 0;
    int32_t streaming_decode_emitted_frames = 0;
    int32_t streaming_decode_context_frames = 0;
    int32_t streaming_decode_graph_rebuilds = 0;
    int64_t streaming_decode_graph_build_ms = 0;
    int64_t streaming_decode_graph_alloc_ms = 0;
    int64_t streaming_decode_input_upload_ms = 0;
    int64_t streaming_decode_graph_compute_ms = 0;
    int64_t streaming_decode_output_read_ms = 0;

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

enum tts_text_alignment_kind {
    TTS_TEXT_ALIGNMENT_NONE = 0,
    TTS_TEXT_ALIGNMENT_ESTIMATED = 1,
    TTS_TEXT_ALIGNMENT_EXACT = 2,
};

// Audio chunk metadata for streaming synthesis. Ranges are end-exclusive.
// Text byte ranges refer to the original input text and are currently estimated.
struct tts_audio_chunk {
    const float * samples = nullptr;
    int32_t n_samples = 0;
    int32_t sample_rate = 0;
    int64_t start_sample = 0;
    int64_t end_sample = 0;
    int32_t start_frame = 0;
    int32_t end_frame = 0;
    int32_t start_text_byte = -1;
    int32_t end_text_byte = -1;
    int32_t text_alignment_kind = TTS_TEXT_ALIGNMENT_NONE;
    float confidence = 0.0f;
};

// Audio chunk callback type for streaming synthesis.
// Return false to stop synthesis early.
using tts_audio_chunk_callback_t = std::function<bool(const tts_audio_chunk & chunk)>;

// Streaming synthesis options. The regular tts_params stay nested so the
// streaming API is additive and does not change the default batch path.
struct tts_streaming_params {
    tts_params generation;
    float chunk_sec = 1.0f;
    float left_context_sec = 2.0f;
    bool collect_audio = false;
};

// Main TTS class that orchestrates the full pipeline
class Qwen3TTS {
public:
    Qwen3TTS();
    ~Qwen3TTS();
    
    // Load all models from directory
    // model_dir should contain: transformer.gguf, tokenizer.gguf, vocoder.gguf
    bool load_models(const std::string & model_dir,
                     const std::string & model_name = "",
                     const std::string & tokenizer_model_path = "");

    // Load only the speaker encoder tensors needed by extract_speaker_embedding().
    bool load_speaker_encoder_only(const std::string & model_dir, const std::string & model_name = "");

    // Load only the components needed to extract reusable full-ICL prompts.
    bool load_icl_prompt_encoder_only(const std::string & model_dir,
                                      const std::string & model_name = "",
                                      const std::string & tokenizer_model_path = "");
    
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

    // Streaming variants emit decoded audio chunks while frames are generated.
    // The normal synthesize* APIs remain the buffered full-audio path.
    tts_result synthesize_streaming(const std::string & text,
                                    const tts_audio_chunk_callback_t & on_audio_chunk,
                                    const tts_streaming_params & params = tts_streaming_params());

    tts_result synthesize_with_voice_streaming(const std::string & text,
                                               const std::string & reference_audio,
                                               const tts_audio_chunk_callback_t & on_audio_chunk,
                                               const tts_streaming_params & params = tts_streaming_params());

    tts_result synthesize_with_voice_streaming(const std::string & text,
                                               const float * ref_samples,
                                               int32_t n_ref_samples,
                                               const tts_audio_chunk_callback_t & on_audio_chunk,
                                               const tts_streaming_params & params = tts_streaming_params());

    tts_result synthesize_with_speaker_embedding_streaming(
        const std::string & text,
        const std::vector<float> & speaker_embedding,
        const tts_audio_chunk_callback_t & on_audio_chunk,
        const tts_streaming_params & params = tts_streaming_params());

    // Extract speaker embedding from reference audio file (WAV)
    bool extract_speaker_embedding(const std::string & reference_audio,
                                   std::vector<float> & speaker_embedding,
                                   int64_t * encode_time_ms = nullptr);

    // Extract a reusable full-ICL prompt from reference audio and transcript.
    // The resulting prompt contains the speaker embedding and reference speech
    // codes, so later synthesis does not need to re-run either audio encoder.
    bool extract_icl_prompt(const std::string & reference_audio,
                            const std::string & reference_text,
                            icl_prompt & prompt,
                            int64_t * encode_time_ms = nullptr);
    
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

    struct voice_prompt_cache_entry {
        bool valid = false;
        uint64_t sample_hash = 0;
        int32_t n_samples = 0;
        std::string reference_text;
        std::vector<int32_t> reference_token_ids;
        bool has_auto_reference_codes = false;
        bool reference_codes_cuda = false;
        std::vector<float> speaker_embedding;
        std::optional<speech_codes> reference_codes;
    };

    TextTokenizer tokenizer_;
    TTSTransformer transformer_;
    AudioTokenizerEncoder audio_encoder_;
    SpeechTokenizerEncoder speech_encoder_;
    AudioTokenizerDecoder audio_decoder_;
    AudioTokenizerDecoder streaming_audio_decoder_;
    voice_prompt_cache_entry voice_prompt_cache_;
    
    bool models_loaded_ = false;
    bool text_tokenizer_loaded_ = false;
    bool encoder_loaded_ = false;
    bool speech_encoder_loaded_ = false;
    bool speech_encoder_cuda_enabled_ = false;
    bool transformer_loaded_ = false;
    bool decoder_loaded_ = false;
    bool streaming_decoder_loaded_ = false;
    bool low_mem_mode_ = false;
    std::string error_msg_;
    std::string tts_model_path_;
    std::string speaker_encoder_model_path_;
    std::string tokenizer_model_path_;
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

bool load_icl_prompt_file(const std::string & path,
                          icl_prompt & prompt);

bool save_icl_prompt_file(const std::string & path,
                          const icl_prompt & prompt);

} // namespace qwen3_tts
