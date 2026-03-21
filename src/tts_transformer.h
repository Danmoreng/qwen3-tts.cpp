#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_cgraph;
struct gguf_context;

namespace qwen3_tts {
struct tts_transformer_private;
namespace transformer_internal {
struct ops;
}

// TTS Transformer configuration (Qwen2-based Talker)
struct tts_transformer_config {
    // Text embedding
    int32_t text_vocab_size = 151936;
    int32_t text_embd_dim = 2048;
    
    // Talker transformer
    int32_t hidden_size = 1024;
    int32_t n_layers = 28;
    int32_t n_attention_heads = 16;
    int32_t n_key_value_heads = 8;
    int32_t intermediate_size = 3072;
    int32_t head_dim = 128;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    
    // M-RoPE sections [time, freq, channel] = [24, 20, 20]
    int32_t mrope_section[3] = {24, 20, 20};
    bool use_mrope = false;
    int32_t code_pred_mrope_section[3] = {24, 20, 20};
    bool code_pred_use_mrope = false;
    
    // Codec vocabulary
    int32_t codec_vocab_size = 3072;  // talker.codec_embd/codec_head
    int32_t n_codebooks = 16;
    
    // Code predictor
    int32_t code_pred_layers = 5;
    int32_t code_pred_vocab_size = 2048;  // Per-codebook vocab
    int32_t code_pred_hidden_size = 1024;
    int32_t code_pred_intermediate_size = 3072;
    int32_t code_pred_n_attention_heads = 16;
    int32_t code_pred_n_key_value_heads = 8;
    int32_t code_pred_head_dim = 128;
    float code_pred_rms_norm_eps = 1e-6f;
    float code_pred_rope_theta = 1000000.0f;
    
    // Special codec tokens
    int32_t codec_pad_id = 2148;
    int32_t codec_bos_id = 2149;
    int32_t codec_eos_id = 2150;

    int32_t tts_bos_token_id = 151672;
    int32_t tts_eos_token_id = 151673;
    int32_t tts_pad_token_id = 151671;

    int32_t codec_think_id = 2154;
    int32_t codec_nothink_id = 2155;
    int32_t codec_think_bos_id = 2156;
    int32_t codec_think_eos_id = 2157;

    int32_t english_language_id = 2050;

    // Model variant metadata (for 1.7B CustomVoice / VoiceDesign behavior)
    std::string tts_model_type = "base";
    bool has_supports_instruction = false;
    bool supports_instruction = false;
    std::map<std::string, int32_t> speaker_id_map;
};

// TTS Transformer class
class TTSTransformer {
public:
    TTSTransformer();
    ~TTSTransformer();
    
    // Load model from GGUF file
    bool load_model(const std::string & model_path);

    // Configure runtime thread count for CPU-capable backends.
    void set_n_threads(int32_t n_threads);

    // Release all model/runtime resources
    void unload_model();
    
    // Initialize KV cache
    bool init_kv_cache(int32_t n_ctx);
    
    // Clear KV cache
    void clear_kv_cache();
    
    // Initialize code predictor KV cache (5 layers, max 16 context)
    bool init_code_pred_kv_cache(int32_t n_ctx);
    
    // Clear code predictor KV cache
    void clear_code_pred_kv_cache();
    
    // Forward pass for text tokens (prefill phase)
    // text_tokens: input text token IDs [n_tokens]
    // speaker_embd: speaker embedding [hidden_size] (optional, can be nullptr)
    // n_past: number of tokens already in KV cache
    // output: hidden states [n_tokens, hidden_size]
    bool forward_text(const int32_t * text_tokens, int32_t n_tokens,
                      const float * speaker_embd, int32_t n_past,
                      std::vector<float> & output);

    bool forward_prefill(const float * prefill_embd, int32_t n_tokens,
                         int32_t n_past, std::vector<float> & output,
                         std::vector<float> * logits_out = nullptr);
    
    // Forward pass for codec tokens (generation phase)
    // codec_token: single codec token for first codebook
    // n_past: number of tokens already in KV cache
    // output: logits for next codec token [codec_vocab_size]
    bool forward_codec(int32_t codec_token, int32_t n_past,
                       std::vector<float> & output);

    bool forward_step(const float * step_embd, int32_t n_past,
                      std::vector<float> & output,
                      std::vector<float> * hidden_out = nullptr);
    
    // Get hidden states from last forward pass (for code predictor)
    bool get_hidden_states(std::vector<float> & hidden) const;
    
    // Run code predictor to get all 16 codebook predictions
    // hidden: hidden states from talker [hidden_size]
    // prev_codes: previous codes for codebooks 1-15 (can be nullptr for first step)
    // output: logits for all 16 codebooks [16, code_pred_vocab_size]
    bool predict_codes(const float * hidden, const int32_t * prev_codes,
                       std::vector<float> & output);
    
    // Run code predictor autoregressively to generate 15 codes (codebooks 1-15)
    // hidden: hidden states from talker [hidden_size]
    // codebook_0_token: the codebook 0 token (used to create 2-token prefill input)
    // output: generated codes for codebooks 1-15 [15]
    bool predict_codes_autoregressive(const float * hidden, int32_t codebook_0_token, 
                                       std::vector<int32_t> & output,
                                       float temperature = 0.9f,
                                       int32_t top_k = 50,
                                       float top_p = 1.0f,
                                       int32_t trace_frame = -1);
    
    // Generate speech codes autoregressively
    // text_tokens: input text token IDs [n_tokens]
    // speaker_embd: speaker embedding [hidden_size]
    // max_len: maximum number of frames to generate
    // output: generated speech codes [n_frames, n_codebooks]
    bool generate(const int32_t * text_tokens, int32_t n_tokens,
                  const float * speaker_embd, int32_t max_len,
                  std::vector<int32_t> & output,
                  int32_t language_id = 2050,
                  float repetition_penalty = 1.05f,
                  float temperature = 0.9f,
                  int32_t top_k = 50,
                  float top_p = 1.0f,
                  const int32_t * instruct_tokens = nullptr,
                  int32_t n_instruct_tokens = 0);
    
    const tts_transformer_config & get_config() const;
    
    const std::string & get_error() const { return error_msg_; }

    // Resolve a named speaker (CustomVoice models) into a codec embedding vector.
    bool get_named_speaker_embedding(const std::string & speaker_name,
                                     std::vector<float> & speaker_embedding);
    
    // Legacy interface for compatibility
    bool forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                 std::vector<float> & output);
    
    bool forward_with_audio(const int32_t * tokens, int32_t n_tokens,
                            const float * audio_embd, int32_t n_audio,
                            int32_t audio_start_pos, int32_t n_past,
                            std::vector<float> & output);
    
private:
    friend struct transformer_internal::ops;

    std::unique_ptr<tts_transformer_private> impl_;
    std::string error_msg_;
    
    // Cached hidden states from last forward pass
    std::vector<float> last_hidden_;
};

} // namespace qwen3_tts
