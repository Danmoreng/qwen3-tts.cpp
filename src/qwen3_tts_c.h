#ifndef QWEN3_TTS_C_H
#define QWEN3_TTS_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  if defined(QWEN3_TTS_EXPORT) || defined(COMPILING_DLL)
#    define QWEN3_TTS_API __declspec(dllexport)
#  else
#    define QWEN3_TTS_API __declspec(dllimport)
#  endif
#else
#  define QWEN3_TTS_API __attribute__((visibility("default")))
#endif

typedef struct qwen3_tts_context qwen3_tts_context_t;

typedef struct {
    int32_t max_audio_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    int32_t n_threads;
    int32_t print_progress; // Use int32 instead of bool for ABI stability
    int32_t print_timing;   // Use int32
    float repetition_penalty;
    int32_t language_id;
    const char* instruction;
    const char* speaker;
} qwen3_tts_params_t;

typedef struct {
    float* audio;
    int32_t audio_len;
    int32_t sample_rate;
    int32_t success;        // Use int32
    char* error_msg;
    int64_t t_total_ms;
} qwen3_tts_result_t;

typedef struct {
    qwen3_tts_params_t generation;
    float chunk_sec;
    float left_context_sec;
    int32_t collect_audio;
} qwen3_tts_streaming_params_t;

typedef enum {
    QWEN3_TTS_MODEL_KIND_UNKNOWN = 0,
    QWEN3_TTS_MODEL_KIND_BASE = 1,
    QWEN3_TTS_MODEL_KIND_CUSTOM_VOICE = 2,
    QWEN3_TTS_MODEL_KIND_VOICE_DESIGN = 3,
} qwen3_tts_model_kind_t;

typedef enum {
    QWEN3_TTS_BACKEND_AUTO = 0,
    QWEN3_TTS_BACKEND_CPU = 1,
    QWEN3_TTS_BACKEND_CUDA = 2,
} qwen3_tts_backend_preference_t;

typedef struct {
    int32_t loaded;
    int32_t supports_voice_clone;
    int32_t supports_named_speakers;
    int32_t supports_instruction;
    int32_t speaker_embedding_dim;
    int32_t speaker_count;
    int32_t model_kind; // qwen3_tts_model_kind_t
} qwen3_tts_model_capabilities_t;

typedef enum {
    QWEN3_TTS_TEXT_ALIGNMENT_NONE = 0,
    QWEN3_TTS_TEXT_ALIGNMENT_ESTIMATED = 1,
    QWEN3_TTS_TEXT_ALIGNMENT_EXACT = 2,
} qwen3_tts_text_alignment_kind_t;

// Streaming audio chunk metadata. Ranges are end-exclusive.
// samples is valid only for the duration of the callback.
typedef struct {
    const float* samples;
    int32_t n_samples;
    int32_t sample_rate;
    int64_t start_sample;
    int64_t end_sample;
    int32_t start_frame;
    int32_t end_frame;
    int32_t start_text_byte;
    int32_t end_text_byte;
    int32_t text_alignment_kind; // qwen3_tts_text_alignment_kind_t
    float confidence;
} qwen3_tts_audio_chunk_t;

typedef void (*qwen3_tts_progress_callback)(int tokens_generated, int max_tokens, void* user_data);
typedef int32_t (*qwen3_tts_audio_chunk_callback)(
    const qwen3_tts_audio_chunk_t* chunk,
    void* user_data
);

QWEN3_TTS_API qwen3_tts_context_t* qwen3_tts_init();
QWEN3_TTS_API void qwen3_tts_free(qwen3_tts_context_t* ctx);
QWEN3_TTS_API int32_t qwen3_tts_set_backend_preference(int32_t preference);
QWEN3_TTS_API int32_t qwen3_tts_get_compiled_backend_mask();
QWEN3_TTS_API char* qwen3_tts_get_active_backend_name();

QWEN3_TTS_API int32_t qwen3_tts_load_models(qwen3_tts_context_t* ctx, const char* model_dir);
QWEN3_TTS_API int32_t qwen3_tts_load_models_with_name(
    qwen3_tts_context_t* ctx,
    const char* model_dir,
    const char* model_name
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize(
    qwen3_tts_context_t* ctx, 
    const char* text, 
    qwen3_tts_params_t params
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize_with_voice(
    qwen3_tts_context_t* ctx, 
    const char* text, 
    const char* reference_audio, 
    qwen3_tts_params_t params
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize_with_speaker_embedding(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* speaker_embedding_file,
    qwen3_tts_params_t params
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize_with_voice_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* reference_audio,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize_with_speaker_embedding_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* speaker_embedding_file,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
);

QWEN3_TTS_API int32_t qwen3_tts_extract_speaker_embedding(
    qwen3_tts_context_t* ctx,
    const char* reference_audio,
    const char* output_path
);

QWEN3_TTS_API qwen3_tts_model_capabilities_t qwen3_tts_get_model_capabilities(
    qwen3_tts_context_t* ctx
);

// Newline-separated speaker names (lowercase), or empty string if unavailable.
// Returned string is heap-allocated and must be released with qwen3_tts_free_string().
QWEN3_TTS_API char* qwen3_tts_get_available_speakers(qwen3_tts_context_t* ctx);
QWEN3_TTS_API void qwen3_tts_free_string(char* value);

QWEN3_TTS_API void qwen3_tts_free_result(qwen3_tts_result_t result);

QWEN3_TTS_API void qwen3_tts_set_progress_callback(
    qwen3_tts_context_t* ctx, 
    qwen3_tts_progress_callback callback, 
    void* user_data
);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_TTS_C_H
