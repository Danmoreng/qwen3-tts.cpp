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

typedef enum {
    QWEN3_TTS_MODEL_KIND_UNKNOWN = 0,
    QWEN3_TTS_MODEL_KIND_BASE = 1,
    QWEN3_TTS_MODEL_KIND_CUSTOM_VOICE = 2,
    QWEN3_TTS_MODEL_KIND_VOICE_DESIGN = 3,
} qwen3_tts_model_kind_t;

typedef struct {
    int32_t loaded;
    int32_t supports_voice_clone;
    int32_t supports_named_speakers;
    int32_t supports_instruction;
    int32_t speaker_embedding_dim;
    int32_t speaker_count;
    int32_t model_kind; // qwen3_tts_model_kind_t
} qwen3_tts_model_capabilities_t;

typedef struct {
    int32_t valid;
    int32_t model_compatible;
    char* error_msg;
} qwen3_tts_voice_clone_prompt_validation_t;

typedef void (*qwen3_tts_progress_callback)(int tokens_generated, int max_tokens, void* user_data);

QWEN3_TTS_API qwen3_tts_context_t* qwen3_tts_init();
QWEN3_TTS_API void qwen3_tts_free(qwen3_tts_context_t* ctx);

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

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize_with_voice_clone_prompt(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* voice_clone_prompt_file,
    qwen3_tts_params_t params
);

QWEN3_TTS_API int32_t qwen3_tts_create_voice_clone_prompt(
    qwen3_tts_context_t* ctx,
    const char* reference_audio,
    const char* reference_text,
    const char* output_path
);

QWEN3_TTS_API qwen3_tts_voice_clone_prompt_validation_t qwen3_tts_validate_voice_clone_prompt(
    qwen3_tts_context_t* ctx,
    const char* voice_clone_prompt_file
);

QWEN3_TTS_API void qwen3_tts_free_voice_clone_prompt_validation(
    qwen3_tts_voice_clone_prompt_validation_t validation
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
