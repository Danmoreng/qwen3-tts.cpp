#include "qwen3_tts_c.h"
#include "qwen3_tts.h"
#include "gguf_loader.h"
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <vector>
#include <string>

#ifdef _WIN32
#define strdup _strdup
#endif

struct qwen3_tts_context {
    qwen3_tts::Qwen3TTS tts;
    qwen3_tts_progress_callback progress_callback = nullptr;
    void* user_data = nullptr;
    std::string last_error;
};

static void clear_last_error(qwen3_tts_context_t* ctx) {
    if (ctx) {
        ctx->last_error.clear();
    }
}

static void set_last_error(qwen3_tts_context_t* ctx, const std::string & message) {
    if (ctx) {
        ctx->last_error = message;
    }
}

static int32_t to_model_kind(const std::string & model_type) {
    if (model_type == "base") return QWEN3_TTS_MODEL_KIND_BASE;
    if (model_type == "custom_voice") return QWEN3_TTS_MODEL_KIND_CUSTOM_VOICE;
    if (model_type == "voice_design") return QWEN3_TTS_MODEL_KIND_VOICE_DESIGN;
    return QWEN3_TTS_MODEL_KIND_UNKNOWN;
}

static qwen3_tts::tts_params convert_params(qwen3_tts_params_t params) {
    qwen3_tts::tts_params p;
    p.max_audio_tokens = params.max_audio_tokens;
    p.temperature = params.temperature;
    p.top_p = params.top_p;
    p.top_k = params.top_k;
    p.n_threads = params.n_threads;
    p.print_progress = params.print_progress != 0;
    p.print_timing = params.print_timing != 0;
    p.repetition_penalty = params.repetition_penalty;
    p.language_id = params.language_id;
    if (params.vocoder_left_context_sec > 0.0f) {
        p.vocoder_left_context_sec = params.vocoder_left_context_sec;
    }
    if (params.instruction) {
        p.instruction = params.instruction;
    }
    if (params.speaker) {
        p.speaker = params.speaker;
    }
    return p;
}

void qwen3_tts_bridge_abi_sizes(int32_t* sizes) {
    if (!sizes) return;
    sizes[0] = sizeof(qwen3_tts_params_t);
    sizes[1] = sizeof(qwen3_tts_streaming_params_t);
    sizes[2] = sizeof(qwen3_tts_audio_chunk_t);
    sizes[3] = sizeof(qwen3_tts_result_t);
}

static int32_t copy_abi_table(const int32_t* table, int32_t count, int32_t* out, int32_t capacity) {
    if (out) {
        for (int32_t i = 0; i < count && i < capacity; ++i) {
            out[i] = table[i];
        }
    }
    return count;
}

// Field list shared by the layout and the type table: keep in declaration order.
#define QWEN3_TTS_ABI_FIELDS(X)                                        \
    X(qwen3_tts_params_t, max_audio_tokens)                            \
    X(qwen3_tts_params_t, temperature)                                 \
    X(qwen3_tts_params_t, top_p)                                       \
    X(qwen3_tts_params_t, top_k)                                       \
    X(qwen3_tts_params_t, n_threads)                                   \
    X(qwen3_tts_params_t, print_progress)                              \
    X(qwen3_tts_params_t, print_timing)                                \
    X(qwen3_tts_params_t, repetition_penalty)                          \
    X(qwen3_tts_params_t, language_id)                                 \
    X(qwen3_tts_params_t, instruction)                                 \
    X(qwen3_tts_params_t, speaker)                                     \
    X(qwen3_tts_params_t, vocoder_left_context_sec)                    \
    X(qwen3_tts_streaming_params_t, generation)                        \
    X(qwen3_tts_streaming_params_t, chunk_sec)                         \
    X(qwen3_tts_streaming_params_t, left_context_sec)                  \
    X(qwen3_tts_streaming_params_t, collect_audio)                     \
    X(qwen3_tts_audio_chunk_t, samples)                                \
    X(qwen3_tts_audio_chunk_t, n_samples)                              \
    X(qwen3_tts_audio_chunk_t, sample_rate)                            \
    X(qwen3_tts_audio_chunk_t, start_sample)                           \
    X(qwen3_tts_audio_chunk_t, end_sample)                             \
    X(qwen3_tts_audio_chunk_t, start_frame)                            \
    X(qwen3_tts_audio_chunk_t, end_frame)                              \
    X(qwen3_tts_audio_chunk_t, start_text_byte)                        \
    X(qwen3_tts_audio_chunk_t, end_text_byte)                          \
    X(qwen3_tts_audio_chunk_t, text_alignment_kind)                    \
    X(qwen3_tts_audio_chunk_t, confidence)                             \
    X(qwen3_tts_result_t, audio)                                       \
    X(qwen3_tts_result_t, audio_len)                                   \
    X(qwen3_tts_result_t, sample_rate)                                 \
    X(qwen3_tts_result_t, success)                                     \
    X(qwen3_tts_result_t, error_msg)                                   \
    X(qwen3_tts_result_t, t_total_ms)

template <typename T> struct abi_type_code { static const int32_t value = 5; };  // nested struct
template <> struct abi_type_code<int32_t> { static const int32_t value = 1; };
template <> struct abi_type_code<int64_t> { static const int32_t value = 2; };
template <> struct abi_type_code<float> { static const int32_t value = 3; };
template <typename T> struct abi_type_code<T*> { static const int32_t value = 4; };

int32_t qwen3_tts_bridge_abi_types(int32_t* out, int32_t capacity) {
#define QWEN3_TTS_TYPE(type, field) abi_type_code<decltype(type::field)>::value,
    static const int32_t types[] = { QWEN3_TTS_ABI_FIELDS(QWEN3_TTS_TYPE) };
#undef QWEN3_TTS_TYPE
    return copy_abi_table(types, (int32_t) (sizeof(types) / sizeof(types[0])), out, capacity);
}

int32_t qwen3_tts_bridge_abi_layout(int32_t* out, int32_t capacity) {
#define QWEN3_TTS_OFFSET(type, field) (int32_t) offsetof(type, field)
    static const int32_t layout[] = {
        (int32_t) sizeof(qwen3_tts_params_t),
        QWEN3_TTS_OFFSET(qwen3_tts_params_t, max_audio_tokens),
        QWEN3_TTS_OFFSET(qwen3_tts_params_t, temperature),
        QWEN3_TTS_OFFSET(qwen3_tts_params_t, top_p),
        QWEN3_TTS_OFFSET(qwen3_tts_params_t, top_k),
        QWEN3_TTS_OFFSET(qwen3_tts_params_t, n_threads),
        QWEN3_TTS_OFFSET(qwen3_tts_params_t, print_progress),
        QWEN3_TTS_OFFSET(qwen3_tts_params_t, print_timing),
        QWEN3_TTS_OFFSET(qwen3_tts_params_t, repetition_penalty),
        QWEN3_TTS_OFFSET(qwen3_tts_params_t, language_id),
        QWEN3_TTS_OFFSET(qwen3_tts_params_t, instruction),
        QWEN3_TTS_OFFSET(qwen3_tts_params_t, speaker),
        QWEN3_TTS_OFFSET(qwen3_tts_params_t, vocoder_left_context_sec),

        (int32_t) sizeof(qwen3_tts_streaming_params_t),
        QWEN3_TTS_OFFSET(qwen3_tts_streaming_params_t, generation),
        QWEN3_TTS_OFFSET(qwen3_tts_streaming_params_t, chunk_sec),
        QWEN3_TTS_OFFSET(qwen3_tts_streaming_params_t, left_context_sec),
        QWEN3_TTS_OFFSET(qwen3_tts_streaming_params_t, collect_audio),

        (int32_t) sizeof(qwen3_tts_audio_chunk_t),
        QWEN3_TTS_OFFSET(qwen3_tts_audio_chunk_t, samples),
        QWEN3_TTS_OFFSET(qwen3_tts_audio_chunk_t, n_samples),
        QWEN3_TTS_OFFSET(qwen3_tts_audio_chunk_t, sample_rate),
        QWEN3_TTS_OFFSET(qwen3_tts_audio_chunk_t, start_sample),
        QWEN3_TTS_OFFSET(qwen3_tts_audio_chunk_t, end_sample),
        QWEN3_TTS_OFFSET(qwen3_tts_audio_chunk_t, start_frame),
        QWEN3_TTS_OFFSET(qwen3_tts_audio_chunk_t, end_frame),
        QWEN3_TTS_OFFSET(qwen3_tts_audio_chunk_t, start_text_byte),
        QWEN3_TTS_OFFSET(qwen3_tts_audio_chunk_t, end_text_byte),
        QWEN3_TTS_OFFSET(qwen3_tts_audio_chunk_t, text_alignment_kind),
        QWEN3_TTS_OFFSET(qwen3_tts_audio_chunk_t, confidence),

        (int32_t) sizeof(qwen3_tts_result_t),
        QWEN3_TTS_OFFSET(qwen3_tts_result_t, audio),
        QWEN3_TTS_OFFSET(qwen3_tts_result_t, audio_len),
        QWEN3_TTS_OFFSET(qwen3_tts_result_t, sample_rate),
        QWEN3_TTS_OFFSET(qwen3_tts_result_t, success),
        QWEN3_TTS_OFFSET(qwen3_tts_result_t, error_msg),
        QWEN3_TTS_OFFSET(qwen3_tts_result_t, t_total_ms),
    };
#undef QWEN3_TTS_OFFSET
    return copy_abi_table(layout, (int32_t) (sizeof(layout) / sizeof(layout[0])), out, capacity);
}

static qwen3_tts::tts_streaming_params convert_streaming_params(qwen3_tts_streaming_params_t params) {
    qwen3_tts::tts_streaming_params p;
    p.generation = convert_params(params.generation);
    p.chunk_sec = params.chunk_sec > 0.0f ? params.chunk_sec : p.chunk_sec;
    p.left_context_sec = params.left_context_sec >= 0.0f ? params.left_context_sec : p.left_context_sec;
    p.collect_audio = params.collect_audio != 0;
    return p;
}

static qwen3_tts_result_t convert_result(const qwen3_tts::tts_result& res) {
    qwen3_tts_result_t r;
    r.audio_len = static_cast<int32_t>(res.audio.size());
    if (r.audio_len > 0) {
        r.audio = (float*)malloc(r.audio_len * sizeof(float));
        std::memcpy(r.audio, res.audio.data(), r.audio_len * sizeof(float));
    } else {
        r.audio = nullptr;
    }
    r.sample_rate = res.sample_rate;
    r.success = res.success ? 1 : 0;
    if (!res.error_msg.empty()) {
        r.error_msg = strdup(res.error_msg.c_str());
    } else {
        r.error_msg = nullptr;
    }
    r.t_total_ms = res.t_total_ms;
    return r;
}

static qwen3_tts_result_t make_error_result(const char * message) {
    qwen3_tts_result_t res = {0};
    res.success = 0;
    res.error_msg = strdup(message ? message : "Unknown error");
    return res;
}

// No C++ exception may cross the C boundary: on Windows it would terminate the
// host process (a Python interpreter, say) instead of reaching the caller.
static const char * UNKNOWN_EXCEPTION = "Unhandled C++ exception in the qwen3_tts bridge";

template <typename Body>
static qwen3_tts_result_t guarded_result(Body && body) {
    try {
        return body();
    } catch (const std::exception & error) {
        return make_error_result(error.what());
    } catch (...) {
        return make_error_result(UNKNOWN_EXCEPTION);
    }
}

template <typename Body>
static int32_t guarded_status(qwen3_tts_context_t* ctx, Body && body) {
    try {
        return body();
    } catch (const std::exception & error) {
        set_last_error(ctx, error.what());
        return 0;
    } catch (...) {
        set_last_error(ctx, UNKNOWN_EXCEPTION);
        return 0;
    }
}

static bool load_icl_prompt_params(const char * icl_prompt_file,
                                   qwen3_tts::icl_prompt & prompt,
                                   qwen3_tts::tts_params & params,
                                   qwen3_tts_result_t & error_result) {
    if (!qwen3_tts::load_icl_prompt_file(icl_prompt_file, prompt)) {
        error_result = make_error_result("Failed to load ICL prompt file");
        return false;
    }
    params.reference_text = prompt.reference_text;
    params.reference_token_ids = prompt.reference_token_ids;
    params.reference_codes = prompt.reference_codes;
    return true;
}

static qwen3_tts_audio_chunk_t convert_audio_chunk(const qwen3_tts::tts_audio_chunk & chunk) {
    qwen3_tts_audio_chunk_t out;
    out.samples = chunk.samples;
    out.n_samples = chunk.n_samples;
    out.sample_rate = chunk.sample_rate;
    out.start_sample = chunk.start_sample;
    out.end_sample = chunk.end_sample;
    out.start_frame = chunk.start_frame;
    out.end_frame = chunk.end_frame;
    out.start_text_byte = chunk.start_text_byte;
    out.end_text_byte = chunk.end_text_byte;
    out.text_alignment_kind = chunk.text_alignment_kind;
    out.confidence = chunk.confidence;
    return out;
}

static qwen3_tts::tts_audio_chunk_callback_t wrap_chunk_callback(qwen3_tts_audio_chunk_callback callback,
                                                                 void* user_data) {
    return [callback, user_data](const qwen3_tts::tts_audio_chunk & chunk) {
        qwen3_tts_audio_chunk_t c_chunk = convert_audio_chunk(chunk);
        return callback(&c_chunk, user_data) != 0;
    };
}

qwen3_tts_context_t* qwen3_tts_init() {
    try {
        return new qwen3_tts_context();
    } catch (...) {
        return nullptr;
    }
}

void qwen3_tts_free(qwen3_tts_context_t* ctx) {
    delete ctx;
}

int32_t qwen3_tts_set_backend_preference(int32_t preference) {
    return guarded_status(nullptr, [&]() -> int32_t {
        qwen3_tts::backend_preference native_preference = qwen3_tts::backend_preference::auto_select;
        if (preference == QWEN3_TTS_BACKEND_CPU) {
            native_preference = qwen3_tts::backend_preference::cpu;
        } else if (preference == QWEN3_TTS_BACKEND_CUDA) {
            native_preference = qwen3_tts::backend_preference::cuda;
        }
        return qwen3_tts::set_backend_preference(native_preference) ? 1 : 0;
    });
}

int32_t qwen3_tts_set_cpu_threads(int32_t n_threads) {
    return guarded_status(nullptr, [&]() -> int32_t {
        return qwen3_tts::set_cpu_thread_count(n_threads) ? 1 : 0;
    });
}

int32_t qwen3_tts_get_cpu_threads() {
    return qwen3_tts::get_cpu_thread_count();
}

int32_t qwen3_tts_get_compiled_backend_mask() {
    return qwen3_tts::get_compiled_backend_mask();
}

char* qwen3_tts_get_active_backend_name() {
    try {
        const std::string name = qwen3_tts::get_active_backend_name();
        return strdup(name.c_str());
    } catch (...) {
        return strdup("");
    }
}

int32_t qwen3_tts_load_models(qwen3_tts_context_t* ctx, const char* model_dir) {
    return qwen3_tts_load_models_with_name(ctx, model_dir, nullptr);
}

int32_t qwen3_tts_load_models_with_name(
    qwen3_tts_context_t* ctx,
    const char* model_dir,
    const char* model_name
) {
    if (!ctx || !model_dir) return 0;
    return guarded_status(ctx, [&]() -> int32_t {
        return ctx->tts.load_models(model_dir, model_name ? model_name : "") ? 1 : 0;
    });
}

int32_t qwen3_tts_load_icl_prompt_encoder(
    qwen3_tts_context_t* ctx,
    const char* model_dir
) {
    return qwen3_tts_load_icl_prompt_encoder_with_name(ctx, model_dir, nullptr);
}

int32_t qwen3_tts_load_icl_prompt_encoder_with_name(
    qwen3_tts_context_t* ctx,
    const char* model_dir,
    const char* model_name
) {
    if (!ctx || !model_dir) return 0;
    return guarded_status(ctx, [&]() -> int32_t {
        return ctx->tts.load_icl_prompt_encoder_only(model_dir, model_name ? model_name : "") ? 1 : 0;
    });
}

qwen3_tts_result_t qwen3_tts_synthesize(
    qwen3_tts_context_t* ctx,
    const char* text,
    qwen3_tts_params_t params
) {
    if (!ctx || !text) {
        return make_error_result("Invalid context or text");
    }
    return guarded_result([&]() {
        return convert_result(ctx->tts.synthesize(text, convert_params(params)));
    });
}

qwen3_tts_result_t qwen3_tts_synthesize_with_voice(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* reference_audio,
    qwen3_tts_params_t params
) {
    if (!ctx || !text || !reference_audio) {
        return make_error_result("Invalid context, text, or reference audio");
    }
    return guarded_result([&]() {
        return convert_result(ctx->tts.synthesize_with_voice(text, reference_audio, convert_params(params)));
    });
}

qwen3_tts_result_t qwen3_tts_synthesize_with_speaker_embedding(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* speaker_embedding_file,
    qwen3_tts_params_t params
) {
    if (!ctx || !text || !speaker_embedding_file) {
        return make_error_result("Invalid context, text, or speaker embedding file");
    }
    return guarded_result([&]() {
        std::vector<float> speaker_embedding;
        if (!qwen3_tts::load_speaker_embedding_file(speaker_embedding_file, speaker_embedding)) {
            return make_error_result("Failed to load speaker embedding file");
        }
        return convert_result(
            ctx->tts.synthesize_with_speaker_embedding(text, speaker_embedding, convert_params(params)));
    });
}

qwen3_tts_result_t qwen3_tts_synthesize_with_icl_prompt(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* icl_prompt_file,
    qwen3_tts_params_t params
) {
    if (!ctx || !text || !icl_prompt_file) {
        return make_error_result("Invalid context, text, or ICL prompt file");
    }
    return guarded_result([&]() {
        qwen3_tts::icl_prompt prompt;
        qwen3_tts::tts_params native_params = convert_params(params);
        qwen3_tts_result_t error_result = {0};
        if (!load_icl_prompt_params(icl_prompt_file, prompt, native_params, error_result)) {
            return error_result;
        }
        return convert_result(
            ctx->tts.synthesize_with_speaker_embedding(text, prompt.speaker_embedding, native_params));
    });
}

qwen3_tts_result_t qwen3_tts_synthesize_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !text || !callback) {
        return make_error_result("Invalid context, text, or streaming callback");
    }
    return guarded_result([&]() {
        return convert_result(ctx->tts.synthesize_streaming(
            text, wrap_chunk_callback(callback, user_data), convert_streaming_params(params)));
    });
}

qwen3_tts_result_t qwen3_tts_synthesize_with_voice_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* reference_audio,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !text || !reference_audio || !callback) {
        return make_error_result("Invalid context, text, reference audio, or streaming callback");
    }
    return guarded_result([&]() {
        return convert_result(ctx->tts.synthesize_with_voice_streaming(
            text, reference_audio, wrap_chunk_callback(callback, user_data), convert_streaming_params(params)));
    });
}

qwen3_tts_result_t qwen3_tts_synthesize_with_speaker_embedding_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* speaker_embedding_file,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !text || !speaker_embedding_file || !callback) {
        return make_error_result("Invalid context, text, speaker embedding file, or streaming callback");
    }
    return guarded_result([&]() {
        std::vector<float> speaker_embedding;
        if (!qwen3_tts::load_speaker_embedding_file(speaker_embedding_file, speaker_embedding)) {
            return make_error_result("Failed to load speaker embedding file");
        }
        return convert_result(ctx->tts.synthesize_with_speaker_embedding_streaming(
            text, speaker_embedding, wrap_chunk_callback(callback, user_data), convert_streaming_params(params)));
    });
}

qwen3_tts_result_t qwen3_tts_synthesize_with_icl_prompt_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* icl_prompt_file,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !text || !icl_prompt_file || !callback) {
        return make_error_result("Invalid context, text, ICL prompt file, or streaming callback");
    }
    return guarded_result([&]() {
        qwen3_tts::icl_prompt prompt;
        qwen3_tts::tts_streaming_params native_params = convert_streaming_params(params);
        qwen3_tts_result_t error_result = {0};
        if (!load_icl_prompt_params(icl_prompt_file, prompt, native_params.generation, error_result)) {
            return error_result;
        }
        return convert_result(ctx->tts.synthesize_with_speaker_embedding_streaming(
            text, prompt.speaker_embedding, wrap_chunk_callback(callback, user_data), native_params));
    });
}

int32_t qwen3_tts_extract_speaker_embedding(
    qwen3_tts_context_t* ctx,
    const char* reference_audio,
    const char* output_path
) {
    if (!ctx || !reference_audio || !output_path) {
        set_last_error(ctx, "Invalid context, reference audio, or output path");
        return 0;
    }
    clear_last_error(ctx);
    return guarded_status(ctx, [&]() -> int32_t {
        std::vector<float> speaker_embedding;
        if (!ctx->tts.extract_speaker_embedding(reference_audio, speaker_embedding, nullptr)) {
            set_last_error(ctx, ctx->tts.get_error());
            return 0;
        }
        if (!qwen3_tts::save_speaker_embedding_file(output_path, speaker_embedding)) {
            set_last_error(ctx, std::string("Failed to save speaker embedding file: ") + output_path);
            return 0;
        }
        return 1;
    });
}

int32_t qwen3_tts_extract_icl_prompt(
    qwen3_tts_context_t* ctx,
    const char* reference_audio,
    const char* reference_text,
    const char* output_path
) {
    if (!ctx || !reference_audio || !reference_text || !output_path) {
        set_last_error(ctx, "Invalid context, reference audio, reference text, or output path");
        return 0;
    }
    clear_last_error(ctx);
    return guarded_status(ctx, [&]() -> int32_t {
        qwen3_tts::icl_prompt prompt;
        if (!ctx->tts.extract_icl_prompt(reference_audio, reference_text, prompt, nullptr)) {
            set_last_error(ctx, ctx->tts.get_error());
            return 0;
        }
        if (!qwen3_tts::save_icl_prompt_file(output_path, prompt)) {
            set_last_error(ctx, std::string("Failed to save ICL prompt file: ") + output_path);
            return 0;
        }
        return 1;
    });
}

qwen3_tts_model_capabilities_t qwen3_tts_get_model_capabilities(qwen3_tts_context_t* ctx) {
    qwen3_tts_model_capabilities_t out = {0};
    out.model_kind = QWEN3_TTS_MODEL_KIND_UNKNOWN;
    if (!ctx) {
        return out;
    }
    try {
        const qwen3_tts::tts_model_capabilities caps = ctx->tts.get_model_capabilities();
        out.loaded = caps.loaded ? 1 : 0;
        out.supports_voice_clone = caps.supports_voice_clone ? 1 : 0;
        out.supports_named_speakers = caps.supports_named_speakers ? 1 : 0;
        out.supports_instruction = caps.supports_instruction ? 1 : 0;
        out.speaker_embedding_dim = caps.speaker_embedding_dim;
        out.speaker_count = caps.speaker_count;
        out.model_kind = to_model_kind(caps.model_type);
    } catch (const std::exception & error) {
        set_last_error(ctx, error.what());
    } catch (...) {
        set_last_error(ctx, UNKNOWN_EXCEPTION);
    }
    return out;
}

char* qwen3_tts_get_available_speakers(qwen3_tts_context_t* ctx) {
    if (!ctx) {
        return strdup("");
    }
    try {
        const std::vector<std::string> speakers = ctx->tts.get_available_speakers();
        std::string joined;
        for (size_t i = 0; i < speakers.size(); ++i) {
            if (i != 0) {
                joined.push_back('\n');
            }
            joined += speakers[i];
        }
        return strdup(joined.c_str());
    } catch (...) {
        set_last_error(ctx, UNKNOWN_EXCEPTION);
        return strdup("");
    }
}

char* qwen3_tts_get_last_error(qwen3_tts_context_t* ctx) {
    if (!ctx) {
        return strdup("Invalid context");
    }
    const std::string & error = !ctx->last_error.empty() ? ctx->last_error : ctx->tts.get_error();
    return strdup(error.c_str());
}

void qwen3_tts_free_string(char* value) {
    if (value) {
        free(value);
    }
}

void qwen3_tts_free_result(qwen3_tts_result_t result) {
    if (result.audio) free(result.audio);
    if (result.error_msg) free(result.error_msg);
}

void qwen3_tts_set_progress_callback(
    qwen3_tts_context_t* ctx,
    qwen3_tts_progress_callback callback,
    void* user_data
) {
    if (!ctx) return;
    ctx->progress_callback = callback;
    ctx->user_data = user_data;

    if (callback) {
        ctx->tts.set_progress_callback([ctx](int tokens, int max) {
            ctx->progress_callback(tokens, max, ctx->user_data);
        });
    } else {
        ctx->tts.set_progress_callback(nullptr);
    }
}
