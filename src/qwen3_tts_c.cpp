#include "qwen3_tts_c.h"
#include "qwen3_tts.h"
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

#ifdef _WIN32
#define strdup _strdup
#endif

struct qwen3_tts_context {
    qwen3_tts::Qwen3TTS tts;
    qwen3_tts_progress_callback progress_callback = nullptr;
    void* user_data = nullptr;
};

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
    if (params.instruction) {
        p.instruction = params.instruction;
    }
    if (params.speaker) {
        p.speaker = params.speaker;
    }
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

qwen3_tts_context_t* qwen3_tts_init() {
    return new qwen3_tts_context();
}

void qwen3_tts_free(qwen3_tts_context_t* ctx) {
    delete ctx;
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
    return ctx->tts.load_models(model_dir, model_name ? model_name : "") ? 1 : 0;
}

qwen3_tts_result_t qwen3_tts_synthesize(
    qwen3_tts_context_t* ctx, 
    const char* text, 
    qwen3_tts_params_t params
) {
    if (!ctx || !text) {
        qwen3_tts_result_t res = {0};
        res.success = 0;
        res.error_msg = strdup("Invalid context or text");
        return res;
    }
    auto result = ctx->tts.synthesize(text, convert_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_voice(
    qwen3_tts_context_t* ctx, 
    const char* text, 
    const char* reference_audio, 
    qwen3_tts_params_t params
) {
    if (!ctx || !text || !reference_audio) {
        qwen3_tts_result_t res = {0};
        res.success = 0;
        res.error_msg = strdup("Invalid context, text, or reference audio");
        return res;
    }
    auto result = ctx->tts.synthesize_with_voice(text, reference_audio, convert_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_speaker_embedding(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* speaker_embedding_file,
    qwen3_tts_params_t params
) {
    if (!ctx || !text || !speaker_embedding_file) {
        qwen3_tts_result_t res = {0};
        res.success = 0;
        res.error_msg = strdup("Invalid context, text, or speaker embedding file");
        return res;
    }

    std::vector<float> speaker_embedding;
    if (!qwen3_tts::load_speaker_embedding_file(speaker_embedding_file, speaker_embedding)) {
        qwen3_tts_result_t res = {0};
        res.success = 0;
        res.error_msg = strdup("Failed to load speaker embedding file");
        return res;
    }

    auto result = ctx->tts.synthesize_with_speaker_embedding(text, speaker_embedding, convert_params(params));
    return convert_result(result);
}

int32_t qwen3_tts_extract_speaker_embedding(
    qwen3_tts_context_t* ctx,
    const char* reference_audio,
    const char* output_path
) {
    if (!ctx || !reference_audio || !output_path) return 0;

    std::vector<float> speaker_embedding;
    if (!ctx->tts.extract_speaker_embedding(reference_audio, speaker_embedding, nullptr)) {
        return 0;
    }

    return qwen3_tts::save_speaker_embedding_file(output_path, speaker_embedding) ? 1 : 0;
}

qwen3_tts_model_capabilities_t qwen3_tts_get_model_capabilities(qwen3_tts_context_t* ctx) {
    qwen3_tts_model_capabilities_t out = {0};
    out.model_kind = QWEN3_TTS_MODEL_KIND_UNKNOWN;
    if (!ctx) {
        return out;
    }

    const qwen3_tts::tts_model_capabilities caps = ctx->tts.get_model_capabilities();
    out.loaded = caps.loaded ? 1 : 0;
    out.supports_voice_clone = caps.supports_voice_clone ? 1 : 0;
    out.supports_named_speakers = caps.supports_named_speakers ? 1 : 0;
    out.supports_instruction = caps.supports_instruction ? 1 : 0;
    out.speaker_embedding_dim = caps.speaker_embedding_dim;
    out.speaker_count = caps.speaker_count;
    out.model_kind = to_model_kind(caps.model_type);
    return out;
}

char* qwen3_tts_get_available_speakers(qwen3_tts_context_t* ctx) {
    if (!ctx) {
        return strdup("");
    }

    const std::vector<std::string> speakers = ctx->tts.get_available_speakers();
    std::string joined;
    for (size_t i = 0; i < speakers.size(); ++i) {
        if (i != 0) {
            joined.push_back('\n');
        }
        joined += speakers[i];
    }

    return strdup(joined.c_str());
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
