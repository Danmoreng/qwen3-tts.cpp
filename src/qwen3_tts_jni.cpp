#include <jni.h>
#include "qwen3_tts_c.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

#define LOGE(...) fprintf(stderr, "[QwenEngine_JNI] " __VA_ARGS__); fprintf(stderr, "\n")

static jclass g_result_class = nullptr;
static jmethodID g_result_constructor = nullptr;
static jclass g_caps_class = nullptr;
static jmethodID g_caps_constructor = nullptr;

static jclass g_params_class = nullptr;
static jfieldID g_lang_id_field = nullptr;
static jfieldID g_instruction_field = nullptr;
static jfieldID g_speaker_field = nullptr;
static jfieldID g_max_audio_tokens_field = nullptr;

struct ProgressCallbackState {
    JavaVM* vm = nullptr;
    jobject callback = nullptr;
    jmethodID on_progress = nullptr;
};

static std::mutex g_progress_mutex;
static std::unordered_map<jlong, ProgressCallbackState*> g_progress_callbacks;

static void clear_progress_callback(JNIEnv* env, jlong ctx_ptr) {
    std::lock_guard<std::mutex> lock(g_progress_mutex);
    auto it = g_progress_callbacks.find(ctx_ptr);
    if (it == g_progress_callbacks.end()) {
        return;
    }
    qwen3_tts_set_progress_callback(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), nullptr, nullptr);
    if (it->second != nullptr) {
        if (it->second->callback != nullptr) {
            env->DeleteGlobalRef(it->second->callback);
        }
        delete it->second;
    }
    g_progress_callbacks.erase(it);
}

static jobject make_native_result(JNIEnv* env, qwen3_tts_result_t c_result) {
    if (g_result_class == nullptr || g_result_constructor == nullptr) {
        return nullptr;
    }

    jfloatArray audio_array = nullptr;
    if (c_result.audio_len > 0 && c_result.audio != nullptr) {
        audio_array = env->NewFloatArray(c_result.audio_len);
        if (audio_array != nullptr) {
            env->SetFloatArrayRegion(audio_array, 0, c_result.audio_len, c_result.audio);
        } else {
            env->ExceptionClear();
        }
    }

    jstring error_msg = nullptr;
    if (c_result.error_msg) {
        error_msg = env->NewStringUTF(c_result.error_msg);
        if (error_msg == nullptr) {
            env->ExceptionClear();
        }
    }

    return env->NewObject(g_result_class, g_result_constructor,
                          audio_array,
                          (jint)c_result.sample_rate,
                          (jboolean)(c_result.success != 0),
                          error_msg,
                          (jlong)c_result.t_total_ms);
}

static void fill_params_from_java(JNIEnv* env, jobject params, qwen3_tts_params_t* c_params,
                                  jstring* j_instruction, const char** c_instruction,
                                  jstring* j_speaker, const char** c_speaker) {
    if (params != nullptr && g_lang_id_field != nullptr) {
        c_params->language_id = env->GetIntField(params, g_lang_id_field);
    }
    if (params != nullptr && g_max_audio_tokens_field != nullptr) {
        c_params->max_audio_tokens = env->GetIntField(params, g_max_audio_tokens_field);
    }
    if (params != nullptr && g_instruction_field != nullptr) {
        *j_instruction = (jstring)env->GetObjectField(params, g_instruction_field);
        if (*j_instruction != nullptr) {
            *c_instruction = env->GetStringUTFChars(*j_instruction, nullptr);
            c_params->instruction = *c_instruction;
        }
    }
    if (params != nullptr && g_speaker_field != nullptr) {
        *j_speaker = (jstring)env->GetObjectField(params, g_speaker_field);
        if (*j_speaker != nullptr) {
            *c_speaker = env->GetStringUTFChars(*j_speaker, nullptr);
            c_params->speaker = *c_speaker;
        }
    }
}

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }

    jclass local_result_class = env->FindClass("com/qwen/tts/studio/engine/QwenEngine$NativeResult");
    if (local_result_class == nullptr) {
        LOGE("Could not find NativeResult class");
        return JNI_ERR;
    }
    g_result_class = reinterpret_cast<jclass>(env->NewGlobalRef(local_result_class));
    env->DeleteLocalRef(local_result_class);

    g_result_constructor = env->GetMethodID(g_result_class, "<init>", "([FIZLjava/lang/String;J)V");
    if (g_result_constructor == nullptr) {
        LOGE("Could not find NativeResult constructor");
        return JNI_ERR;
    }

    jclass local_caps_class = env->FindClass("com/qwen/tts/studio/engine/QwenEngine$NativeCapabilities");
    if (local_caps_class == nullptr) {
        LOGE("Could not find NativeCapabilities class");
        return JNI_ERR;
    }
    g_caps_class = reinterpret_cast<jclass>(env->NewGlobalRef(local_caps_class));
    env->DeleteLocalRef(local_caps_class);

    g_caps_constructor = env->GetMethodID(g_caps_class, "<init>", "(ZZZZIII)V");
    if (g_caps_constructor == nullptr) {
        LOGE("Could not find NativeCapabilities constructor");
        return JNI_ERR;
    }

    jclass local_params_class = env->FindClass("com/qwen/tts/studio/engine/QwenEngine$NativeParams");
    if (local_params_class == nullptr) {
        LOGE("Could not find NativeParams class");
        return JNI_ERR;
    }
    g_params_class = reinterpret_cast<jclass>(env->NewGlobalRef(local_params_class));
    env->DeleteLocalRef(local_params_class);

    g_lang_id_field = env->GetFieldID(g_params_class, "languageId", "I");
    if (g_lang_id_field == nullptr) {
        LOGE("Could not find languageId field in NativeParams");
        return JNI_ERR;
    }

    g_instruction_field = env->GetFieldID(g_params_class, "instruction", "Ljava/lang/String;");
    if (g_instruction_field == nullptr) {
        LOGE("Could not find instruction field in NativeParams");
        return JNI_ERR;
    }

    g_speaker_field = env->GetFieldID(g_params_class, "speaker", "Ljava/lang/String;");
    if (g_speaker_field == nullptr) {
        LOGE("Could not find speaker field in NativeParams");
        return JNI_ERR;
    }

    g_max_audio_tokens_field = env->GetFieldID(g_params_class, "maxAudioTokens", "I");
    if (g_max_audio_tokens_field == nullptr) {
        LOGE("Could not find maxAudioTokens field in NativeParams");
        return JNI_ERR;
    }

    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK) {
        if (g_result_class != nullptr) env->DeleteGlobalRef(g_result_class);
        if (g_caps_class != nullptr) env->DeleteGlobalRef(g_caps_class);
        if (g_params_class != nullptr) env->DeleteGlobalRef(g_params_class);
    }
}

JNIEXPORT jlong JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeInit(JNIEnv* env, jobject thiz) {
    return reinterpret_cast<jlong>(qwen3_tts_init());
}

JNIEXPORT void JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeFree(JNIEnv* env, jobject thiz, jlong ctx_ptr) {
    if (ctx_ptr == 0) return;
    clear_progress_callback(env, ctx_ptr);
    qwen3_tts_free(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr));
}

JNIEXPORT jboolean JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeSetBackendPreference(
    JNIEnv* env, jobject thiz, jint preference
) {
    return qwen3_tts_set_backend_preference(static_cast<int32_t>(preference)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeSetCpuThreads(
    JNIEnv* env, jobject thiz, jint n_threads
) {
    return qwen3_tts_set_cpu_threads(static_cast<int32_t>(n_threads)) != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeGetCpuThreads(
    JNIEnv* env, jobject thiz
) {
    return static_cast<jint>(qwen3_tts_get_cpu_threads());
}

JNIEXPORT jboolean JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeSetProgressCallback(
    JNIEnv* env, jobject thiz, jlong ctx_ptr, jobject callback
) {
    if (ctx_ptr == 0) return JNI_FALSE;

    clear_progress_callback(env, ctx_ptr);
    if (callback == nullptr) {
        return JNI_TRUE;
    }

    jclass callback_class = env->GetObjectClass(callback);
    if (callback_class == nullptr) {
        return JNI_FALSE;
    }
    jmethodID on_progress = env->GetMethodID(callback_class, "onProgress", "(II)V");
    env->DeleteLocalRef(callback_class);
    if (on_progress == nullptr) {
        env->ExceptionClear();
        return JNI_FALSE;
    }

    auto* state = new ProgressCallbackState();
    env->GetJavaVM(&state->vm);
    state->callback = env->NewGlobalRef(callback);
    state->on_progress = on_progress;
    if (state->callback == nullptr) {
        delete state;
        return JNI_FALSE;
    }

    {
        std::lock_guard<std::mutex> lock(g_progress_mutex);
        g_progress_callbacks[ctx_ptr] = state;
    }

    qwen3_tts_set_progress_callback(
        reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr),
        [](int tokens_generated, int max_tokens, void* user_data) {
            auto* state = static_cast<ProgressCallbackState*>(user_data);
            if (state == nullptr || state->vm == nullptr || state->callback == nullptr) {
                return;
            }

            JNIEnv* env = nullptr;
            bool did_attach = false;
            jint get_env = state->vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
            if (get_env == JNI_EDETACHED) {
                void* attached_env = nullptr;
                if (state->vm->AttachCurrentThread(&attached_env, nullptr) != JNI_OK) {
                    return;
                }
                env = static_cast<JNIEnv*>(attached_env);
                did_attach = true;
            } else if (get_env != JNI_OK || env == nullptr) {
                return;
            }

            env->CallVoidMethod(
                state->callback,
                state->on_progress,
                static_cast<jint>(tokens_generated),
                static_cast<jint>(max_tokens)
            );
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
            }
            if (did_attach) {
                state->vm->DetachCurrentThread();
            }
        },
        state
    );

    return JNI_TRUE;
}

JNIEXPORT jint JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeGetCompiledBackendMask(
    JNIEnv* env, jobject thiz
) {
    return static_cast<jint>(qwen3_tts_get_compiled_backend_mask());
}

JNIEXPORT jstring JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeGetActiveBackendName(
    JNIEnv* env, jobject thiz
) {
    char* name = qwen3_tts_get_active_backend_name();
    if (!name) return nullptr;
    jstring result = env->NewStringUTF(name);
    qwen3_tts_free_string(name);
    return result;
}

JNIEXPORT jboolean JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeLoadModels(
    JNIEnv* env, jobject thiz, jlong ctx_ptr, jstring model_dir, jstring model_name
) {
    if (ctx_ptr == 0 || model_dir == nullptr) return JNI_FALSE;
    const char* c_model_dir = env->GetStringUTFChars(model_dir, nullptr);
    if (c_model_dir == nullptr) return JNI_FALSE; // Check for OOM

    const char* c_model_name = nullptr;
    if (model_name != nullptr) {
        c_model_name = env->GetStringUTFChars(model_name, nullptr);
        if (c_model_name == nullptr) {
            env->ReleaseStringUTFChars(model_dir, c_model_dir);
            return JNI_FALSE;
        }
    }

    int32_t result = qwen3_tts_load_models_with_name(
        reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_model_dir, c_model_name);

    if (c_model_name) env->ReleaseStringUTFChars(model_name, c_model_name);
    env->ReleaseStringUTFChars(model_dir, c_model_dir);
    return result != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeLoadIclPromptEncoder(
    JNIEnv* env, jobject thiz, jlong ctx_ptr, jstring model_dir, jstring model_name
) {
    if (ctx_ptr == 0 || model_dir == nullptr) return JNI_FALSE;
    const char* c_model_dir = env->GetStringUTFChars(model_dir, nullptr);
    if (c_model_dir == nullptr) return JNI_FALSE;

    const char* c_model_name = nullptr;
    if (model_name != nullptr) {
        c_model_name = env->GetStringUTFChars(model_name, nullptr);
        if (c_model_name == nullptr) {
            env->ReleaseStringUTFChars(model_dir, c_model_dir);
            return JNI_FALSE;
        }
    }

    int32_t result = qwen3_tts_load_icl_prompt_encoder_with_name(
        reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_model_dir, c_model_name);

    if (c_model_name) env->ReleaseStringUTFChars(model_name, c_model_name);
    env->ReleaseStringUTFChars(model_dir, c_model_dir);
    return result != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jobject JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeSynthesize(
    JNIEnv* env, jobject thiz, jlong ctx_ptr, jstring text, jstring reference_wav, jstring speaker_embedding_path, jobject params
) {
    if (ctx_ptr == 0 || text == nullptr) return nullptr;

    const char* c_text = env->GetStringUTFChars(text, nullptr);
    if (c_text == nullptr) return nullptr;

    const char* c_ref_wav = nullptr;
    const char* c_speaker_embedding = nullptr;
    if (reference_wav != nullptr) {
        c_ref_wav = env->GetStringUTFChars(reference_wav, nullptr);
        if (c_ref_wav == nullptr) {
            env->ReleaseStringUTFChars(text, c_text);
            return nullptr;
        }
    }
    if (speaker_embedding_path != nullptr) {
        c_speaker_embedding = env->GetStringUTFChars(speaker_embedding_path, nullptr);
        if (c_speaker_embedding == nullptr) {
            if (c_ref_wav) env->ReleaseStringUTFChars(reference_wav, c_ref_wav);
            env->ReleaseStringUTFChars(text, c_text);
            return nullptr;
        }
    }

    qwen3_tts_params_t c_params = {4096, 0.9f, 1.0f, 50, 4, 0, 1, 1.05f, -1, nullptr, nullptr, 2.0f};
    
    jstring j_instruction = nullptr;
    const char* c_instruction = nullptr;
    jstring j_speaker = nullptr;
    const char* c_speaker = nullptr;

    if (params != nullptr && g_lang_id_field != nullptr) {
        c_params.language_id = env->GetIntField(params, g_lang_id_field);
    }
    if (params != nullptr && g_max_audio_tokens_field != nullptr) {
        c_params.max_audio_tokens = env->GetIntField(params, g_max_audio_tokens_field);
    }
    if (params != nullptr && g_instruction_field != nullptr) {
        j_instruction = (jstring)env->GetObjectField(params, g_instruction_field);
        if (j_instruction != nullptr) {
            c_instruction = env->GetStringUTFChars(j_instruction, nullptr);
            c_params.instruction = c_instruction;
        }
    }
    if (params != nullptr && g_speaker_field != nullptr) {
        j_speaker = (jstring)env->GetObjectField(params, g_speaker_field);
        if (j_speaker != nullptr) {
            c_speaker = env->GetStringUTFChars(j_speaker, nullptr);
            c_params.speaker = c_speaker;
        }
    }

    qwen3_tts_result_t c_result;
    if (c_speaker_embedding && strlen(c_speaker_embedding) > 0) {
        c_result = qwen3_tts_synthesize_with_speaker_embedding(
            reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_text, c_speaker_embedding, c_params);
    } else if (c_ref_wav && strlen(c_ref_wav) > 0) {
        c_result = qwen3_tts_synthesize_with_voice(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_text, c_ref_wav, c_params);
    } else {
        c_result = qwen3_tts_synthesize(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_text, c_params);
    }

    env->ReleaseStringUTFChars(text, c_text);
    if (c_ref_wav) env->ReleaseStringUTFChars(reference_wav, c_ref_wav);
    if (c_speaker_embedding) env->ReleaseStringUTFChars(speaker_embedding_path, c_speaker_embedding);
    if (c_instruction) env->ReleaseStringUTFChars(j_instruction, c_instruction);
    if (c_speaker) env->ReleaseStringUTFChars(j_speaker, c_speaker);

    jobject result_obj = make_native_result(env, c_result);
    qwen3_tts_free_result(c_result);
    return result_obj;
}

JNIEXPORT jobject JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeSynthesizeWithIclPrompt(
    JNIEnv* env, jobject thiz, jlong ctx_ptr, jstring text, jstring icl_prompt_path, jobject params
) {
    if (ctx_ptr == 0 || text == nullptr || icl_prompt_path == nullptr) return nullptr;

    const char* c_text = env->GetStringUTFChars(text, nullptr);
    if (c_text == nullptr) return nullptr;
    const char* c_icl_prompt = env->GetStringUTFChars(icl_prompt_path, nullptr);
    if (c_icl_prompt == nullptr) {
        env->ReleaseStringUTFChars(text, c_text);
        return nullptr;
    }

    qwen3_tts_params_t c_params = {4096, 0.9f, 1.0f, 50, 4, 0, 1, 1.05f, -1, nullptr, nullptr, 2.0f};

    jstring j_instruction = nullptr;
    const char* c_instruction = nullptr;
    jstring j_speaker = nullptr;
    const char* c_speaker = nullptr;
    fill_params_from_java(env, params, &c_params, &j_instruction, &c_instruction, &j_speaker, &c_speaker);

    qwen3_tts_result_t c_result = qwen3_tts_synthesize_with_icl_prompt(
        reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_text, c_icl_prompt, c_params);

    env->ReleaseStringUTFChars(text, c_text);
    env->ReleaseStringUTFChars(icl_prompt_path, c_icl_prompt);
    if (c_instruction) env->ReleaseStringUTFChars(j_instruction, c_instruction);
    if (c_speaker) env->ReleaseStringUTFChars(j_speaker, c_speaker);

    jobject result_obj = make_native_result(env, c_result);
    qwen3_tts_free_result(c_result);
    return result_obj;
}

JNIEXPORT jobject JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeSynthesizeStreaming(
    JNIEnv* env,
    jobject thiz,
    jlong ctx_ptr,
    jstring text,
    jstring reference_wav,
    jstring speaker_embedding_path,
    jstring icl_prompt_path,
    jobject params,
    jfloat chunk_seconds,
    jfloat left_context_seconds,
    jboolean collect_audio,
    jobject callback
) {
    if (ctx_ptr == 0 || text == nullptr || callback == nullptr) return nullptr;

    const char* c_text = env->GetStringUTFChars(text, nullptr);
    if (c_text == nullptr) return nullptr;

    const char* c_ref_wav = nullptr;
    const char* c_speaker_embedding = nullptr;
    const char* c_icl_prompt = nullptr;
    if (reference_wav != nullptr) {
        c_ref_wav = env->GetStringUTFChars(reference_wav, nullptr);
        if (c_ref_wav == nullptr) {
            env->ReleaseStringUTFChars(text, c_text);
            return nullptr;
        }
    }
    if (speaker_embedding_path != nullptr) {
        c_speaker_embedding = env->GetStringUTFChars(speaker_embedding_path, nullptr);
        if (c_speaker_embedding == nullptr) {
            if (c_ref_wav) env->ReleaseStringUTFChars(reference_wav, c_ref_wav);
            env->ReleaseStringUTFChars(text, c_text);
            return nullptr;
        }
    }
    if (icl_prompt_path != nullptr) {
        c_icl_prompt = env->GetStringUTFChars(icl_prompt_path, nullptr);
        if (c_icl_prompt == nullptr) {
            if (c_ref_wav) env->ReleaseStringUTFChars(reference_wav, c_ref_wav);
            if (c_speaker_embedding) env->ReleaseStringUTFChars(speaker_embedding_path, c_speaker_embedding);
            env->ReleaseStringUTFChars(text, c_text);
            return nullptr;
        }
    }

    jclass callback_class = env->GetObjectClass(callback);
    jmethodID on_audio_chunk = env->GetMethodID(
        callback_class,
        "onAudioChunk",
        "([FIJJIIIIIF)Z"
    );
    env->DeleteLocalRef(callback_class);
    if (on_audio_chunk == nullptr) {
        if (c_ref_wav) env->ReleaseStringUTFChars(reference_wav, c_ref_wav);
        if (c_speaker_embedding) env->ReleaseStringUTFChars(speaker_embedding_path, c_speaker_embedding);
        if (c_icl_prompt) env->ReleaseStringUTFChars(icl_prompt_path, c_icl_prompt);
        env->ReleaseStringUTFChars(text, c_text);
        return nullptr;
    }

    qwen3_tts_params_t c_params = {4096, 0.9f, 1.0f, 50, 4, 0, 1, 1.05f, -1, nullptr, nullptr, 2.0f};

    jstring j_instruction = nullptr;
    const char* c_instruction = nullptr;
    jstring j_speaker = nullptr;
    const char* c_speaker = nullptr;
    fill_params_from_java(env, params, &c_params, &j_instruction, &c_instruction, &j_speaker, &c_speaker);

    qwen3_tts_streaming_params_t c_stream_params = {
        c_params,
        chunk_seconds,
        left_context_seconds,
        collect_audio ? 1 : 0
    };

    jobject callback_ref = env->NewGlobalRef(callback);
    struct CallbackState {
        JNIEnv* env;
        jobject callback;
        jmethodID method;
    } state = {env, callback_ref, on_audio_chunk};

    qwen3_tts_audio_chunk_callback c_callback =
        [](const qwen3_tts_audio_chunk_t* chunk, void* user_data) -> int32_t {
        CallbackState* state = static_cast<CallbackState*>(user_data);
        if (chunk == nullptr || state == nullptr || state->env == nullptr || state->callback == nullptr) {
            return 0;
        }

        JNIEnv* env = state->env;
        jfloatArray audio = env->NewFloatArray(chunk->n_samples);
        if (audio == nullptr) {
            env->ExceptionClear();
            return 0;
        }
        if (chunk->n_samples > 0 && chunk->samples != nullptr) {
            env->SetFloatArrayRegion(audio, 0, chunk->n_samples, chunk->samples);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                env->DeleteLocalRef(audio);
                return 0;
            }
        }

        jboolean keep_going = env->CallBooleanMethod(
            state->callback,
            state->method,
            audio,
            (jint)chunk->sample_rate,
            (jlong)chunk->start_sample,
            (jlong)chunk->end_sample,
            (jint)chunk->start_frame,
            (jint)chunk->end_frame,
            (jint)chunk->start_text_byte,
            (jint)chunk->end_text_byte,
            (jint)chunk->text_alignment_kind,
            (jfloat)chunk->confidence
        );
        env->DeleteLocalRef(audio);

        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
            return 0;
        }

        return keep_going == JNI_TRUE ? 1 : 0;
    };

    qwen3_tts_result_t c_result;
    if (c_icl_prompt && strlen(c_icl_prompt) > 0) {
        c_result = qwen3_tts_synthesize_with_icl_prompt_streaming(
            reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr),
            c_text,
            c_icl_prompt,
            c_stream_params,
            c_callback,
            &state
        );
    } else if (c_speaker_embedding && strlen(c_speaker_embedding) > 0) {
        c_result = qwen3_tts_synthesize_with_speaker_embedding_streaming(
            reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr),
            c_text,
            c_speaker_embedding,
            c_stream_params,
            c_callback,
            &state
        );
    } else if (c_ref_wav && strlen(c_ref_wav) > 0) {
        c_result = qwen3_tts_synthesize_with_voice_streaming(
            reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr),
            c_text,
            c_ref_wav,
            c_stream_params,
            c_callback,
            &state
        );
    } else {
        c_result = qwen3_tts_synthesize_streaming(
            reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr),
            c_text,
            c_stream_params,
            c_callback,
            &state
        );
    }

    env->DeleteGlobalRef(callback_ref);
    if (c_ref_wav) env->ReleaseStringUTFChars(reference_wav, c_ref_wav);
    if (c_speaker_embedding) env->ReleaseStringUTFChars(speaker_embedding_path, c_speaker_embedding);
    if (c_icl_prompt) env->ReleaseStringUTFChars(icl_prompt_path, c_icl_prompt);
    if (c_instruction) env->ReleaseStringUTFChars(j_instruction, c_instruction);
    if (c_speaker) env->ReleaseStringUTFChars(j_speaker, c_speaker);
    env->ReleaseStringUTFChars(text, c_text);

    jobject result_obj = make_native_result(env, c_result);
    qwen3_tts_free_result(c_result);
    return result_obj;
}

JNIEXPORT jboolean JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeExtractSpeakerEmbedding(
    JNIEnv* env, jobject thiz, jlong ctx_ptr, jstring reference_wav, jstring output_path
) {
    if (ctx_ptr == 0 || reference_wav == nullptr || output_path == nullptr) return JNI_FALSE;

    const char* c_ref_wav = env->GetStringUTFChars(reference_wav, nullptr);
    if (c_ref_wav == nullptr) return JNI_FALSE;
    const char* c_output_path = env->GetStringUTFChars(output_path, nullptr);
    if (c_output_path == nullptr) {
        env->ReleaseStringUTFChars(reference_wav, c_ref_wav);
        return JNI_FALSE;
    }

    const int32_t ok = qwen3_tts_extract_speaker_embedding(
        reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_ref_wav, c_output_path);

    env->ReleaseStringUTFChars(reference_wav, c_ref_wav);
    env->ReleaseStringUTFChars(output_path, c_output_path);
    return ok != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeExtractIclPrompt(
    JNIEnv* env, jobject thiz, jlong ctx_ptr, jstring reference_wav, jstring reference_text, jstring output_path
) {
    if (ctx_ptr == 0 || reference_wav == nullptr || reference_text == nullptr || output_path == nullptr) {
        return JNI_FALSE;
    }

    const char* c_ref_wav = env->GetStringUTFChars(reference_wav, nullptr);
    if (c_ref_wav == nullptr) return JNI_FALSE;
    const char* c_reference_text = env->GetStringUTFChars(reference_text, nullptr);
    if (c_reference_text == nullptr) {
        env->ReleaseStringUTFChars(reference_wav, c_ref_wav);
        return JNI_FALSE;
    }
    const char* c_output_path = env->GetStringUTFChars(output_path, nullptr);
    if (c_output_path == nullptr) {
        env->ReleaseStringUTFChars(reference_wav, c_ref_wav);
        env->ReleaseStringUTFChars(reference_text, c_reference_text);
        return JNI_FALSE;
    }

    const int32_t ok = qwen3_tts_extract_icl_prompt(
        reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr),
        c_ref_wav,
        c_reference_text,
        c_output_path);

    env->ReleaseStringUTFChars(reference_wav, c_ref_wav);
    env->ReleaseStringUTFChars(reference_text, c_reference_text);
    env->ReleaseStringUTFChars(output_path, c_output_path);
    return ok != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeGetAvailableSpeakers(
    JNIEnv* env, jobject thiz, jlong ctx_ptr
) {
    if (ctx_ptr == 0) return nullptr;

    char* speakers = qwen3_tts_get_available_speakers(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr));
    if (!speakers) return nullptr;

    jstring result = env->NewStringUTF(speakers);
    qwen3_tts_free_string(speakers);
    return result;
}

JNIEXPORT jstring JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeGetLastError(
    JNIEnv* env, jobject thiz, jlong ctx_ptr
) {
    if (ctx_ptr == 0) return nullptr;

    char* error = qwen3_tts_get_last_error(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr));
    if (!error) return nullptr;

    jstring result = env->NewStringUTF(error);
    qwen3_tts_free_string(error);
    return result;
}

JNIEXPORT jobject JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeGetModelCapabilities(
    JNIEnv* env, jobject thiz, jlong ctx_ptr
) {
    if (ctx_ptr == 0 || g_caps_class == nullptr || g_caps_constructor == nullptr) {
        return nullptr;
    }

    const qwen3_tts_model_capabilities_t caps =
        qwen3_tts_get_model_capabilities(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr));

    return env->NewObject(
        g_caps_class,
        g_caps_constructor,
        (jboolean)(caps.loaded != 0),
        (jboolean)(caps.supports_voice_clone != 0),
        (jboolean)(caps.supports_named_speakers != 0),
        (jboolean)(caps.supports_instruction != 0),
        (jint)caps.speaker_embedding_dim,
        (jint)caps.model_kind,
        (jint)caps.speaker_count
    );
}

}
