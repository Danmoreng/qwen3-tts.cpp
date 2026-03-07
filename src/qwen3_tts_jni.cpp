#include <jni.h>
#include "qwen3_tts_c.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>

#define LOGE(...) fprintf(stderr, "[QwenEngine_JNI] " __VA_ARGS__); fprintf(stderr, "\n")

static jclass g_result_class = nullptr;
static jmethodID g_result_constructor = nullptr;
static jclass g_caps_class = nullptr;
static jmethodID g_caps_constructor = nullptr;

static jclass g_params_class = nullptr;
static jfieldID g_lang_id_field = nullptr;
static jfieldID g_instruction_field = nullptr;
static jfieldID g_speaker_field = nullptr;

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
    qwen3_tts_free(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr));
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

    qwen3_tts_params_t c_params = {4096, 0.9f, 1.0f, 50, 4, 0, 1, 1.05f, 2050, nullptr, nullptr};
    
    jstring j_instruction = nullptr;
    const char* c_instruction = nullptr;
    jstring j_speaker = nullptr;
    const char* c_speaker = nullptr;

    if (params != nullptr && g_lang_id_field != nullptr) {
        c_params.language_id = env->GetIntField(params, g_lang_id_field);
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

    if (g_result_class == nullptr || g_result_constructor == nullptr) {
        qwen3_tts_free_result(c_result);
        return nullptr;
    }

    jfloatArray audio_array = nullptr;
    if (c_result.audio_len > 0 && c_result.audio != nullptr) {
        audio_array = env->NewFloatArray(c_result.audio_len);
        if (audio_array != nullptr) {
            env->SetFloatArrayRegion(audio_array, 0, c_result.audio_len, c_result.audio);
        } else {
            // NewFloatArray threw OutOfMemoryError, clear it so we can safely return null or handle it
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

    jobject result_obj = env->NewObject(g_result_class, g_result_constructor, 
                                        audio_array, 
                                        (jint)c_result.sample_rate, 
                                        (jboolean)(c_result.success != 0), 
                                        error_msg, 
                                        (jlong)c_result.t_total_ms);

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
