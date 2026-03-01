#include <jni.h>
#include "qwen3_tts_c.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>

#define LOGE(...) fprintf(stderr, "[QwenEngine_JNI] " __VA_ARGS__); fprintf(stderr, "\n")

static jclass g_result_class = nullptr;
static jmethodID g_result_constructor = nullptr;

static jclass g_params_class = nullptr;
static jfieldID g_lang_id_field = nullptr;

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

    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK) {
        if (g_result_class != nullptr) env->DeleteGlobalRef(g_result_class);
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

JNIEXPORT jboolean JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeLoadModels(JNIEnv* env, jobject thiz, jlong ctx_ptr, jstring model_dir) {
    if (ctx_ptr == 0 || model_dir == nullptr) return JNI_FALSE;
    const char* c_model_dir = env->GetStringUTFChars(model_dir, nullptr);
    if (c_model_dir == nullptr) return JNI_FALSE; // Check for OOM

    int32_t result = qwen3_tts_load_models(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_model_dir);
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

    qwen3_tts_params_t c_params = {4096, 0.9f, 1.0f, 50, 4, 0, 1, 1.05f, 2050};
    
    if (params != nullptr && g_lang_id_field != nullptr) {
        c_params.language_id = env->GetIntField(params, g_lang_id_field);
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

}
