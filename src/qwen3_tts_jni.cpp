#include <jni.h>
#include "qwen3_tts_c.h"
#include <string>
#include <vector>
#include <cstdio>

#define LOGE(...) fprintf(stderr, "[QwenEngine_JNI] " __VA_ARGS__); fprintf(stderr, "\n")

extern "C" {

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
    int32_t result = qwen3_tts_load_models(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_model_dir);
    env->ReleaseStringUTFChars(model_dir, c_model_dir);
    return result != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jobject JNICALL Java_com_qwen_tts_studio_engine_QwenEngine_nativeSynthesize(JNIEnv* env, jobject thiz, jlong ctx_ptr, jstring text, jstring reference_wav, jobject params) {
    if (ctx_ptr == 0 || text == nullptr) return nullptr;

    qwen3_tts_params_t c_params = {4096, 0.9f, 1.0f, 50, 4, 0, 1, 1.05f, 2050};
    
    const char* c_text = env->GetStringUTFChars(text, nullptr);
    const char* c_ref_wav = reference_wav ? env->GetStringUTFChars(reference_wav, nullptr) : nullptr;
    
    qwen3_tts_result_t c_result;
    if (c_ref_wav && strlen(c_ref_wav) > 0) {
        c_result = qwen3_tts_synthesize_with_voice(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_text, c_ref_wav, c_params);
    } else {
        c_result = qwen3_tts_synthesize(reinterpret_cast<qwen3_tts_context_t*>(ctx_ptr), c_text, c_params);
    }

    env->ReleaseStringUTFChars(text, c_text);
    if (c_ref_wav) env->ReleaseStringUTFChars(reference_wav, c_ref_wav);

    jclass result_class = env->FindClass("com/qwen/tts/studio/engine/QwenEngine$NativeResult");
    if (result_class == nullptr) {
        LOGE("Could not find NativeResult class");
        qwen3_tts_free_result(c_result);
        return nullptr;
    }

    jmethodID constructor = env->GetMethodID(result_class, "<init>", "([FIZLjava/lang/String;J)V");
    if (constructor == nullptr) {
        LOGE("Could not find NativeResult constructor");
        qwen3_tts_free_result(c_result);
        return nullptr;
    }

    jfloatArray audio_array = nullptr;
    if (c_result.audio_len > 0 && c_result.audio != nullptr) {
        audio_array = env->NewFloatArray(c_result.audio_len);
        env->SetFloatArrayRegion(audio_array, 0, c_result.audio_len, c_result.audio);
    }

    jstring error_msg = nullptr;
    if (c_result.error_msg) {
        error_msg = env->NewStringUTF(c_result.error_msg);
    }

    jobject result_obj = env->NewObject(result_class, constructor, 
                                        audio_array, 
                                        (jint)c_result.sample_rate, 
                                        (jboolean)(c_result.success != 0), 
                                        error_msg, 
                                        (jlong)c_result.t_total_ms);

    qwen3_tts_free_result(c_result);
    return result_obj;
}

}
