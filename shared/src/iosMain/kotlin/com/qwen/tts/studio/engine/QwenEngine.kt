package com.qwen.tts.studio.engine

import kotlinx.cinterop.*
import com.example.qwen3tts.cinterop.*

/**
 * iOS implementation using C Interop.
 */
actual class QwenEngine actual constructor() {
    private var nativePtr: CPointer<qwen3_tts_context>? = qwen3_tts_init()

    actual fun loadModels(modelDir: String): Boolean {
        return qwen3_tts_load_models(nativePtr, modelDir) != 0
    }

    actual fun synthesize(text: String, referenceWav: String?): NativeResult {
        memScoped {
            val params = alloc<qwen3_tts_params_t>()
            // Set defaults (must match C implementation)
            params.max_audio_tokens = 4096
            params.temperature = 0.9f
            params.top_p = 1.0f
            params.top_k = 50
            params.n_threads = 4
            params.print_progress = 0
            params.print_timing = 1
            params.repetition_penalty = 1.05f
            params.language_id = 2050

            val cResult = if (referenceWav != null && referenceWav.isNotEmpty()) {
                qwen3_tts_synthesize_with_voice(nativePtr, text, referenceWav, params.readValue())
            } else {
                qwen3_tts_synthesize(nativePtr, text, params.readValue())
            }
            
            val audio = if (cResult.audio_len > 0) {
                FloatArray(cResult.audio_len) { i -> cResult.audio!![i] }
            } else null
            
            val result = NativeResult(
                audio = audio,
                sampleRate = cResult.sample_rate,
                success = cResult.success != 0,
                errorMsg = cResult.error_msg?.toKString(),
                timeMs = cResult.t_total_ms
            )
            
            qwen3_tts_free_result(cResult.readValue())
            return result
        }
    }

    actual fun close() {
        if (nativePtr != null) {
            qwen3_tts_free(nativePtr)
            nativePtr = null
        }
    }

    actual class NativeResult actual constructor(
        actual val audio: FloatArray?,
        actual val sampleRate: Int,
        actual val success: Boolean,
        actual val errorMsg: String?,
        actual val timeMs: Long
    )
}
