package com.qwen.tts.studio.engine

import kotlinx.cinterop.*
import com.example.qwen3tts.cinterop.*

/**
 * iOS implementation using C Interop.
 */
actual class QwenEngine actual constructor() {
    private var nativePtr: CPointer<qwen3_tts_context>? = qwen3_tts_init()

    actual fun loadModels(modelDir: String, modelName: String?): Boolean {
        return qwen3_tts_load_models_with_name(nativePtr, modelDir, modelName) != 0
    }

    actual fun loadIclPromptEncoder(modelDir: String, modelName: String?): Boolean {
        return qwen3_tts_load_icl_prompt_encoder_with_name(nativePtr, modelDir, modelName) != 0
    }

    actual fun synthesize(
        text: String,
        referenceWav: String?,
        speakerEmbeddingPath: String?,
        params: NativeParams
    ): NativeResult {
        memScoped {
            val cParams = alloc<qwen3_tts_params_t>()
            // Set defaults (must match C implementation)
            cParams.max_audio_tokens = 4096
            cParams.temperature = 0.9f
            cParams.top_p = 1.0f
            cParams.top_k = 50
            cParams.n_threads = 4
            cParams.print_progress = 0
            cParams.print_timing = 1
            cParams.repetition_penalty = 1.05f
            cParams.language_id = params.languageId
            cParams.instruction = params.instruction?.cstr?.getPointer(this)
            cParams.speaker = params.speaker?.cstr?.getPointer(this)

            val cResult = if (speakerEmbeddingPath != null && speakerEmbeddingPath.isNotEmpty()) {
                qwen3_tts_synthesize_with_speaker_embedding(
                    nativePtr,
                    text,
                    speakerEmbeddingPath,
                    cParams.readValue()
                )
            } else if (referenceWav != null && referenceWav.isNotEmpty()) {
                qwen3_tts_synthesize_with_voice(nativePtr, text, referenceWav, cParams.readValue())
            } else {
                qwen3_tts_synthesize(nativePtr, text, cParams.readValue())
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

    actual fun synthesizeWithIclPrompt(
        text: String,
        iclPromptPath: String,
        params: NativeParams
    ): NativeResult {
        memScoped {
            val cParams = alloc<qwen3_tts_params_t>()
            cParams.max_audio_tokens = 4096
            cParams.temperature = 0.9f
            cParams.top_p = 1.0f
            cParams.top_k = 50
            cParams.n_threads = 4
            cParams.print_progress = 0
            cParams.print_timing = 1
            cParams.repetition_penalty = 1.05f
            cParams.language_id = params.languageId
            cParams.instruction = params.instruction?.cstr?.getPointer(this)
            cParams.speaker = params.speaker?.cstr?.getPointer(this)

            val cResult = qwen3_tts_synthesize_with_icl_prompt(
                nativePtr,
                text,
                iclPromptPath,
                cParams.readValue()
            )
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

    actual fun extractSpeakerEmbedding(referenceWav: String, outputPath: String): Boolean {
        return qwen3_tts_extract_speaker_embedding(nativePtr, referenceWav, outputPath) != 0
    }

    actual fun extractIclPrompt(referenceWav: String, referenceText: String, outputPath: String): Boolean {
        return qwen3_tts_extract_icl_prompt(nativePtr, referenceWav, referenceText, outputPath) != 0
    }

    actual fun getAvailableSpeakers(): List<String> {
        val speakersRawPtr = qwen3_tts_get_available_speakers(nativePtr) ?: return emptyList()
        val speakersRaw = speakersRawPtr.toKString()
        qwen3_tts_free_string(speakersRawPtr)
        if (speakersRaw.isBlank()) return emptyList()
        return speakersRaw
            .lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .toList()
    }

    actual fun close() {
        if (nativePtr != null) {
            qwen3_tts_free(nativePtr)
            nativePtr = null
        }
    }

    actual class NativeParams actual constructor(
        actual val languageId: Int = 2050,
        actual val instruction: String? = null,
        actual val speaker: String? = null
    )

    actual class NativeResult actual constructor(
        actual val audio: FloatArray?,
        actual val sampleRate: Int,
        actual val success: Boolean,
        actual val errorMsg: String?,
        actual val timeMs: Long
    )
}
