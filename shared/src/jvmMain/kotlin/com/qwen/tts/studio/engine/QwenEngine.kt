package com.qwen.tts.studio.engine

/**
 * JVM/Android implementation using JNI.
 */
actual class QwenEngine actual constructor() {
    private var nativePtr: Long = 0

    class NativeCapabilities(
        val loaded: Boolean,
        val supportsCloning: Boolean,
        val supportsNamedSpeakers: Boolean,
        val supportsInstruction: Boolean,
        val speakerEmbeddingDim: Int,
        val modelKind: Int,
        val speakerCount: Int
    )

    init {
        System.loadLibrary("qwen3_tts_jni")
        nativePtr = nativeInit()
    }

    actual fun loadModels(modelDir: String, modelName: String?): Boolean =
        nativeLoadModels(nativePtr, modelDir, modelName)

    actual fun synthesize(
        text: String,
        referenceWav: String?,
        speakerEmbeddingPath: String?,
        params: NativeParams
    ): NativeResult =
        nativeSynthesize(nativePtr, text, referenceWav, speakerEmbeddingPath, params)

    actual fun extractSpeakerEmbedding(referenceWav: String, outputPath: String): Boolean =
        nativeExtractSpeakerEmbedding(nativePtr, referenceWav, outputPath)

    actual fun getAvailableSpeakers(): List<String> {
        val raw = nativeGetAvailableSpeakers(nativePtr).orEmpty()
        if (raw.isBlank()) return emptyList()
        return raw
            .lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .toList()
    }

    actual fun close() {
        if (nativePtr != 0L) {
            nativeFree(nativePtr)
            nativePtr = 0
        }
    }

    fun getModelCapabilities(): NativeCapabilities? =
        nativeGetModelCapabilities(nativePtr)

    private external fun nativeInit(): Long
    private external fun nativeFree(ptr: Long)
    private external fun nativeLoadModels(ptr: Long, modelDir: String, modelName: String?): Boolean
    private external fun nativeSynthesize(
        ptr: Long,
        text: String,
        referenceWav: String?,
        speakerEmbeddingPath: String?,
        params: NativeParams?
    ): NativeResult
    private external fun nativeExtractSpeakerEmbedding(
        ptr: Long,
        referenceWav: String,
        outputPath: String
    ): Boolean
    private external fun nativeGetAvailableSpeakers(ptr: Long): String?
    private external fun nativeGetModelCapabilities(ptr: Long): NativeCapabilities?

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
