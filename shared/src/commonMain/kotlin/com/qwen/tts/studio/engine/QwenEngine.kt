package com.qwen.tts.studio.engine

/**
 * Common interface for QwenEngine across all platforms.
 */
expect class QwenEngine() {
    fun loadModels(modelDir: String): Boolean
    fun synthesize(text: String, referenceWav: String? = null): NativeResult
    fun close()

    class NativeResult(
        val audio: FloatArray?,
        val sampleRate: Int,
        val success: Boolean,
        val errorMsg: String?,
        val timeMs: Long
    )
}
