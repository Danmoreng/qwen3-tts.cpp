package com.qwen.tts.studio.engine

/**
 * Common interface for QwenEngine across all platforms.
 */
expect class QwenEngine() {
    fun loadModels(modelDir: String, modelName: String? = null): Boolean
    fun loadIclPromptEncoder(modelDir: String, modelName: String? = null): Boolean
    fun synthesize(
        text: String,
        referenceWav: String? = null,
        speakerEmbeddingPath: String? = null,
        params: NativeParams = NativeParams()
    ): NativeResult
    fun synthesizeWithIclPrompt(
        text: String,
        iclPromptPath: String,
        params: NativeParams = NativeParams()
    ): NativeResult
    fun extractSpeakerEmbedding(referenceWav: String, outputPath: String): Boolean
    fun extractIclPrompt(referenceWav: String, referenceText: String, outputPath: String): Boolean
    fun getAvailableSpeakers(): List<String>
    fun close()

    class NativeParams(
        val languageId: Int = 2050,
        val instruction: String? = null,
        val speaker: String? = null
    )

    class NativeResult(
        val audio: FloatArray?,
        val sampleRate: Int,
        val success: Boolean,
        val errorMsg: String?,
        val timeMs: Long
    )
}
