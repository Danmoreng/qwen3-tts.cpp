package com.qwen.tts.studio.engine

/**
 * JVM/Android implementation using JNI.
 */
actual class QwenEngine actual constructor() {
    private var nativePtr: Long = 0

    init {
        System.loadLibrary("qwen3_tts_jni")
        nativePtr = nativeInit()
    }

    actual fun loadModels(modelDir: String): Boolean = nativeLoadModels(nativePtr, modelDir)

    actual fun synthesize(text: String, referenceWav: String?): NativeResult = 
        nativeSynthesize(nativePtr, text, referenceWav, null)

    actual fun close() {
        if (nativePtr != 0L) {
            nativeFree(nativePtr)
            nativePtr = 0
        }
    }

    private external fun nativeInit(): Long
    private external fun nativeFree(ptr: Long)
    private external fun nativeLoadModels(ptr: Long, modelDir: String): Boolean
    private external fun nativeSynthesize(ptr: Long, text: String, referenceWav: String?, params: Any?): NativeResult

    actual class NativeResult actual constructor(
        actual val audio: FloatArray?,
        actual val sampleRate: Int,
        actual val success: Boolean,
        actual val errorMsg: String?,
        actual val timeMs: Long
    )
}
