# CUDA Optimization Plan for Qwen3-TTS GGML

## Overview
This document outlines the findings and strategy for optimizing the Qwen3-TTS GGML implementation by leveraging CUDA. The goal is to achieve performance parity with optimized Python implementations like `Faster-Qwen3-TTS` (target RTF > 3.5x).

## ✅ Completed Optimizations (Streaming-Capable Achieved)

As of February 28, 2026, the implementation has been optimized to an RTF of **~1.07x** (Internal Throughput) on modern GPUs (e.g., RTX 5080 Laptop).

1. **Fused Snake Activation Kernel (Vocoder Decoder):**
   - **Problem:** The Vocoder Decoder took 27.1% of pipeline time. GGML's default element-wise operations for the Snake activation created massive memory overhead.
   - **Solution:** Implemented a native `GGML_OP_SNAKE` with a highly optimized fused CUDA kernel.
   - **Result:** Drastically reduced Vocoder decode times and VRAM bandwidth.

2. **Static KV Cache & GGML CUDA Graphs (Code Generation):**
   - **Problem:** Dynamic `ggml_view_3d` offsets forced node-by-node CPU evaluation on every autoregressive step.
   - **Solution:** Refactored `tts_transformer.cpp` to use a static graph topology with `ggml_set_rows`. Enabled `GGML_CUDA_GRAPHS`.
   - **Result:** ~26.6% speedup in code generation and near-zero CPU overhead during inference.

## 🚀 Road to Parity (Target: 3.5x RTF)

To match `Faster-Qwen3-TTS`, we must address the remaining bottlenecks in the Speaker Encoder, Vocoder, and Predictor orchestration.

### Phase 1: Optimize the Speaker Encoder (Voice Cloning Latency)
*   **Target:** Reduce cloning latency from ~9s to < 1s.
*   **Problem:** The ECAPA-TDNN architecture in `audio_tokenizer_encoder.cpp` relies on hundreds of sequential 1D convolutions.
*   **Strategy:**
    *   **Fused Res2Net Kernels:** Implement custom CUDA kernels for the Res2Net blocks to parallelize multi-scale branches.
    *   **cuDNN Integration:** Map `ggml_conv_1d` to cuDNN/cutlass for the encoder layers to leverage specialized hardware acceleration.

### Phase 2: Parallelize/Batch Codebook Prediction
*   **Target:** Increase generation throughput from 1.0x to 2.0x RTF.
*   **Problem:** The 5-layer Predictor runs 14 times sequentially *per frame*, resulting in 14 separate GGML graph executions per 12Hz step.
*   **Strategy (Graph Unrolling):**
    *   **Single-Graph Prediction:** "Unroll" all 14 predictor steps into a single, static GGML graph. This reduces 14 `ggml_graph_compute` calls to one, matching the orchestration strategy of `Faster-Qwen3-TTS`.

### Phase 3: High-Performance Vocoder (Throughput)
*   **Target:** Increase generation throughput from 2.0x to 3.5x+ RTF.
*   **Problem:** The `WavTokenizer` decoder uses many small upsampling blocks that create high VRAM bandwidth pressure.
*   **Strategy:**
    *   **Convolution Fusing:** Fuse `Conv1D + Snake + Upsample` layers into single CUDA kernels to minimize intermediate memory transfers.
    *   **Weight Packing:** Optimize vocoder weight layout for Blackwell/Ada Lovelace memory alignment.

## ⚒️ Development Timeline

| Milestone | Task | Priority |
| :--- | :--- | :--- |
| **M1** | ECAPA-TDNN Fused Kernels | High (User Latency) |
| **M2** | Unrolled Predictor Graph | Medium (Throughput) |
| **M3** | Fused Vocoder Blocks | Medium (Throughput) |
