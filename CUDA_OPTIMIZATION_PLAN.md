# CUDA Optimization Plan for Qwen3-TTS GGML

## Overview
This document outlines the findings and strategy for optimizing the Qwen3-TTS GGML implementation by leveraging CUDA. The goal is to significantly reduce the Real-Time Factor (RTF), which was originally at 1.94x (slower than real-time), to achieve streaming-capable performance (RTF < 1.0x).

## ✅ Completed Optimizations (Streaming-Capable Achieved)

As of February 27, 2026, the implementation has been optimized to an RTF of **~0.92x**, making it fully streaming-capable on modern GPUs (e.g., RTX 5080 Laptop).

1. **Fused Snake Activation Kernel (Vocoder Decoder):**
   - **Problem:** The Vocoder Decoder took 27.1% of pipeline time (11.6s). GGML's default element-wise operations for the Snake activation created massive memory overhead and many intermediate tensors.
   - **Solution:** Forked `ggml`, implemented a native `GGML_OP_SNAKE` with broadcasting support for `alpha` and `beta` parameters. Added multithreaded CPU fallback and a highly optimized fused CUDA kernel.
   - **Result:** Drastically reduced Vocoder decode times and VRAM bandwidth, achieving faster-than-realtime synthesis.

2. **Static KV Cache & GGML CUDA Graphs (Code Generation):**
   - **Problem:** Code Generation took 9.1% of pipeline time (3.9s). The transformer's KV cache used dynamic `ggml_view_3d` offsets based on `n_past`, forcing the CPU to re-evaluate the computation graph node-by-node on every autoregressive step, preventing CUDA graph capture.
   - **Solution:** Refactored `tts_transformer.cpp` to use a completely static graph topology. Replaced dynamic views with `ggml_set_rows` triggered by an `inp_pos` index array. Enabled `-DGGML_CUDA_GRAPHS=ON`.
   - **Result:** ~26.6% speedup in code generation. The GPU now captures the execution graph and replays it autonomously, reducing CPU overhead to near-zero during the 12Hz generation loop.

3. **Flash Attention Validation:**
   - **Status:** Verified that `ggml_flash_attn_ext` was already implemented and is actively working in the generation loop (using properly scaled F16 masks for the static KV cache).

4. **Windows MSVC Compatibility:**
   - **Status:** Fixed Linux-specific includes (`sys/resource.h`) and missing math constants (`M_PI`), allowing the codebase and CUDA kernels to compile natively on Windows via Ninja.

## 🚀 Next Steps / Future Optimizations

While standard text-to-speech is now very fast, Voice Cloning remains a significant bottleneck.

### 1. Optimize the Speaker Encoder (The "Voice Cloning" Bottleneck)
- **Problem:** When providing a reference audio file, the Speaker Encoder takes up **~64%** of the entire pipeline time (e.g., ~27 seconds on CPU). It uses an ECAPA-TDNN architecture with Res2Net blocks that rely heavily on many small, 1D convolutions that branch and merge. GGML handles these sequentially rather than in parallel.
- **Action Plan:** Write a custom, fused CUDA kernel designed to evaluate the Res2Net branches in parallel, or rewrite the `apply_conv1d` logic in `audio_tokenizer_encoder.cpp` to use cuDNN/im2col memory layouts for massive GPU speedups.

### 2. Parallelizing Codebook Prediction (12Hz Loop)
- **Problem:** The generation loop works at 12Hz. In each step, it predicts the 1st codebook token autoregressively, and then predicts codebooks 2-15 sequentially based on the 1st token.
- **Action Plan:** Since the predictions for codebooks 2-15 depend *only* on codebook 1 (and the hidden state), we can explore batching codebooks 2-15 together into a single graph execution, forcing the GPU to calculate all 14 remaining tokens simultaneously instead of evaluating the `build_code_pred_step_graph` 14 separate times per step.

### 3. Batching Multiple Generations
- **Problem:** The CLI currently processes one text prompt at a time.
- **Action Plan:** Update the Transformer and CLI to support processing a batch of prompts simultaneously (e.g., generating 4 sentences at once). This is especially powerful for Voice Cloning because the expensive Speaker Encoder only needs to run once, and the resulting speaker embedding can be broadcast across the entire batch.