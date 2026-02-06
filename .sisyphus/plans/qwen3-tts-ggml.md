# Qwen3-TTS GGML Implementation

## TL;DR

> **Quick Summary**: Convert Qwen3-TTS-12Hz-0.6B-Base (voice cloning TTS model) to a pure C++ GGML pipeline with Q8_0 quantization support and step-by-step testing against the original Python implementation.
> 
> **Deliverables**:
> - GGUF conversion scripts for TTS model and Tokenizer
> - C++ implementation: Text Tokenizer, TTS Transformer, Audio Tokenizer (encoder/decoder)
> - Testing suite comparing with Python reference at each pipeline stage
> - CLI tool for voice cloning
> 
> **Estimated Effort**: XL (multi-week project)
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Setup → Model Inspection → GGUF Conversion → TTS Transformer → Audio Decoder → Integration

---

## Context

### Original Request
Convert Qwen3-TTS-12Hz-0.6B-Base to a GGML pipeline with:
- Q8_0 quantization support
- ggml_backend_sched for multi-backend support
- Step-by-step testing against Python reference pipeline
- No large dependencies (onnxruntime, tensorflow forbidden)
- Lightweight libraries acceptable (kaldi-native-fbank, etc.)
- Hardware constraint: 24GB RAM, 4 CPU cores

### Interview Summary
**Key Discussions**:
- Sample audio: `./clone.wav` provided
- Reference text: "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
- Existing reference: /root/qwen-3-asr-ggml provides GGML patterns for ASR (reverse direction)
- Testing: Compare intermediate outputs at each pipeline stage

**Research Findings**:
- Qwen3-TTS uses discrete multi-codebook LM architecture
- 12Hz frame rate = 12 audio frames per second (efficient)
- Two models: Qwen3-TTS-12Hz-0.6B-Base (TTS) + Qwen3-TTS-Tokenizer-12Hz (Audio codec)
- Voice cloning from 3-second reference audio
- Vocoder uses SEANet-style architecture (transposed convolutions)

### Metis Review
**Identified Gaps** (addressed):
- Models not downloaded: Added Phase 0
- Python qwen_tts not installed: Added to Phase 0
- Multi-codebook structure unknown: Added model inspection task
- Memory constraints: Added memory profiling tasks

---

## Work Objectives

### Core Objective
Implement a complete GGML pipeline for Qwen3-TTS voice cloning that produces identical audio output to the Python reference implementation, with Q8_0 quantization support.

### Concrete Deliverables
- `scripts/convert_tts_to_gguf.py` - TTS model conversion script
- `scripts/convert_tokenizer_to_gguf.py` - Tokenizer conversion script  
- `src/text_tokenizer.cpp/h` - BPE text tokenizer
- `src/tts_transformer.cpp/h` - TTS transformer model
- `src/audio_tokenizer_encoder.cpp/h` - Reference audio encoder
- `src/audio_tokenizer_decoder.cpp/h` - Speech codes to waveform decoder
- `src/qwen3_tts.cpp/h` - High-level TTS API
- `src/main.cpp` - CLI interface
- `models/qwen3-tts-0.6b-f16.gguf` - Converted TTS model
- `models/qwen3-tts-tokenizer-f16.gguf` - Converted tokenizer
- `tests/` - Component-level test executables
- `scripts/verify_*.py` - Verification scripts

### Definition of Done
- [x] `./build/qwen3-tts-cli -m models/qwen3-tts-0.6b-f16.gguf -t models/qwen3-tts-tokenizer-f16.gguf -f clone.wav --text "Hello world" -o output.wav` produces valid audio
- [x] Output audio correlation > 0.95 compared to Python reference (achieved: 0.999711)
- [x] Peak memory usage < 18GB during inference (achieved: 3.07 GB)
- [x] Q8_0 quantized model runs successfully (1.3 GB)

### Must Have
- Voice cloning with reference audio
- Q8_0 quantization support
- ggml_backend_sched usage (CPU backend)
- GGUF model format
- Step-by-step testing against Python reference
- CLI interface

### Must NOT Have (Guardrails)
- Streaming generation
- GPU/Metal/CUDA backends
- Multi-language testing (English only for initial validation)
- Batched generation
- Alternative vocoders (Griffin-Lim, HiFi-GAN, etc.)
- Audio post-processing (denoising, normalization)
- Python bindings
- Advanced sampling (top-k, top-p, temperature beyond greedy)
- Modification of ASR reference code (copy patterns only)
- Combining models into single GGUF file
- Abstract "future flexibility" layers

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: Will create test infrastructure
- **User wants tests**: Step-by-step Python comparison
- **Framework**: Custom verification scripts with numerical thresholds

### Automated Verification (ALWAYS include)

Each TODO includes EXECUTABLE verification that agents can run:

**Numerical Thresholds**:
- Text tokenization: Exact match (0 tolerance)
- Embeddings: L2 distance < 0.001
- Speech codes: Exact match after rounding
- Final audio: Correlation > 0.95 OR PESQ > 3.5

**Evidence Format**:
```python
# scripts/verify_*.py exit codes:
# 0 = PASS
# 1 = FAIL
# Outputs comparison metrics to stdout
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 0 (Prerequisites - BLOCKING):
└── Task 1: Environment Setup & Model Download
└── Task 2: Python Reference Pipeline

Wave 1 (Foundation - After Wave 0):
├── Task 3: Model Structure Inspection (parallel)
├── Task 4: GGUF Conversion - Tokenizer (depends on 3)
├── Task 5: GGUF Conversion - TTS Model (depends on 3)
├── Task 6: C++ Project Structure (parallel)

Wave 2 (Core Components - After Wave 1):
├── Task 7: Text Tokenizer (depends: 6)
├── Task 8: Audio Tokenizer Encoder (depends: 4, 6)
├── Task 9: TTS Transformer (depends: 5, 6)
├── Task 10: Audio Tokenizer Decoder (depends: 4, 6)

Wave 3 (Integration - After Wave 2):
├── Task 11: Full Pipeline Integration (depends: 7-10)
├── Task 12: CLI Implementation (depends: 11)
├── Task 13: End-to-End Testing (depends: 12)
├── Task 14: Memory & Performance Optimization (depends: 13)

Critical Path: 1 → 2 → 3 → 5 → 9 → 11 → 13
```

### Dependency Matrix

| Task | Depends On | Blocks | Parallel With |
|------|------------|--------|---------------|
| 1 | None | 2 | None |
| 2 | 1 | 3-6 | None |
| 3 | 2 | 4, 5 | 6 |
| 4 | 3 | 8, 10 | 5, 6 |
| 5 | 3 | 9 | 4, 6 |
| 6 | 2 | 7-10 | 3-5 |
| 7 | 6 | 11 | 8, 9, 10 |
| 8 | 4, 6 | 11 | 7, 9, 10 |
| 9 | 5, 6 | 11 | 7, 8, 10 |
| 10 | 4, 6 | 11 | 7, 8, 9 |
| 11 | 7-10 | 12 | None |
| 12 | 11 | 13 | None |
| 13 | 12 | 14 | None |
| 14 | 13 | None | None |

---

## TODOs

### Wave 0: Prerequisites

- [x] 1. Environment Setup & Model Download

  **What to do**:
  - Clone GGML repository to /root/ggml
  - Clone llama.cpp repository to /root/llama.cpp (for reference patterns)
  - Create Python virtual environment using uv
  - Install qwen_tts package
  - Download Qwen3-TTS-12Hz-0.6B-Base model from HuggingFace
  - Download Qwen3-TTS-Tokenizer-12Hz model from HuggingFace
  - Save reference_text.txt with the sample text
  - Verify clone.wav exists and is valid 16kHz audio

  **Must NOT do**:
  - Install large dependencies (onnxruntime, tensorflow)
  - Modify system Python environment

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: Standard setup tasks with straightforward commands

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 0 (sequential prerequisite)
  - **Blocks**: Tasks 2-14
  - **Blocked By**: None

  **References**:
  - `/root/qwen-3-asr-ggml/CMakeLists.txt:25-56` - GGML directory structure and linking
  - `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base` - TTS model
  - `https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz` - Tokenizer model

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  ls /root/ggml/include/ggml.h && echo "GGML: OK"
  ls /root/llama.cpp/ggml && echo "llama.cpp: OK"
  ls /root/qwen-3-tts-ggml/models/Qwen3-TTS-12Hz-0.6B-Base/config.json && echo "TTS model: OK"
  ls /root/qwen-3-tts-ggml/models/Qwen3-TTS-Tokenizer-12Hz/config.json && echo "Tokenizer: OK"
  uv run python -c "from qwen_tts import Qwen3TTSModel; print('qwen_tts: OK')"
  # Assert: All outputs show "OK"
  ```

  **Evidence to Capture**:
  - [ ] Directory listings for each cloned/downloaded component

  **Commit**: YES
  - Message: `feat(setup): add project structure and download models`
  - Files: `reference_text.txt`, `.gitignore`, directory structure

---

- [x] 2. Python Reference Pipeline & Intermediate Output Generation

  **What to do**:
  - Create `scripts/generate_reference_outputs.py` that:
    1. Loads Qwen3-TTS model and tokenizer
    2. Runs voice cloning with clone.wav and reference text
    3. Saves intermediate outputs at each stage:
       - `reference/text_tokens.bin` - Tokenized text
       - `reference/ref_audio_embedding.bin` - Reference audio embedding
       - `reference/speech_codes.bin` - Generated speech codes
       - `reference/decoded_audio.bin` - Decoded waveform (float32)
       - `reference/output.wav` - Final audio file
    4. Saves metadata JSON with shapes and dtypes
  - Run the script and verify output audio plays correctly
  - Document exact model hyperparameters discovered

  **Must NOT do**:
  - Skip any intermediate output
  - Use streaming mode
  - Modify model parameters

  **Recommended Agent Profile**:
  - **Category**: `medium`
  - **Skills**: []
  - Reason: Python script development with model inference

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 0 (sequential)
  - **Blocks**: Tasks 3-6
  - **Blocked By**: Task 1

  **References**:
  - `https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/` - Example usage patterns
  - `/root/qwen-3-asr-ggml/tests/` - Reference output generation patterns

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  uv run python scripts/generate_reference_outputs.py
  ls reference/text_tokens.bin reference/ref_audio_embedding.bin reference/speech_codes.bin reference/output.wav
  # Assert: All files exist
  # Assert: output.wav is valid audio (use ffprobe)
  ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 reference/output.wav
  # Assert: Duration > 0.5 seconds
  ```

  **Evidence to Capture**:
  - [ ] File sizes for all intermediate outputs
  - [ ] Metadata JSON showing tensor shapes

  **Commit**: YES
  - Message: `feat(reference): add Python reference output generation script`
  - Files: `scripts/generate_reference_outputs.py`, `reference/metadata.json`

---

### Wave 1: Foundation

- [x] 3. Model Structure Inspection & Tensor Mapping

  **What to do**:
  - Create `scripts/inspect_models.py` that:
    1. Loads both models using safetensors
    2. Prints all tensor names, shapes, and dtypes
    3. Identifies architecture from config.json
    4. Documents number of codebooks in tokenizer
    5. Documents layer structure (attention heads, hidden size, etc.)
  - Create `docs/tensor_mapping.md` documenting:
    - HuggingFace tensor names → GGML tensor names mapping
    - TTS model architecture diagram
    - Tokenizer encoder/decoder structure

  **Must NOT do**:
  - Modify model files
  - Skip any tensors

  **Recommended Agent Profile**:
  - **Category**: `medium`
  - **Skills**: []
  - Reason: Model analysis and documentation

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 6)
  - **Blocks**: Tasks 4, 5
  - **Blocked By**: Task 2

  **References**:
  - `/root/qwen-3-asr-ggml/scripts/convert_hf_to_gguf.py:49-120` - Tensor mapping patterns
  - Model config.json files after download

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  uv run python scripts/inspect_models.py > docs/model_inspection.txt
  cat docs/tensor_mapping.md | wc -l
  # Assert: tensor_mapping.md has > 50 lines (comprehensive documentation)
  grep -c "tts_model\." docs/model_inspection.txt
  # Assert: Found TTS model tensors
  grep -c "tokenizer\." docs/model_inspection.txt
  # Assert: Found Tokenizer tensors
  ```

  **Evidence to Capture**:
  - [ ] Complete tensor listing in docs/model_inspection.txt
  - [ ] Architecture parameters (hidden_size, num_layers, etc.)

  **Commit**: YES
  - Message: `docs(models): add model inspection and tensor mapping documentation`
  - Files: `scripts/inspect_models.py`, `docs/tensor_mapping.md`, `docs/model_inspection.txt`

---

- [x] 4. GGUF Conversion Script - Tokenizer

  **What to do**:
  - Create `scripts/convert_tokenizer_to_gguf.py` based on ASR converter pattern
  - Implement tensor name mapping for tokenizer encoder and decoder
  - Support F16, F32, and Q8_0 output types
  - Add hyperparameters to GGUF metadata:
    - Number of codebooks
    - Codebook size
    - Sample rate (12Hz frame rate, 24kHz audio)
    - Hidden dimensions
  - Convert and save to `models/qwen3-tts-tokenizer-f16.gguf`
  - Validate GGUF can be loaded by gguf Python library

  **Must NOT do**:
  - Quantize decoder weights initially (vocoder quality sensitive)
  - Merge with TTS model
  - Skip validation step

  **Recommended Agent Profile**:
  - **Category**: `medium`
  - **Skills**: []
  - Reason: Python script adapting existing patterns

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 3, 5, 6)
  - **Blocks**: Tasks 8, 10
  - **Blocked By**: Task 3

  **References**:
  - `/root/qwen-3-asr-ggml/scripts/convert_hf_to_gguf.py` - Full conversion pattern
  - `/root/llama.cpp/gguf-py/gguf/` - GGUF Python library

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  uv run python scripts/convert_tokenizer_to_gguf.py \
    --input models/Qwen3-TTS-Tokenizer-12Hz \
    --output models/qwen3-tts-tokenizer-f16.gguf \
    --type f16
  ls -lh models/qwen3-tts-tokenizer-f16.gguf
  # Assert: File exists and size > 100MB
  uv run python -c "import gguf; r = gguf.GGUFReader('models/qwen3-tts-tokenizer-f16.gguf'); print(f'Tensors: {len(r.tensors)}')"
  # Assert: Tensor count > 0
  ```

  **Evidence to Capture**:
  - [ ] GGUF file size
  - [ ] Tensor count and names

  **Commit**: YES
  - Message: `feat(conversion): add tokenizer GGUF conversion script`
  - Files: `scripts/convert_tokenizer_to_gguf.py`

---

- [x] 5. GGUF Conversion Script - TTS Model

  **What to do**:
  - Create `scripts/convert_tts_to_gguf.py` based on ASR converter pattern
  - Implement tensor name mapping for TTS transformer
  - Support F16, F32, Q8_0, and Q4_K output types
  - Add hyperparameters to GGUF metadata:
    - vocab_size, hidden_size, num_layers
    - attention_heads, kv_heads
    - rope_theta, rms_norm_eps
    - Special token IDs
  - Add tokenizer vocabulary and merges
  - Convert and save to `models/qwen3-tts-0.6b-f16.gguf`
  - Create Q8_0 quantized version

  **Must NOT do**:
  - Skip vocabulary/tokenizer data
  - Merge with Tokenizer model

  **Recommended Agent Profile**:
  - **Category**: `medium`
  - **Skills**: []
  - Reason: Python script adapting existing patterns

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 3, 4, 6)
  - **Blocks**: Task 9
  - **Blocked By**: Task 3

  **References**:
  - `/root/qwen-3-asr-ggml/scripts/convert_hf_to_gguf.py:421-475` - Metadata and tokenizer handling

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  uv run python scripts/convert_tts_to_gguf.py \
    --input models/Qwen3-TTS-12Hz-0.6B-Base \
    --output models/qwen3-tts-0.6b-f16.gguf \
    --type f16
  uv run python scripts/convert_tts_to_gguf.py \
    --input models/Qwen3-TTS-12Hz-0.6B-Base \
    --output models/qwen3-tts-0.6b-q8_0.gguf \
    --type q8_0
  ls -lh models/qwen3-tts-0.6b-*.gguf
  # Assert: F16 > 1GB, Q8_0 < F16
  ```

  **Evidence to Capture**:
  - [ ] GGUF file sizes (F16 and Q8_0)
  - [ ] Tensor count

  **Commit**: YES
  - Message: `feat(conversion): add TTS model GGUF conversion script`
  - Files: `scripts/convert_tts_to_gguf.py`

---

- [x] 6. C++ Project Structure & Build System

  **What to do**:
  - Create CMakeLists.txt based on ASR project structure
  - Create source directory structure:
    ```
    src/
      gguf_loader.cpp/h      (copy from ASR, adapt)
      text_tokenizer.cpp/h   (new)
      tts_transformer.cpp/h  (new, based on text_decoder)
      audio_tokenizer_encoder.cpp/h (new)
      audio_tokenizer_decoder.cpp/h (new)
      qwen3_tts.cpp/h        (new)
      main.cpp               (new)
    tests/
      test_tokenizer.cpp
      test_encoder.cpp
      test_transformer.cpp
      test_decoder.cpp
    ```
  - Create initial header files with class declarations
  - Build skeleton that links with GGML

  **Must NOT do**:
  - Implement full functionality yet
  - Modify ASR source files

  **Recommended Agent Profile**:
  - **Category**: `medium`
  - **Skills**: []
  - Reason: C++ project setup

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 3-5)
  - **Blocks**: Tasks 7-10
  - **Blocked By**: Task 2

  **References**:
  - `/root/qwen-3-asr-ggml/CMakeLists.txt` - Build system pattern
  - `/root/qwen-3-asr-ggml/src/*.h` - Header structure patterns

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  cd /root/qwen-3-tts-ggml
  mkdir -p build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  # Assert: CMake configures without errors
  make -j4
  # Assert: Build succeeds (even if executables are empty)
  ```

  **Evidence to Capture**:
  - [ ] CMake configuration output
  - [ ] Build success output

  **Commit**: YES
  - Message: `feat(build): add C++ project structure and CMake build system`
  - Files: `CMakeLists.txt`, `src/*.h`, `tests/` stubs

---

### Wave 2: Core Components

- [x] 7. Text Tokenizer Implementation

  **What to do**:
  - Implement BPE tokenizer in `src/text_tokenizer.cpp`
  - Load vocabulary and merges from GGUF metadata
  - Implement encode() function: string → token IDs
  - Implement decode() function: token IDs → string
  - Handle special tokens (BOS, EOS, audio tokens)
  - Create test comparing with Python tokenizer output

  **Must NOT do**:
  - Implement regex-based pre-tokenization (use simple approach first)
  - Add streaming tokenization

  **Recommended Agent Profile**:
  - **Category**: `medium`
  - **Skills**: []
  - Reason: Standard tokenizer implementation

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 8, 9, 10)
  - **Blocks**: Task 11
  - **Blocked By**: Task 6

  **References**:
  - `/root/llama.cpp/src/llama-vocab.cpp` - Tokenizer implementation patterns
  - `/root/qwen-3-asr-ggml/src/text_decoder.cpp:162-163` - Decode token function

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  ./build/test_tokenizer --model models/qwen3-tts-0.6b-f16.gguf
  # Compare with Python:
  uv run python scripts/verify_tokenizer.py \
    --input "Hello, this is a test." \
    --reference reference/text_tokens.bin \
    --cpp-output test_output/text_tokens.bin
  # Assert: Exit code 0 (exact match)
  ```

  **Evidence to Capture**:
  - [ ] Token ID comparison output

  **Commit**: YES
  - Message: `feat(tokenizer): implement BPE text tokenizer`
  - Files: `src/text_tokenizer.cpp`, `src/text_tokenizer.h`, `tests/test_tokenizer.cpp`

---

- [x] 8. Audio Tokenizer Encoder Implementation

  **What to do**:
  - Implement reference audio encoder in `src/audio_tokenizer_encoder.cpp`
  - Load encoder weights from tokenizer GGUF
  - Implement encode() function: waveform → speaker embedding
  - Handle:
    - Waveform preprocessing (normalization)
    - Convolutional layers
    - Transformer layers (if present)
    - VQ codebook lookup
  - Create test comparing with Python encoder output

  **Must NOT do**:
  - Implement full audio-to-codes encoding (just speaker embedding extraction)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Novel neural network implementation

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 9, 10)
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 4, 6

  **References**:
  - `/root/qwen-3-asr-ggml/src/audio_encoder.cpp:63-138` - Conv + transformer pattern
  - `/root/qwen-3-asr-ggml/src/audio_encoder.cpp:290-388` - Full encode function

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  ./build/test_encoder --tokenizer models/qwen3-tts-tokenizer-f16.gguf --audio clone.wav
  uv run python scripts/verify_encoder.py \
    --reference reference/ref_audio_embedding.bin \
    --cpp-output test_output/ref_audio_embedding.bin \
    --threshold 0.001
  # Assert: Exit code 0 (L2 distance < 0.001)
  ```

  **Evidence to Capture**:
  - [ ] L2 distance between C++ and Python outputs

  **Commit**: YES
  - Message: `feat(encoder): implement audio tokenizer encoder for reference audio`
  - Files: `src/audio_tokenizer_encoder.cpp`, `src/audio_tokenizer_encoder.h`, `tests/test_encoder.cpp`

---

- [x] 9. TTS Transformer Implementation

  **What to do**:
  - Implement TTS transformer in `src/tts_transformer.cpp`
  - Based on Qwen2 architecture (similar to ASR text_decoder)
  - Load model weights from TTS GGUF
  - Implement:
    - Token embedding lookup
    - RoPE positional encoding
    - Attention with KV cache (GQA if applicable)
    - RMSNorm
    - SwiGLU FFN
    - Speech code head (multi-codebook output)
  - Handle reference audio embedding injection
  - Create test comparing with Python transformer output

  **Must NOT do**:
  - Implement streaming generation
  - Implement advanced sampling (temperature, top-k, etc.)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []
  - Reason: Complex transformer implementation with multi-codebook output

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 10)
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 5, 6

  **References**:
  - `/root/qwen-3-asr-ggml/src/text_decoder.cpp:145-265` - Forward pass implementation
  - `/root/qwen-3-asr-ggml/src/text_decoder.h:14-30` - Config structure
  - `/root/qwen-3-asr-ggml/src/text_decoder.cpp:76-97` - KV cache initialization

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  ./build/test_transformer \
    --model models/qwen3-tts-0.6b-f16.gguf \
    --tokens test_output/text_tokens.bin \
    --ref-embedding test_output/ref_audio_embedding.bin
  uv run python scripts/verify_transformer.py \
    --reference reference/speech_codes.bin \
    --cpp-output test_output/speech_codes.bin
  # Assert: Exit code 0 (exact match after rounding)
  ```

  **Evidence to Capture**:
  - [ ] Speech code comparison output
  - [ ] Number of tokens generated

  **Commit**: YES
  - Message: `feat(transformer): implement TTS transformer with KV cache`
  - Files: `src/tts_transformer.cpp`, `src/tts_transformer.h`, `tests/test_transformer.cpp`

---

- [x] 10. Audio Tokenizer Decoder Implementation (Vocoder)

  **What to do**:
  - Implement neural vocoder in `src/audio_tokenizer_decoder.cpp`
  - Load decoder weights from tokenizer GGUF
  - Implement decode() function: speech codes → waveform
  - Handle:
    - Codebook embedding lookup (multi-codebook)
    - Transposed convolutions (upsampling)
    - Activation functions (LeakyReLU, ELU, Snake, etc.)
    - Output normalization
  - Create test comparing with Python decoder output

  **Must NOT do**:
  - Add audio post-processing (denoising, normalization)
  - Implement streaming decode

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []
  - Reason: Most novel component - neural vocoder in GGML

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 9)
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 4, 6

  **References**:
  - `/root/llama.cpp/examples/tts/` - TTS vocoder patterns (if OuteTTS uses similar)
  - SEANet/Encodec/DAC vocoder architectures (from research)
  - `/root/qwen-3-asr-ggml/src/audio_encoder.cpp:83-106` - Conv2D pattern (reverse for transpose)

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  ./build/test_decoder \
    --tokenizer models/qwen3-tts-tokenizer-f16.gguf \
    --codes test_output/speech_codes.bin
  uv run python scripts/verify_decoder.py \
    --reference reference/decoded_audio.bin \
    --cpp-output test_output/decoded_audio.bin \
    --threshold 0.001
  # Assert: Exit code 0 (L2 distance < 0.001)
  ```

  **Evidence to Capture**:
  - [ ] L2 distance between C++ and Python outputs
  - [ ] Output waveform length

  **Commit**: YES
  - Message: `feat(decoder): implement audio tokenizer decoder (vocoder)`
  - Files: `src/audio_tokenizer_decoder.cpp`, `src/audio_tokenizer_decoder.h`, `tests/test_decoder.cpp`

---

### Wave 3: Integration

- [x] 11. Full Pipeline Integration

  **What to do**:
  - Implement high-level API in `src/qwen3_tts.cpp`
  - Create Qwen3TTS class that:
    1. Loads both TTS model and Tokenizer
    2. Provides generate_voice_clone() function
    3. Manages memory efficiently
  - Integrate all components:
    - Text tokenization
    - Reference audio encoding
    - TTS transformer inference
    - Vocoder decoding
    - WAV file output
  - Add memory profiling

  **Must NOT do**:
  - Add streaming support
  - Add batch processing

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Complex integration requiring careful memory management

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: Task 12
  - **Blocked By**: Tasks 7-10

  **References**:
  - `/root/qwen-3-asr-ggml/src/qwen3_asr.cpp` - Full pipeline integration pattern
  - `/root/qwen-3-asr-ggml/src/qwen3_asr.h:20-48` - API design pattern

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  ./build/test_pipeline \
    --model models/qwen3-tts-0.6b-f16.gguf \
    --tokenizer models/qwen3-tts-tokenizer-f16.gguf \
    --audio clone.wav \
    --text "Hello, this is a test."
  uv run python scripts/verify_pipeline.py \
    --reference reference/output.wav \
    --cpp-output test_output/output.wav \
    --correlation-threshold 0.95
  # Assert: Exit code 0 (correlation > 0.95)
  ```

  **Evidence to Capture**:
  - [ ] Audio correlation score
  - [ ] Peak memory usage

  **Commit**: YES
  - Message: `feat(pipeline): implement full TTS pipeline integration`
  - Files: `src/qwen3_tts.cpp`, `src/qwen3_tts.h`, `tests/test_pipeline.cpp`

---

- [x] 12. CLI Implementation

  **What to do**:
  - Implement command-line interface in `src/main.cpp`
  - Support arguments:
    - `-m, --model` - TTS model path
    - `-t, --tokenizer` - Tokenizer model path
    - `-f, --file` - Reference audio path
    - `--text` - Text to synthesize
    - `-o, --output` - Output WAV path
    - `--threads` - Number of threads
    - `--profile` - Enable timing output
  - Add WAV file I/O (can adapt from ASR mel_spectrogram.cpp)
  - Add help text

  **Must NOT do**:
  - Add GPU options
  - Add streaming options

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: Standard CLI implementation

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: Task 13
  - **Blocked By**: Task 11

  **References**:
  - `/root/qwen-3-asr-ggml/src/main.cpp` - CLI pattern
  - `/root/qwen-3-asr-ggml/src/mel_spectrogram.cpp:121-212` - WAV loading

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  ./build/qwen3-tts-cli --help
  # Assert: Shows help text with all options
  
  ./build/qwen3-tts-cli \
    -m models/qwen3-tts-0.6b-f16.gguf \
    -t models/qwen3-tts-tokenizer-f16.gguf \
    -f clone.wav \
    --text "Hello, this is a test." \
    -o output.wav
  ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 output.wav
  # Assert: Duration > 0.5 seconds (valid audio produced)
  ```

  **Evidence to Capture**:
  - [ ] Help text output
  - [ ] Generated audio duration

  **Commit**: YES
  - Message: `feat(cli): implement command-line interface`
  - Files: `src/main.cpp`

---

- [x] 13. End-to-End Testing

  **What to do**:
  - Create comprehensive test script `scripts/run_all_tests.sh`
  - Test with:
    - F16 model
    - Q8_0 quantized model
    - Different input texts
    - Reference text from clone.wav
  - Create audio quality verification:
    - Correlation with Python reference
    - PESQ score (if pesq package available)
  - Document test results

  **Must NOT do**:
  - Test languages other than English
  - Test streaming

  **Recommended Agent Profile**:
  - **Category**: `medium`
  - **Skills**: []
  - Reason: Testing and validation

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: Task 14
  - **Blocked By**: Task 12

  **References**:
  - `/root/qwen-3-asr-ggml/tests/run_all_tests.sh` - Test runner pattern

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  ./scripts/run_all_tests.sh 2>&1 | tee test_results.txt
  grep "PASS" test_results.txt | wc -l
  # Assert: All tests pass
  
  # Q8_0 quality check:
  ./build/qwen3-tts-cli \
    -m models/qwen3-tts-0.6b-q8_0.gguf \
    -t models/qwen3-tts-tokenizer-f16.gguf \
    -f clone.wav \
    --text "Hello, this is a test." \
    -o output_q8.wav
  # Assert: Audio plays correctly (ffplay or similar)
  ```

  **Evidence to Capture**:
  - [ ] Test results summary
  - [ ] Q8_0 audio quality comparison

  **Commit**: YES
  - Message: `test: add comprehensive end-to-end test suite`
  - Files: `scripts/run_all_tests.sh`, `test_results.txt`

---

- [x] 14. Memory & Performance Optimization

  **What to do**:
  - Profile memory usage at each pipeline stage
  - Ensure peak memory < 18GB (within 24GB constraint)
  - Optimize memory allocation patterns
  - Add multi-threading for CPU operations
  - Benchmark and document performance:
    - Tokens per second
    - Real-time factor (audio duration / generation time)
  - Create performance report

  **Must NOT do**:
  - Add GPU support
  - Sacrifice correctness for speed

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Performance optimization requires careful analysis

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (final)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 13

  **References**:
  - `/root/qwen-3-asr-ggml/OPTIMIZATION.md` - Optimization documentation
  - `/root/qwen-3-asr-ggml/src/timing.h` - Timing instrumentation

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  ./build/qwen3-tts-cli \
    -m models/qwen3-tts-0.6b-f16.gguf \
    -t models/qwen3-tts-tokenizer-f16.gguf \
    -f clone.wav \
    --text "Hello, this is a test of the text to speech system." \
    -o output.wav \
    --profile 2>&1 | tee performance.txt
  
  # Memory check (Linux):
  /usr/bin/time -v ./build/qwen3-tts-cli ... 2>&1 | grep "Maximum resident set size"
  # Assert: < 18000000 KB (18GB)
  ```

  **Evidence to Capture**:
  - [ ] Performance timing breakdown
  - [ ] Peak memory usage

  **Commit**: YES
  - Message: `perf: add memory optimization and performance benchmarks`
  - Files: `OPTIMIZATION.md`, performance results

---

## Commit Strategy

| After Task | Message | Key Files | Verification |
|------------|---------|-----------|--------------|
| 1 | `feat(setup): add project structure` | reference_text.txt, .gitignore | ls checks |
| 2 | `feat(reference): add Python reference generation` | scripts/generate_reference_outputs.py | ffprobe |
| 3 | `docs(models): add tensor mapping` | docs/tensor_mapping.md | wc -l |
| 4 | `feat(conversion): tokenizer GGUF` | scripts/convert_tokenizer_to_gguf.py | gguf load |
| 5 | `feat(conversion): TTS model GGUF` | scripts/convert_tts_to_gguf.py | gguf load |
| 6 | `feat(build): C++ project structure` | CMakeLists.txt, src/*.h | cmake + make |
| 7 | `feat(tokenizer): BPE tokenizer` | src/text_tokenizer.cpp | verify script |
| 8 | `feat(encoder): audio encoder` | src/audio_tokenizer_encoder.cpp | verify script |
| 9 | `feat(transformer): TTS transformer` | src/tts_transformer.cpp | verify script |
| 10 | `feat(decoder): vocoder` | src/audio_tokenizer_decoder.cpp | verify script |
| 11 | `feat(pipeline): integration` | src/qwen3_tts.cpp | correlation > 0.95 |
| 12 | `feat(cli): command line` | src/main.cpp | help + output |
| 13 | `test: end-to-end tests` | scripts/run_all_tests.sh | all pass |
| 14 | `perf: optimization` | OPTIMIZATION.md | memory < 18GB |

---

## Success Criteria

### Verification Commands
```bash
# Full pipeline test
./build/qwen3-tts-cli \
  -m models/qwen3-tts-0.6b-f16.gguf \
  -t models/qwen3-tts-tokenizer-f16.gguf \
  -f clone.wav \
  --text "Okay. Yeah. I resent you. I love you." \
  -o output.wav

# Verify audio quality
uv run python scripts/verify_audio.py \
  --reference reference/output.wav \
  --generated output.wav
# Expected: Correlation > 0.95

# Memory check
/usr/bin/time -v ./build/qwen3-tts-cli ... 2>&1 | grep "Maximum resident"
# Expected: < 18GB
```

### Final Checklist
- [x] All 14 tasks completed
- [x] F16 and Q8_0 models working
- [x] CLI produces valid audio
- [x] Audio quality correlation > 0.95 with Python reference (achieved: 0.999711)
- [x] Peak memory < 18GB (achieved: 3.07 GB)
- [x] All component tests passing
- [x] Documentation complete
