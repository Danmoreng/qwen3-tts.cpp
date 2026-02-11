# qwen3-tts.cpp

C++ inference for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base) using the [GGML](https://github.com/ggml-org/ggml) tensor library.

Runs the full TTS pipeline — text tokenization, speaker encoding, transformer code generation, and vocoder decoding — in pure C++17 without Python or PyTorch at inference time.

## Features

- Full text-to-speech pipeline in C++17 with GGML backend
- Voice cloning from reference audio (ECAPA-TDNN x-vector extraction)
- Greedy and sampled decoding (temperature, top-p)
- GGUF model format (F16 and Q8_0 quantization)
- Deterministic reference tests comparing C++ output against Python
- Compile-time timing instrumentation with zero overhead in normal builds

## Prerequisites

- C++17 compiler (GCC 9+ or Clang 10+)
- CMake 3.14+
- [GGML](https://github.com/ggml-org/ggml) built from source
- Python 3.10+ with [uv](https://github.com/astral-sh/uv) (model conversion only)

## Build

```bash
# Build GGML (if not already)
git clone https://github.com/ggml-org/ggml.git /root/ggml
cd /root/ggml && mkdir build && cd build && cmake .. && make -j4

# Build qwen3-tts.cpp
git clone https://github.com/predict-woo/qwen3-tts.cpp.git
cd qwen3-tts.cpp
mkdir build && cd build
cmake ..
make -j4
```

> **Note:** The CMakeLists.txt expects GGML at `/root/ggml`. Set `GGML_DIR` to override.

## Model Conversion

Convert HuggingFace models to GGUF format:

```bash
# Download the model
huggingface-cli download Qwen/Qwen3-TTS-0.6B-Base \
    --local-dir models/Qwen3-TTS-12Hz-0.6B-Base

# Convert TTS model (transformer + speaker encoder + tokenizer)
python scripts/convert_tts_to_gguf.py \
    models/Qwen3-TTS-12Hz-0.6B-Base \
    models/qwen3-tts-0.6b-f16.gguf

# Convert vocoder (audio decoder)
python scripts/convert_tokenizer_to_gguf.py \
    models/Qwen3-TTS-12Hz-0.6B-Base \
    models/qwen3-tts-tokenizer-f16.gguf
```

Place both `.gguf` files in a `models/` directory.

## Usage

```bash
# Basic synthesis
./build/qwen3-tts-cli -m models -t "Hello, world!" -o hello.wav

# Voice cloning from reference audio
./build/qwen3-tts-cli -m models -t "Hello! How are you?" -r reference.wav -o cloned.wav

# Greedy decoding with max length
./build/qwen3-tts-cli -m models -t "Hello!" -r ref.wav -o out.wav \
    --temperature 0 --max-tokens 2048
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model <dir>` | Model directory containing GGUF files | (required) |
| `-t, --text <text>` | Text to synthesize | (required) |
| `-o, --output <file>` | Output WAV file path | `output.wav` |
| `-r, --reference <file>` | Reference audio for voice cloning | (none) |
| `--temperature <val>` | Sampling temperature (0 = greedy) | 0.7 |
| `--top-p <val>` | Top-p nucleus sampling | 0.9 |
| `--max-tokens <n>` | Maximum audio frames to generate | 4096 |
| `-j, --threads <n>` | Number of compute threads | 4 |

## Architecture

```
Text ──► [Tokenizer] ──► token IDs
                              │
Reference Audio ──► [Speaker Encoder] ──► speaker embedding
                              │
token IDs + speaker embedding ──► [TTS Transformer] ──► speech codes (N frames x 16 codebooks)
                              │
speech codes ──► [Vocoder] ──► audio waveform (24kHz)
```

### Source Files

| File | Component | Description |
|------|-----------|-------------|
| `text_tokenizer.{h,cpp}` | Tokenizer | BPE text tokenizer from GGUF |
| `audio_tokenizer_encoder.{h,cpp}` | Speaker Encoder | ECAPA-TDNN x-vector extractor |
| `tts_transformer.{h,cpp}` | TTS Transformer | 28-layer Qwen2 talker + 5-layer code predictor |
| `audio_tokenizer_decoder.{h,cpp}` | Vocoder | WavTokenizer decoder (codes to waveform) |
| `qwen3_tts.{h,cpp}` | Pipeline | Full pipeline orchestration |
| `main.cpp` | CLI | Command-line interface |
| `gguf_loader.{h,cpp}` | GGUF | Model loading utilities |

### TTS Transformer Details

The transformer generates speech codes in two stages per frame:

1. **Talker** (28 layers, 16 heads, 1024 hidden) — produces a hidden state and codebook-0 logits
2. **Code Predictor** (5 layers) — autoregressively generates codebooks 1-15 from the hidden state

The prefill embedding mirrors the Python pipeline exactly:
- Positions 0-2: text-projected role tokens (`<|im_start|>`, `assistant`, `\n`)
- Positions 3-6: TTS pad + codec embeddings (think tokens, language ID)
- Position 7: TTS pad + speaker embedding
- Position 8: TTS BOS + codec pad embedding
- Position 9+: text-projected text tokens + codec BOS/embeddings

## Testing

```bash
# Run full test suite
bash scripts/run_all_tests.sh

# Individual component tests
./build/test_tokenizer --model models/qwen3-tts-0.6b-f16.gguf
./build/test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf \
    --audio clone.wav --reference reference/ref_audio_embedding.bin
./build/test_transformer --model models/qwen3-tts-0.6b-f16.gguf \
    --ref-dir reference/
./build/test_decoder --tokenizer models/qwen3-tts-tokenizer-f16.gguf \
    --codes reference/speech_codes.bin --reference reference/decoded_audio.bin

# End-to-end Python vs C++ comparison
uv run python scripts/compare_e2e.py

# Generate deterministic reference data from Python
uv run python scripts/generate_deterministic_reference.py
```

### Test Results (F16 model)

- Prefill logits: cosine similarity = 0.99999994 with Python reference
- Codebook 0 match rate: 81% (frame-level exact match)
- Codebooks 1-4: ~84% match rate
- Audio output is perceptually equivalent; low waveform correlation is expected due to autoregressive divergence from F16 precision

## Profiling

Build with compile-time timing instrumentation (zero overhead when disabled):

```bash
cmake .. -DQWEN3_TTS_TIMING=ON
make -j4
```

Example output (92 frames, 7.3s audio):

```
=== Detailed Generation Timing (92 frames) ===

  Prefill:
      Compute:           175.9 ms

  Talker forward_step:
      Graph build:        21.8 ms   (0.2 ms/frame)
      Graph alloc:        34.1 ms   (0.4 ms/frame)
      Compute:          7717.4 ms   (83.9 ms/frame)

  Code predictor:
      Init/KV/embed:       7.7 ms   (0.1 ms/frame)
      Prefill (2tok):   1393.2 ms   (15.1 ms/frame)
      Steps (14):      19531.7 ms   (212.3 ms/frame)
      Compute:         20702.6 ms   (225.0 ms/frame)

  Total generate:      28915.0 ms   (3.2 frames/s)
```

The code predictor (15 sequential forward passes per frame) accounts for ~71% of generation time.

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-0.6B-Base) by Alibaba Qwen team
- [GGML](https://github.com/ggml-org/ggml) tensor library by Georgi Gerganov
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) vocoder architecture
