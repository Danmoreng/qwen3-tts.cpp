# Architecture Refactor Plan

Last updated: 2026-03-09

## Purpose

This document captures the current code-structure findings for the native `qwen3-tts.cpp` codebase and proposes a concrete refactor path to make the project easier to maintain without changing model behavior.

The immediate goal is not to redesign the runtime architecture. The goal is to:

- Split oversized source files into clearer modules
- Reduce implementation detail leakage in public headers
- Make ownership boundaries easier to understand
- Preserve current behavior, tests, and performance characteristics while refactoring

## Current High-Level Architecture

The runtime is already organized around sensible pipeline stages:

1. `TextTokenizer`
2. `AudioTokenizerEncoder` (speaker encoder)
3. `TTSTransformer` (talker + code predictor)
4. `AudioTokenizerDecoder` (vocoder)
5. `Qwen3TTS` orchestration facade

At a coarse level, this is a good architecture. The main maintainability problems are not at the pipeline level; they are inside the implementation units.

## Current File Hotspots

Measured source file sizes in `src/`:

| File | Lines | Notes |
|---|---:|---|
| `src/tts_transformer.cpp` | 3481 | Largest hotspot; mixes many unrelated concerns |
| `src/qwen3_tts.cpp` | 954 | Pipeline orchestration mixed with I/O and utilities |
| `src/audio_tokenizer_decoder.cpp` | 911 | Model loading, graph construction, layer helpers, runtime |
| `src/audio_tokenizer_encoder.cpp` | 712 | DSP frontend plus GGML runtime in one file |
| `src/tts_transformer.h` | 355 | Public header carries large amounts of private implementation detail |
| `src/audio_tokenizer_decoder.h` | 232 | Public header exposes internal model/state/layout structs |
| `src/qwen3_tts.h` | 173 | Public facade is reasonable, but utility functions and private details can still be separated better |

## Findings

### 1. `TTSTransformer` has too many responsibilities

`src/tts_transformer.cpp` currently owns all of the following:

- Debug trace configuration and binary dump helpers
- GGUF config parsing
- Tensor creation and tensor loading
- Backend and scheduler lifecycle
- Talker KV-cache management
- Code predictor KV-cache management
- Embedding lookup helpers
- Text projection
- Prefill embedding construction
- Talker graph construction
- Code predictor graph construction
- CoreML predictor setup and fallback path
- Forward execution helpers
- Sampling helpers
- Outer autoregressive generation loop

This is the single biggest refactor target.

### 2. The public transformer header is carrying implementation-only data

`src/tts_transformer.h` exposes:

- `tts_transformer_model`
- `tts_transformer_state`
- `tts_kv_cache`
- `transformer_layer`
- most private helper declarations

That makes the public header much heavier than it needs to be and couples downstream compilation to internal graph/model layout details.

### 3. `Qwen3TTS` mixes orchestration with unrelated utility code

`src/qwen3_tts.cpp` currently contains:

- GGML log filtering
- process memory snapshot helpers
- model path discovery and lazy-load policy
- synthesis orchestration
- WAV loading
- WAV saving
- resampling
- speaker embedding text/binary parsing
- speaker embedding serialization

These are separable concerns that do not need to live in a single translation unit.

### 4. Encoder frontend DSP and runtime graph code are coupled

`src/audio_tokenizer_encoder.cpp` currently mixes:

- mel filterbank generation
- DFT/window helpers
- mel spectrogram frontend
- GGUF loading
- graph construction
- graph execution

The DSP frontend should be split from model/runtime code.

### 5. Decoder layer definitions, graph assembly, and runtime are coupled

`src/audio_tokenizer_decoder.cpp` currently mixes:

- model loading
- codebook normalization
- graph cache management
- layer helper implementations (`Snake`, RMSNorm, residual/upsample/decoder blocks)
- full graph assembly
- runtime execution

The decoder has natural seams and should be split accordingly.

### 6. The current CMake target split is good enough

The existing top-level target structure is reasonable:

- `text_tokenizer`
- `tts_transformer`
- `audio_tokenizer_encoder`
- `audio_tokenizer_decoder`
- `qwen3_tts`

The refactor should not begin by changing target boundaries. The first step should be splitting source files inside the existing targets. That keeps build/test risk lower.

### 7. Oversplitting would also be a mistake

The right goal is not "tiny files". The right goal is coherent files with one main concern.

Practical target:

- Small helpers: under ~200 lines
- Normal implementation units: ~200-500 lines
- Complex graph/runtime units: ~400-700 lines

Files larger than ~800 lines should be exceptional and justified.

## Refactor Strategy

### Principle 1: Refactor by subsystem, not by class name alone

Each subsystem should be split by responsibility:

- model loading
- graph construction
- runtime execution
- utility/helpers
- optional debug/trace tooling

### Principle 2: Keep public APIs stable during Phase 1

Do not change the external `Qwen3TTS`, `TTSTransformer`, encoder, or decoder API surface in the first pass unless necessary for cleanup.

Phase 1 should be mostly movement and delegation.

### Principle 3: Move implementation detail out of public headers

Once the file split is stable, reduce header surface:

- move private structs into internal headers
- stop exposing model/state/block layout in public interfaces
- consider `pimpl` only after the split is complete

## Proposed Directory Layout

Recommended source layout:

```text
src/
  common/
    audio_io.cpp
    audio_resample.cpp
    process_metrics.cpp
    speaker_embedding_io.cpp
  pipeline/
    pipeline_models.cpp
    pipeline_synthesize.cpp
    pipeline_runtime.cpp
  transformer/
    transformer_internal.h
    transformer_loader.cpp
    transformer_cache.cpp
    transformer_embeddings.cpp
    transformer_graph_talker.cpp
    transformer_graph_code_pred.cpp
    transformer_runtime.cpp
    transformer_generate.cpp
    transformer_debug.cpp
  encoder/
    encoder_internal.h
    encoder_loader.cpp
    encoder_frontend.cpp
    encoder_graph.cpp
    encoder_runtime.cpp
  decoder/
    decoder_internal.h
    decoder_loader.cpp
    decoder_layers.cpp
    decoder_graph.cpp
    decoder_runtime.cpp
```

The existing public headers may remain in `src/` initially to minimize churn.

## Proposed Splits

### `TTSTransformer`

Current file:

- `src/tts_transformer.cpp`

Proposed split:

- `src/transformer/transformer_loader.cpp`
  - `parse_config()`
  - `create_tensors()`
  - `load_tensor_data()`
  - `load_model()`
  - `unload_model()`
  - `try_init_coreml_code_predictor()`

- `src/transformer/transformer_cache.cpp`
  - `init_kv_cache()`
  - `clear_kv_cache()`
  - `init_code_pred_kv_cache()`
  - `clear_code_pred_kv_cache()`
  - `maybe_reserve_scheduler_graphs()`

- `src/transformer/transformer_embeddings.cpp`
  - `lookup_embedding_rows()`
  - `lookup_single_embedding_row()`
  - `project_text_tokens()`
  - `build_prefill_graph()`
  - named speaker lookup helpers

- `src/transformer/transformer_graph_talker.cpp`
  - `build_prefill_forward_graph()`
  - `build_step_graph()`

- `src/transformer/transformer_graph_code_pred.cpp`
  - `build_code_pred_graph()`
  - `build_code_pred_prefill_graph()`
  - `build_code_pred_step_graph()`

- `src/transformer/transformer_runtime.cpp`
  - `forward_prefill()`
  - `forward_text()`
  - `forward_step()`
  - `forward_codec()`
  - `get_hidden_states()`
  - `predict_codes()`

- `src/transformer/transformer_generate.cpp`
  - `predict_codes_autoregressive_coreml()`
  - `predict_codes_autoregressive()`
  - `generate()`
  - local sampling helpers such as `argmax()`

- `src/transformer/transformer_debug.cpp`
  - debug trace config
  - manifest/text/bin dump helpers

### `Qwen3TTS`

Current file:

- `src/qwen3_tts.cpp`

Proposed split:

- `src/pipeline/pipeline_models.cpp`
  - `load_models()`
  - model discovery
  - lazy-load policy

- `src/pipeline/pipeline_synthesize.cpp`
  - `synthesize()`
  - `synthesize_with_voice()`
  - `synthesize_with_speaker_embedding()`
  - `extract_speaker_embedding()`
  - `synthesize_internal()`
  - `set_progress_callback()`

- `src/pipeline/pipeline_runtime.cpp`
  - GGML log filtering
  - timing helpers
  - process memory metrics

- `src/common/audio_io.cpp`
  - WAV read/write helpers

- `src/common/audio_resample.cpp`
  - `resample_linear()`

- `src/common/speaker_embedding_io.cpp`
  - speaker embedding text/bin parse/load/save

### `AudioTokenizerEncoder`

Current file:

- `src/audio_tokenizer_encoder.cpp`

Proposed split:

- `src/encoder/encoder_loader.cpp`
  - model parsing/loading

- `src/encoder/encoder_frontend.cpp`
  - filterbank generation
  - window generation
  - DFT/STFT helpers
  - mel spectrogram creation

- `src/encoder/encoder_graph.cpp`
  - graph construction helpers

- `src/encoder/encoder_runtime.cpp`
  - `encode()`

### `AudioTokenizerDecoder`

Current file:

- `src/audio_tokenizer_decoder.cpp`

Proposed split:

- `src/decoder/decoder_loader.cpp`
  - model loading
  - codebook normalization
  - unload path

- `src/decoder/decoder_layers.cpp`
  - `apply_snake()`
  - `apply_rms_norm()`
  - `apply_pre_tfm_layer()`
  - `apply_upsample_block()`
  - `apply_residual_block()`
  - `apply_decoder_block()`

- `src/decoder/decoder_graph.cpp`
  - graph cache helpers
  - `build_graph()`
  - `build_graph_impl()`

- `src/decoder/decoder_runtime.cpp`
  - `decode()`

## Header Cleanup Plan

After the `.cpp` split is stable, reduce public header weight.

Recommended changes:

1. Keep public headers limited to:
   - public config/result types
   - public class declarations
   - public free-function declarations intended for consumers

2. Move implementation-only structs into internal headers:
   - `transformer_internal.h`
   - `encoder_internal.h`
   - `decoder_internal.h`

3. Remove large private helper declarations from public headers where possible by:
   - introducing internal helper classes or free functions in internal headers
   - using internal namespaces within each subsystem

4. Consider `pimpl` only after the split is complete.

`pimpl` is attractive for `TTSTransformer`, `AudioTokenizerEncoder`, and `AudioTokenizerDecoder`, but it is a Phase 2 or Phase 3 cleanup, not the first move.

## Recommended Migration Phases

### Phase 1: Split implementation files only

Scope:

- Introduce subsystem directories under `src/`
- Move logic into new `.cpp` files
- Keep existing class APIs stable
- Keep tests/build behavior unchanged

Exit criteria:

- Build and test commands remain the same
- No behavioral regressions
- `tts_transformer.cpp` reduced to a thin facade or removed entirely

### Phase 2: Move private structs and helpers into internal headers

Scope:

- create internal headers for transformer/encoder/decoder
- reduce public header surface
- remove implementation-only types from public headers where possible

Exit criteria:

- public headers are materially smaller
- implementation types no longer leak widely across the project

### Phase 3: Optional deeper encapsulation

Scope:

- evaluate `pimpl` for large runtime classes
- tighten translation-unit boundaries further

Exit criteria:

- public compile surface is stable and minimal
- internal ownership boundaries are clearer than today

## Refactor Order

Recommended order:

1. Transformer
2. Pipeline facade and common utilities
3. Decoder
4. Encoder

Why this order:

- `TTSTransformer` is the largest and highest-churn file
- `Qwen3TTS` becomes easier to clean once transformer responsibilities are clearer
- decoder and encoder splits are straightforward after the common pattern is established

## Guardrails

During refactor, preserve the following:

- public API names and behavior unless there is a deliberate API change
- exact prefill embedding behavior
- code predictor KV-cache semantics
- backend selection order
- deterministic/reference tests and regression scripts
- existing model-loading assumptions unless explicitly improved

Avoid combining refactor and optimization in the same changeset unless the optimization is trivial and mechanically obvious.

## Validation Expectations

After each major split:

- build all existing targets
- run component tests where available
- run at least one CLI smoke test
- for transformer work, run `test_transformer`
- for pipeline changes, verify speaker embedding I/O and WAV save/load behavior

## Immediate Next Refactor Task

Start with Phase 1 on `TTSTransformer`.

First concrete step:

1. Create `src/transformer/transformer_internal.h`
2. Move debug trace helpers into `src/transformer/transformer_debug.cpp`
3. Move loader/config/tensor functions into `src/transformer/transformer_loader.cpp`
4. Update `CMakeLists.txt` to compile the new transformer source list
5. Rebuild and run transformer-focused tests before touching graph logic

This keeps the first refactor pass mostly mechanical and low-risk while immediately shrinking the largest file.
