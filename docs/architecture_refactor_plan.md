# Architecture Refactor Plan

Last updated: 2026-03-14

## Purpose

This document captures the current code-structure findings for the native `qwen3-tts.cpp` codebase and proposes a concrete refactor path to make the project easier to maintain without changing model behavior.

The immediate goal is not to redesign the runtime architecture. The goal is to:

- Split oversized source files into clearer modules
- Reduce implementation detail leakage in public headers
- Make ownership boundaries easier to understand
- Preserve current behavior, tests, and performance characteristics while refactoring

## Execution Status

Current branch status on `refactor/architecture-split`:

- Completed: internal transformer helper boundary introduced in `src/transformer/transformer_internal.h`
- Completed: debug trace implementation extracted into `src/transformer/transformer_debug.cpp`
- Completed: transformer model loading/config/tensor setup extracted into `src/transformer/transformer_loader.cpp`
- Completed: transformer KV-cache and scheduler reservation lifecycle extracted into `src/transformer/transformer_cache.cpp`
- Completed: transformer embedding lookup, named speaker resolution, and prefill embedding construction extracted into `src/transformer/transformer_embeddings.cpp`
- Completed: talker graph construction extracted into `src/transformer/transformer_graph_talker.cpp`
- Completed: code predictor graph construction extracted into `src/transformer/transformer_graph_code_pred.cpp`
- Completed: talker runtime execution extracted into `src/transformer/transformer_runtime.cpp`
- Completed: code predictor runtime execution extracted into `src/transformer/transformer_runtime_code_pred.cpp`
- Completed: outer autoregressive generation extracted into `src/transformer/transformer_generate.cpp`
- Completed: transformer private model/state/runtime members moved out of `src/tts_transformer.h` into `src/transformer/transformer_state_internal.h`
- Completed: public transformer header no longer carries private helper member declarations; those now route through `transformer_internal::ops` in `src/transformer/transformer_internal.h`
- Completed: pipeline runtime/logging/memory/resample helpers extracted into `src/pipeline/pipeline_runtime.cpp`
- Completed: pipeline model discovery, lazy-load policy, and capability helpers extracted into `src/pipeline/pipeline_models.cpp`
- Completed: pipeline synthesis orchestration extracted into `src/pipeline/pipeline_synthesize.cpp`
- Completed: WAV load/save helpers extracted into `src/common/audio_io.cpp`
- Completed: speaker embedding parse/load/save helpers extracted into `src/common/speaker_embedding_io.cpp`
- Completed: decoder-private model/state/layout structs moved into `src/decoder/decoder_internal.h`
- Completed: decoder model loading, codebook normalization, and unload lifecycle extracted into `src/decoder/decoder_loader.cpp`
- Completed: decoder layer helper implementations extracted into `src/decoder/decoder_layers.cpp`
- Completed: decoder graph construction extracted into `src/decoder/decoder_graph.cpp`
- Completed: decoder cached-graph lifecycle extracted into `src/decoder/decoder_cache.cpp`
- Completed: decoder runtime execution extracted into `src/decoder/decoder_runtime.cpp`
- Completed: decoder private model/state/scratch storage moved behind `src/decoder/decoder_state_internal.h`
- Completed: public decoder header no longer includes `src/decoder/decoder_internal.h`
- Completed: encoder frontend DSP helpers extracted into `src/encoder/encoder_frontend.cpp`
- Completed: encoder graph construction extracted into `src/encoder/encoder_graph.cpp`
- Completed: encoder model loading/backend setup extracted into `src/encoder/encoder_loader.cpp`
- Completed: encoder runtime execution extracted into `src/encoder/encoder_runtime.cpp`
- Completed: encoder private model/state storage moved behind `src/encoder/encoder_state_internal.h`
- Completed: public encoder header no longer exposes GGML-heavy private storage details
- Confirmed after each completed step: local rebuild and test pass on the current Windows/CUDA workflow

Current transformer split status:

- Still in `src/tts_transformer.cpp`: constructor/destructor facade, legacy forward wrappers, and free helpers
- Now moved out of `src/tts_transformer.cpp`: debug trace helpers, model load/unload path, GGUF config parsing, tensor creation, tensor data loading, CoreML loader hookup, KV-cache lifecycle, scheduler reserve warmup, embedding lookup helpers, named speaker lookup, prefill embedding construction, talker graph builders, code predictor graph builders, talker runtime execution, code predictor runtime execution, outer autoregressive generation
- Now moved out of `src/tts_transformer.h`: `transformer_layer`, `tts_transformer_model`, `tts_transformer_state`, `tts_kv_cache`, timing state, CoreML/runtime members, GGML/GGUF-heavy private storage details, and the former private graph/build/runtime helper declarations
- Remaining transformer cleanup is mostly optional internal boundary polish rather than a blocking structural split

Recommended next step from this point:

- Keep the encoder surface stable and limit follow-up work there to small boundary polish if it falls out naturally
- Shift the next structural pass to whichever remaining subsystem still has the largest maintenance burden

Guardrail for ongoing work:

- Keep each extraction functionality-preserving and stop for rebuild/test after every mechanical step

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
| `src/audio_tokenizer_encoder.cpp` | 712 | DSP frontend plus GGML runtime in one file |
| `src/tts_transformer.h` | 172 | Public header is materially smaller after helper-declaration cleanup |
| `src/decoder/decoder_layers.cpp` | 192 | Decoder layer helper implementations are now isolated |
| `src/qwen3_tts.h` | 173 | Public facade is stable after pipeline/common splits |
| `src/decoder/decoder_graph.cpp` | 145 | Decoder graph assembly is now isolated |
| `src/decoder/decoder_loader.cpp` | 329 | Decoder model loading and backend setup are isolated |
| `src/audio_tokenizer_decoder.h` | 101 | Public decoder header now uses a private impl boundary |
| `src/decoder/decoder_runtime.cpp` | 67 | Decoder runtime execution is now isolated |
| `src/decoder/decoder_cache.cpp` | 52 | Decoder cached graph lifecycle is now isolated |
| `src/tts_transformer.cpp` | 58 | Thin facade/free-helper translation unit after Phase 1 split |
| `src/qwen3_tts.cpp` | 8 | Thin facade translation unit |

## Findings

### 1. `TTSTransformer` has too many responsibilities

Historically, the transformer implementation owned all of the following in one place:

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

This is no longer concentrated in one translation unit. The remaining transformer work is mainly boundary cleanup, while the encoder file is now the largest unsplit implementation hotspot.

### 2. The public transformer header is carrying implementation-only data

`src/tts_transformer.h` originally exposed:

- `tts_transformer_model`
- `tts_transformer_state`
- `tts_kv_cache`
- `transformer_layer`
- most private helper declarations

The struct/type exposure is now behind `src/transformer/transformer_state_internal.h`, and the former private helper declarations now live behind the internal `transformer_internal::ops` boundary. The public header is no longer the primary structural problem.

### 3. `Qwen3TTS` mixes orchestration with unrelated utility code

`src/qwen3_tts.cpp` is now effectively a thin facade translation unit.

The runtime/logging, model-loading, and synthesis orchestration slices have been moved into `src/pipeline/`, and the remaining WAV/speaker-embedding utilities have been moved into `src/common/`.

These are separable concerns that do not need to live in a single translation unit.

### 4. Encoder split is now structurally complete

The encoder is now separated into focused units:

- `src/encoder/encoder_frontend.cpp`
- `src/encoder/encoder_graph.cpp`
- `src/encoder/encoder_loader.cpp`
- `src/encoder/encoder_runtime.cpp`
- `src/encoder/encoder_state_internal.h`

The public header no longer exposes encoder-private GGML state.

### 5. Decoder split is now structurally complete

The decoder is now separated into focused units:

- `src/decoder/decoder_loader.cpp`
- `src/decoder/decoder_layers.cpp`
- `src/decoder/decoder_graph.cpp`
- `src/decoder/decoder_cache.cpp`
- `src/decoder/decoder_runtime.cpp`
- `src/decoder/decoder_state_internal.h`

The public header no longer exposes decoder-private model/state storage.

The decoder and encoder are no longer the primary refactor hotspots; the remaining work is smaller boundary cleanup.

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

### Principle 4: Preserve history where practical

Large file splits can destroy useful `git blame` history if they are done as a single copy-paste rewrite.

Recommended approach:

- extract one subsystem slice at a time
- keep the first extraction mechanically simple
- prefer `git mv` when one file is the obvious primary successor
- accept normal Git copy detection for smaller helper files rather than forcing a noisy all-at-once duplication strategy

The goal is to preserve useful history without creating a temporary explosion of duplicated code.

### Principle 5: Lock a same-environment regression baseline before movement

Before Phase 1 begins, capture a deterministic baseline in the current environment.

Recommended baseline artifacts:

- existing `test_transformer` output expectations
- one fixed CLI smoke prompt under deterministic settings
- optional debug trace or speech-code dump for the same prompt
- optional same-machine hash of generated WAV and/or code output

Important constraint:

- bit-exact hashes are only a strong guard when backend, platform, and build configuration stay fixed

Use hashes as a local refactor guard, not as a portable cross-environment invariant.

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
Do not scaffold every file up front before code is moved. Create directories and implementation files incrementally as each extraction starts.

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
   - `transformer_model_internal.h`
   - `transformer_graph_internal.h`
   - `transformer_debug_internal.h`
   - `encoder_internal.h`
   - `decoder_internal.h`

3. Remove large private helper declarations from public headers where possible by:
   - introducing internal helper classes or free functions in internal headers
   - using internal namespaces within each subsystem

Avoid replacing one oversized public header with one oversized internal header. Internal headers should also be split by concern when the subsystem is large enough to justify it.

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

Refactor commits should also stay mechanically legible:

- separate file moves from behavioral edits where possible
- separate header-surface reduction from implementation relocation
- keep the first extraction in each subsystem intentionally conservative

## Validation Expectations

After each major split:

- build all existing targets
- run component tests where available
- run at least one CLI smoke test
- for transformer work, run `test_transformer`
- for pipeline changes, verify speaker embedding I/O and WAV save/load behavior

Before each major split:

- record the branch tip and target files being extracted
- capture the same-environment deterministic baseline used for comparison

## Immediate Next Refactor Task

Continue Phase 1 on `AudioTokenizerDecoder`.

First concrete step:

1. Create `src/decoder/decoder_layers.cpp` for `apply_snake()`, `apply_rms_norm()`, and the upsample/residual/decoder block helpers
2. Leave `build_graph*()` and `decode()` in `src/audio_tokenizer_decoder.cpp` for that step
3. Rebuild and run the standard Windows/CUDA regression pass plus `test_decoder`
4. Follow with graph-cache/graph-assembly extraction into `src/decoder/decoder_graph.cpp`

This keeps the next pass mechanical while continuing to shrink the decoder without changing its public API.
