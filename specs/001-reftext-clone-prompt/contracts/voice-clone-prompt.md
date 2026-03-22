# Contract: Reference-Text Voice Clone Prompt

## Goal

Define the additive public contract for creating, persisting, loading, and
reusing richer voice-clone prompts while preserving the existing audio-only
speaker-embedding flow.

## CLI Contract

### Existing Inputs That Must Remain Valid

- `--reference <file>` for audio-only cloning from a reference WAV
- `--speaker-embedding <file>` for reusing a saved audio-only embedding
- `--dump-speaker-embedding <file>` for persisting the current audio-only asset

### New Inputs

- `--reference-text <text>`: Provide text that matches the reference audio.
- `--reference-text-file <file>`: Load matching reference text from a file.
- `--voice-clone-prompt <file>`: Reuse a previously created prompt asset.
- `--dump-voice-clone-prompt <file>`: Save a reusable prompt asset created from
  `--reference` and optional reference text.

### CLI Validation Rules

- `--voice-clone-prompt` is mutually exclusive with `--reference`,
  `--speaker-embedding`, and `--speaker`.
- `--dump-voice-clone-prompt` requires `--reference`.
- `--reference-text` and `--reference-text-file` are mutually exclusive.
- Supplying reference text with `--reference` requests `reference_aware` mode.
- Supplying `--reference` without reference text keeps the current `audio_only`
  mode.
- Invalid prompt assets or incompatible prompt metadata fail with a clear error.
- Reference-aware prompt creation currently depends on the helper bridge in
  `scripts/create_voice_clone_prompt.py` and upstream Python/HF assets.

## C++ API Contract

### New Public Types

- `voice_clone_prompt_mode`: enum with `audio_only` and `reference_aware`
- `voice_clone_prompt_asset`: reusable native struct matching the persisted asset
- `voice_clone_prompt_validation`: structured compatibility-check result

### New Public Operations

- `create_voice_clone_prompt(reference_audio, reference_text, out_asset)`
- `synthesize_with_voice_clone_prompt(text, asset, params)`
- `load_voice_clone_prompt_file(path, out_asset)`
- `save_voice_clone_prompt_file(path, asset)`
- `validate_voice_clone_prompt(asset)`

### Compatibility Rules

- Existing `synthesize_with_voice`, `synthesize_with_speaker_embedding`, and
  `extract_speaker_embedding` remain supported.
- Audio-only assets can still be represented and reused without reference text.
- Reference-aware assets add behavior; they do not redefine the meaning of the
  existing speaker-embedding-only entry points.

## C API Contract

### New Public Types

- `qwen3_tts_voice_clone_prompt_validation_t`

### New Public Operations

- `qwen3_tts_create_voice_clone_prompt(...)`
- `qwen3_tts_synthesize_with_voice_clone_prompt(...)`
- `qwen3_tts_validate_voice_clone_prompt(...)`
- Matching free helpers for validation error strings

### ABI Rules

- All new structs use explicit integer widths and pointer ownership rules.
- Existing C ABI functions remain intact.
- New fields are added through new types or new functions instead of mutating
  the memory layout of existing public structs in an ABI-breaking way.

## Persisted Asset Shape

### JSON Object

```json
{
  "format_version": "1",
  "prompt_mode": "reference_aware",
  "model_kind": "base",
  "model_name": "qwen3-tts-1.7b-base-f16.gguf",
  "speaker_embedding_dim": 2048,
  "speaker_embedding": ["<float>", "..."],
  "reference_text": "Matching transcript for the reference audio",
  "reference_codebooks": 16,
  "reference_frames": 36,
  "reference_codes": ["<int>", "..."]
}
```

### Asset Semantics

- `audio_only` assets store `speaker_embedding` and compatibility metadata only.
- `reference_aware` assets additionally store `reference_text` and
  `reference_codes`.
- Consumers MUST validate `format_version`, `model_kind`, and
  `speaker_embedding_dim` before reuse.
- Reference-aware assets created in the current implementation are produced by
  the upstream helper bridge, while asset reuse remains native in C++.

## Verification Contract

- Deterministic reference scripts must be able to create and serialize both
  prompt modes.
- Regression tests must prove that reference-aware reuse avoids requiring the
  original inputs again.
- Compatibility tests must cover malformed JSON, missing required fields,
  incompatible model metadata, and unsupported prompt modes.
