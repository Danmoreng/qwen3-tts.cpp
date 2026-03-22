# Quickstart: Reference-Text Voice Cloning

## Goal

Exercise the intended user journey for creating and reusing a richer
voice-cloning prompt while preserving the current audio-only path.

## Prerequisites

- Built `qwen3-tts-cli`
- Local model directory under `models/`
- A short reference WAV file
- Matching transcript text for the reference WAV
- For reference-aware prompt creation only: the upstream `qwen-tts` Python
  package and local Hugging Face model assets under `models/`

## 1. Create a reference-aware prompt asset

```bash
./build/qwen3-tts-cli \
  -m models \
  -t "Prompt creation smoke test." \
  -r examples/readme_clone_input.wav \
  --reference-text-file reference_text.txt \
  --dump-voice-clone-prompt prompt_reftext.json \
  -o /tmp/prompt-build.wav
```

Expected outcome:

- `prompt_reftext.json` is created
- The file reports `reference_aware` mode
- The command uses the helper bridge in `scripts/create_voice_clone_prompt.py`
  to extract `reference_codes`

## 2. Reuse the richer prompt asset for multiple utterances

```bash
./build/qwen3-tts-cli \
  -m models \
  -t "First reused utterance." \
  --voice-clone-prompt prompt_reftext.json \
  -o out_reuse_1.wav

./build/qwen3-tts-cli \
  -m models \
  -t "Second reused utterance." \
  --voice-clone-prompt prompt_reftext.json \
  -o out_reuse_2.wav
```

Expected outcome:

- Both commands succeed
- Neither command requires the original `--reference` audio or reference text
- Runtime logs identify prompt reuse rather than fresh prompt extraction

## 3. Preserve the current audio-only workflow

```bash
./build/qwen3-tts-cli \
  -m models \
  -t "Audio-only cloning remains available." \
  -r examples/readme_clone_input.wav \
  --dump-speaker-embedding speaker.json \
  -o out_audio_only.wav
```

Expected outcome:

- The current audio-only path still works unchanged
- Existing saved speaker-embedding assets remain reusable

## 4. Validate compatibility failures

Try one of the following:

- Edit `prompt_reftext.json` and break a required field
- Reuse a prompt generated for a different model family
- Remove `reference_codes` from a `reference_aware` asset

Expected outcome:

- The command fails before synthesis starts
- The error clearly states that the prompt asset is malformed or incompatible

## 5. Validation targets before merge

- Deterministic reference generation covers both `audio_only` and
  `reference_aware` prompt modes
- Component and end-to-end tests cover creation, reuse, and compatibility
  rejection
- README examples explain when to use audio-only cloning versus
  reference-text-aware cloning
- Helper-path requirements are documented clearly so missing Python/HF
  dependencies fail fast instead of silently downgrading behavior
