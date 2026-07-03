# Framework Benchmarking Notes

This document describes how to run comparable Qwen3 TTS framework benchmarks for README-quality numbers. The benchmark harness lives in `scripts/benchmark_frameworks.ps1`.

## Harness

Preflight only, without synthesis:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark_frameworks.ps1 -ValidateOnly
```

Run a comparable voice-clone benchmark later:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark_frameworks.ps1 `
  -Scenario voice_clone `
  -Variant 1.7b-base `
  -BenchmarkMode both `
  -ReferenceMaxSec 5.95 `
  -Runs 3 `
  -MaxTokens 128
```

The harness records one CSV with raw runs, one CSV summary, logs, generated WAVs, repository metadata, audio duration, peak/RMS/silence sanity checks, wall time, and RTF. For publishable throughput numbers, use `GenerationSeconds` and `RTF_AudioPerGeneration`: these are based on the measured generate+decode phase and intentionally exclude model load, process startup, reference audio trimming, and speaker/voice-prompt encoding when the framework exposes those phases separately. `WallSeconds` remains a cold-start/process diagnostic.
Rows include `PromptMode` so speaker-embedding-only runs are not mixed with full
ICL voice-clone runs. `PromptMode=speaker_embedding` means the synthesis phase
uses a precomputed speaker embedding/x-vector. `PromptMode=full_icl` means the
reference transcript and reference speech codes are part of the workload.
`PromptMode=full_icl_derived` is used when a framework only exposes internal
timings from a full ICL request rather than a true preencoded prompt API.
Rows also include `BenchmarkScope`, `ModelFormat`, `Precision`,
`ReferenceAudioSec`, and `ReferenceAudioSourceSec`. Publish comparisons only
within the same prompt mode, scope, format, and precision unless the table is
explicitly labeled as a mixed-precision/product comparison.

Use `-ReferenceMaxSec 5.95` (or another explicit value) when the source
reference is long. The harness writes a trimmed WAV and gives that same file to
every implementation, so ICL tokenization, speaker-embedding extraction, and
decoder context all see the same reference duration.

Faster CUDA-graphs streaming/TTFA rows are opt-in and should be reported
separately. The helper performs a warm-up generation first so CUDA graph capture
and cache setup are not included in the measured streaming request:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark_frameworks.ps1 `
  -Implementations faster_python `
  -Scenario voice_clone `
  -Variant 1.7b-base `
  -BenchmarkMode full `
  -ReferenceMaxSec 5.95 `
  -FasterStreaming `
  -FasterChunkSize 8 `
  -Runs 3
```

Those rows use `BenchmarkScope=streaming_ttfa` and include `InternalTtfaMs`.
Do not average them with CLI/process-scope rows. Use `GenerationSeconds` for
best-case throughput and `InternalTtfaMs` for warm TTFA.

For a resident qwen3-tts.cpp session approximation, use
`-QwenCppSessionRepeats N`. This passes `--repeat N` to the CLI, keeps models
loaded inside one process, and records the final repeated WAV/timing row as
`BenchmarkScope=session_repeat`. Use at least `N=2` for best-case rows; the
first request warms the resident session and later requests represent cached
generation.

For audio.cpp, use `-AudioCppSessionRepeats N`. The harness writes a
`--request-sequence` JSON file, runs all requests inside one offline session,
and records the final `measure.wav` row as `BenchmarkScope=session_repeat`.

## Implementation Inputs

For publishable results, record the exact implementation commit or release,
model format, model precision, benchmark mode, and prompt mode alongside each
table. Keep the benchmarked repositories and model artifacts stable for the
full run. If a comparison mixes model formats or precisions, label it as a
mixed-product comparison rather than a pure implementation-efficiency result.

The harness accepts explicit paths for each implementation and model artifact.
Pass matching GGUF model files for GGUF-to-GGUF comparisons, and pass matching
HF model ids or filesystem HF model directories for Python/audio.cpp
comparisons.

## Notes On Fairness

- The current shared scenario is `voice_clone`, because the Python base implementations are centered on reference-audio generation.
- `basic` is useful for C++ stacks, but is not yet a fair all-framework scenario.
- The default scope measures command-line process wall time. That includes model load for each process. Publish generation throughput from `GenerationSeconds` / `RTF_AudioPerGeneration`; process wall time remains useful for cold-start reporting only. Use `BenchmarkScope=session_repeat` for warmed in-process qwen3-tts.cpp and audio.cpp runs, and `BenchmarkScope=streaming_ttfa` for warmed faster-qwen3-tts CUDA-graphs runs.
- Compare `PromptMode=full_icl` rows against other full ICL rows, and compare
  `PromptMode=speaker_embedding` rows against other speaker-embedding rows.
  Do not collapse them into one voice-clone table.
- Audio sanity checks are built in so invalid, empty, silent, or very low-level
  WAV files are marked separately from process success.
- For strict quantization comparisons, pass matching model files and Python dtypes explicitly.
