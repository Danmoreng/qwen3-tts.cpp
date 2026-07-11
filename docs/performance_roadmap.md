# Performance Optimization Roadmap

Last updated: 2026-07-11

## Purpose

This document is the persistent source of truth for performance work in
`qwen3-tts.cpp`. Update it whenever an optimization is proposed, benchmarked,
accepted, or rejected. It records negative results as well as successful work
so that later sessions do not repeat experiments without a new technical reason.

The general project plan remains in `docs/development_plan.md`. Reproducible
cross-framework benchmark rules are documented in
`docs/benchmarking-frameworks.md`.

## Status Legend

| Status | Meaning |
|---|---|
| Completed | Implemented, correctness-tested, benchmarked, and retained |
| Rejected | Implemented or prototyped, measured, and removed because it regressed or did not justify its cost |
| Inspected | Source/backend behavior verified; no useful project change identified |
| Open | Not yet implemented and benchmarked in this repository |
| Partial | Only a smaller version of the proposed architecture was tested |

## Current Performance Model

- The Code Predictor remains the primary generation bottleneck because each
  audio frame requires one prefill plus 14 autoregressive codebook steps.
- The Talker is the second-largest generation component.
- CUDA Graph replay is essential for this launch-heavy workload. Historical
  warm 0.6B measurements showed roughly `0.60 s` with CUDA Graphs versus
  `2.18 s` without them for 64 frames.
- Graph construction and allocation are already negligible in the steady-state
  generation profile. Remaining transformer gains primarily come from fewer
  kernel nodes, fewer synchronizations, and less repeated vector preparation.
- The decoder is smaller than generation in offline inference, but becomes more
  important for long audio, streaming latency, and pipeline overlap.

Primary test hardware for the measurements below:

- Windows
- NVIDIA GeForce RTX 5080 Laptop GPU, 16 GB
- CUDA backend, native `sm_120a` build
- Q8_0 GGUF models from `%USERPROFILE%\.qwen-tts-studio\models`

## Completed Optimizations

| Optimization | Commit | Result | Correctness |
|---|---|---|---|
| Forward benchmark seed to qwen3-tts | `23e6bbf` | Makes framework and A/B runs reproducible | Benchmark command lines verified |
| Use `ggml_swiglu_split()` in recurrent Talker and Code Predictor FFNs | `e9c4a21` | Retained after benchmark and regression testing | Deterministic/component and audio gates passed |
| Reuse physical Code Predictor KV storage instead of synchronously zeroing it every frame | `99ebc44` | Removes ten small synchronous clears per frame; legacy clear remains behind `QWEN3_TTS_CODE_PRED_ZERO_KV=1` | Sampled, greedy, ICL, multi-request, and full regression gates passed |
| Pack Code Predictor recurrent Q/K/V projections on CUDA | `93867a5` | Positive on both 0.6B and 1.7B | Generated codes and WAVs were byte-identical in final A/B cases |
| Pack Talker recurrent Q/K/V projections on CUDA for models wider than 1024 | `93867a5` | Retained for 1.7B; disabled for 0.6B after a small regression | Generated codes and WAVs were byte-identical |
| Correct resident framework benchmark path and publish current README numbers | `93867a5` and preceding timing commits | Current README tables use warm generation metrics and fixed sampling parameters | All benchmark WAV sanity checks passed |
| Use qwentts `examples/freeman.wav` plus its sidecar transcript as the default qwentts matrix reference | `a9534cd` | Prevents accidental comparison with the unrelated local README reference | Default preflight and corrected 8-run matrix passed |

### Packed QKV Final A/B Result

The final QKV comparison used eight interleaved process pairs per model. Each
mode used two warm-ups and three resident measured requests, giving 24 measured
requests per mode and model.

| Model | Baseline median generation | Packed-QKV median | Paired median gain | Positive pairs |
|---|---:|---:|---:|---:|
| 0.6B Q8_0 | 851 ms | 819 ms | 4.11% | 8/8 |
| 1.7B Q8_0 | 1121.5 ms | 1072 ms | 4.97% | 8/8 |

Correctness cases were `0.6b_sampled`, `0.6b_greedy`, `1.7b_sampled`,
`1.7b_greedy`, and `1.7b_icl`. Generated-code and WAV SHA-256 hashes matched
within every before/after pair.

Trade-off: packed Talker QKV adds approximately 0.22 GB of device allocation
for 1.7B. Packed tensors are only created in CUDA builds. The 0.6B Talker path
keeps separate Q/K/V weights and projections.

Runtime fallbacks:

- `QWEN3_TTS_CODE_PRED_PACKED_QKV=0`
- `QWEN3_TTS_TALKER_PACKED_QKV=0`
- `QWEN3_TTS_CODE_PRED_ZERO_KV=1`

## Partial Optimizations

### Sum Rest-Codebook Embeddings Before the Shared Decoder Projection

Status: Partial

The decoder now sums the 15 F32 rest-codebook embeddings before their shared,
bias-free projection, removing 14 matrix multiplications. CUDA measurements
showed a small win for short decodes but regressions at both 64 and 252 frames,
so the automatic path is deliberately conservative:

- CUDA inputs with at most 63 frames use the summed projection.
- Longer CUDA inputs and all unmeasured backends keep the legacy graph.
- `QWEN3_TTS_DECODER_SUM_REST_EMBEDDINGS=0` forces the legacy graph.
- `QWEN3_TTS_DECODER_SUM_REST_EMBEDDINGS=1` forces the summed graph for
  diagnostics and future backend measurements.

No model weights or persistent runtime allocations are added. The experiment
used base commit `ed4ecf3`, GGML `0.15.3` at `eced84c8`, and
`qwen-tokenizer-12hz-Q8_0.gguf` with SHA-256
`1883BEEED99348FC35E23DD225E9082F93F6F8C109330A33D935BAA8ACDBFD94`.
Raw artifacts are under the ignored
`benchmark_output/decoder_rest_embedding_sum` directory.

The final fixed-code A/B used eight alternating process pairs per length. Each
process ran the two normal full-input correctness decodes, two additional
benchmark warm-ups, and ten resident measurements with the Q8_0 codec on the
primary RTX 5080 Laptop GPU. The raw medians below pool the 80 measured decodes;
the paired median uses the eight process-mean pairs.

| Frames | Legacy raw median | Summed raw median | Raw median gain | Paired median gain | Positive pairs |
|---:|---:|---:|---:|---:|---:|
| 32 | 20.170 ms | 20.029 ms | 0.70% | 0.98% | 5/8 |
| 63 | 105.070 ms | 104.254 ms | 0.78% | 2.81% | 7/8 |
| 64 | 40.764 ms | 44.897 ms | -10.14% | -0.92% | 3/8 |
| 252 | 202.431 ms | 208.245 ms | -2.87% | -3.03% | 2/8 |

Correctness validation covered fixed inputs of 1, 63, 64, 73, and 252 frames.
Every output had the expected sample count and finite values. For the active
63-frame path, candidate-versus-legacy maximum absolute error was `0.00191306`,
mean absolute error was `0.000086634`, RMSE was `0.000147576`, and cosine
similarity was `0.999716`. The default 64- and 252-frame paths remained
byte-identical to legacy. The five seeded end-to-end cases (`0.6B`
sampled/greedy, `1.7B` sampled/greedy/ICL) produced identical generated and
decoder code files and equal durations; their worst RMSE was `0.000898` in
PCM16 samples normalized to `[-1, 1]`. The component suite and standard CLI
regression suite passed with 11 passes and no failures. Before/after WAVs are
preserved in the artifact directory for a manual listening pass; no human
perceptual result is claimed here.

The optional 0.6B ICL smoke still produced a pre-existing `0.08 s` output in
both modes. Its generated-code and decoder-code files were byte-identical
between legacy and summed runs, so it is not attributed to this decoder change.

`test_decoder` now supports `--bench-warmup` and `--bench-runs`, emitting
machine-readable per-run and summary timing for future decoder experiments.
Before widening the automatic path, benchmark the untested 65-251-frame range
and each non-CUDA backend independently. A fused gather/reduction is the likely
follow-up if the long-input CUDA regression is revisited.

## Evaluated and Rejected Experiments

Do not repeat these unchanged. Revisit only if GGML kernels, tensor layouts,
hardware, or the surrounding graph architecture changes materially.

| Experiment | Status | Measurement / reason |
|---|---|---|
| Pack Code Predictor FFN Gate+Up weights into one matmul | Rejected | Byte-identical, but 0.6B was effectively neutral and 1.7B regressed by roughly 1-2% in paired measurements |
| Standalone device-side greedy `argmax` | Rejected | Byte-identical, but approximately 1.9% slower on 0.6B and 5.6% slower on 1.7B; the extra kernel does not remove the synchronization by itself |
| Naive decoder graph pointer reuse after scheduler reset | Rejected | Invalid allocation lifetime caused a CUDA illegal-memory-access failure on the second decode |
| Dedicated decoder replay scheduler | Rejected | Correct and byte-identical, but neutral/slower offline; streaming graph rebuilds fell from 13 to 7 while decode time only changed from about 323 to 322 ms |
| Additional RMSNorm fusion patch | Inspected | GGML CUDA already fuses the relevant RMSNorm-to-Mul and RMSNorm-to-Mul-to-Add patterns |
| Packed Talker QKV on 0.6B | Rejected | Small paired regression; packed Talker QKV remains limited to models with `hidden_size > 1024` |

## Open Work Queue

### P1: Fuse Talker Frame Embedding Gather-and-Sum

Status: Open

The recurrent Talker input currently uses 16 `get_rows` operations, 15 adds,
and an overlay add per audio frame in
`src/transformer/transformer_graph_talker.cpp`.

Candidate implementation:

- Add a small CUDA gather-sum operator that consumes the 16 token IDs, their
  embedding tables, and the overlay.
- Keep the existing GGML graph path as a non-CUDA fallback and A/B switch.

Expected impact is modest, likely below 1-1.5% of total generation, but the work
is independent of sampling and applies to sampled and greedy generation.

### P1: Decoder Flash Attention

Status: Open

The eight pre-transformer decoder layers still materialize QK, softmax, and VK
manually and upload a quadratic F32 causal/window mask. Evaluate:

- `ggml_flash_attn_ext`
- F16 mask storage if accepted by the backend path
- exact sliding-window semantics

Required validation includes multiple frame lengths around the sliding-window
boundary, decoder reference comparison, and long-audio WAV checks.

### P1: Fuse Decoder Snake Activation

Status: Open

The decoder Snake path expands to multiply, sine, square, multiply, and add.
It is used repeatedly throughout the upsampling decoder. Investigate either:

- the existing `GGML_OP_SNAKE` path if it can represent the required alpha/beta
  semantics exactly, or
- a CUDA graph-fusion pattern for the current expression.

Benchmark decoder compute separately before measuring end-to-end impact.

### P2: Fully Device-Chained Greedy Code Predictor

Status: Partial

The standalone GPU `argmax` experiment regressed because each codebook step
still synchronized with the host. A useful implementation must remove the
roundtrips, for example:

```text
CodePred prefill/step
-> device argmax
-> persistent token bridge
-> next embedding lookup
-> next CodePred step
... repeated for all 15 codebooks
-> one final synchronization/readback
```

Options:

- One unrolled supergraph for all 15 codebooks.
- Multiple replay graphs enqueued on one stream with device-resident bridges.

Initial scope should be greedy only. Sampled top-k/top-p generation requires a
reproducible device sampler and is a separate project.

This is a high-risk change. Require byte-identical greedy codes, long
multi-request testing, both model sizes, CUDA-Graph on/off diagnostics, and a
clear win over the current host-sampled path.

### P2: Asynchronous Code Predictor I/O

Status: Open

Evaluate persistent/pinned host buffers plus asynchronous tensor upload,
compute, and readback. CPU sampling still creates a dependency between steps,
so first prove with a trace that asynchronous APIs remove a real synchronization
or overlap useful CPU work. Do not assume that replacing calls with `_async`
variants is itself an optimization.

### P2: Stateful Streaming Decoder and Pipeline Overlap

Status: Open

The tested replay scheduler did not materially improve performance. A larger
design would keep persistent decoder state per common chunk width and overlap:

```text
generate chunk N+1        decode chunk N
        GPU/stream A  ||  GPU/stream B or bounded worker queue
```

This requires explicit backpressure, bounded memory, deterministic chunk
ordering, final-tail handling, and comparison against non-streaming audio.

### P3: Continuous Request Batching

Status: Open

For server throughput, batch Talker and Code Predictor work across independent
requests so matrix-vector operations become small matrix-matrix operations and
weights are amortized. This is likely the largest structural throughput lever,
but it changes scheduling, cache ownership, sampling state, and public server
behavior. Treat it as a separate architecture milestone rather than a local
kernel optimization.

### P3: Component-Specific Quantization Profiles

Status: Open

Evaluate mixed profiles such as lower-bit Q/K/Gate/Up weights while retaining
Q8 for sensitive V/Down/output/head tensors. This may provide larger speed and
memory wins but consumes a quality budget and therefore requires broader
speaker, language, style, and perceptual validation.

### Backlog: GGML QKV Multistream Optimization

Status: Open / low priority

The experimental `GGML_CUDA_GRAPH_OPT=1` path was not repaired or benchmarked.
Earlier inspection found tensor-naming and backend-global stream-state issues.
Packed QKV already produced a reliable 4-5% gain, so revisit multistream only
after higher-priority items or relevant upstream GGML changes.

## Benchmark and Correctness Protocol

Every candidate should follow the same sequence.

### 1. Record the Baseline

Record:

- Git commit and dirty files
- CUDA/GGML version and GPU
- exact model filenames and quantization
- prompt, reference WAV/transcript, seed, sampling settings, and max tokens
- environment toggles
- warm-up count and measurement count

Do not compare different prompt modes, precisions, model formats, or audio
durations without clearly labeling the comparison.

### 2. Produce Before/After Audio

At minimum generate:

- 0.6B sampled
- 0.6B greedy
- 1.7B sampled
- 1.7B greedy
- 1.7B full ICL

Use identical seeds and inputs. Preserve WAVs and generated-code dumps under a
candidate-specific `benchmark_output` directory for listening. Record hashes in
a CSV or this document because `benchmark_output` itself is normally ignored.

### 3. Run Correctness Gates

```powershell
.\build.ps1 -UseNinja -EnableCuda -Configuration Release -BuildAll
.\scripts\run_all_tests.ps1 -RequireComponentTests
```

Also run focused tests appropriate to the changed component. For decoder work,
use fixed speech-code inputs. For cache or replay work, include multiple
requests in one resident process.

### 4. Measure Performance

For small expected gains:

- Interleave baseline and candidate processes to reduce thermal/time drift.
- Use at least two warm-ups per process.
- Prefer multiple resident measurements per process.
- Start with an early four-pair rejection gate.
- For a promising candidate, use at least eight interleaved pairs and report
  raw median, paired median, positive-pair count, and wall-time impact.
- Use longer runs or more pairs when the result is close to noise.

For isolated decoder measurements, use fixed codes with the resident loop:

```powershell
.\build\test_decoder.exe --tokenizer <codec.gguf> --codes <codes.bin> `
  --reference <audio.bin> --bench-warmup 2 --bench-runs 10
```

The two requested warm-ups are additional to the normal full-input correctness
decodes that `test_decoder` performs before entering its benchmark loop.

For publishable framework comparisons, follow
`docs/benchmarking-frameworks.md` and keep generation throughput separate from
cold process wall time.

### 5. Acceptance Rule

Retain a change only when:

- correctness and audio gates pass;
- the gain is repeatable and positive on the intended model/hardware;
- regressions on other supported models/backends are avoided or explicitly
  gated;
- memory cost and fallback behavior are documented; and
- complexity is justified by the measured end-to-end impact.

Record rejected work in this document before removing its implementation.

## Result Entry Template

Copy this section for each future experiment:

```markdown
### YYYY-MM-DD: Experiment Name

- Status: Open / Completed / Rejected
- Commit or working-tree identifier:
- Hypothesis:
- Changed files:
- Models and hardware:
- Benchmark command/settings:
- Correctness results:
- Audio artifacts:
- Baseline median:
- Candidate median:
- Paired median gain:
- Positive pairs:
- Memory impact:
- Decision and reason:
- Follow-up:
```

## Recommended Execution Order

1. Evaluate Talker frame gather-sum fusion.
2. Evaluate decoder Flash Attention and Snake fusion independently.
3. Design a true device-chained greedy Code Predictor supergraph.
4. Design stateful streaming decoder overlap.
5. Treat continuous batching and mixed quantization as separate milestones.
