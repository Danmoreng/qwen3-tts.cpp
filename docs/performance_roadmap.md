# Performance Optimization Roadmap

Last updated: 2026-07-12

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

## Rejected Optimizations

### Packed Talker Frame Embedding Reduction

Status: Rejected and removed.

Commit `fb9a9e0` made the 15 Code Predictor embedding tables views into one
contiguous tensor and replaced the sequential recurrent Talker additions with
one reduction. The original Q8 A/B checks were byte-identical, but they only
compared the candidate to the immediately preceding C++ trajectory.

The final CUDA comparison used eight interleaved process pairs per model. Each
process used two warm-ups and three resident measured requests with greedy
decoding, seed 42, and at most 64 frames. The raw generation median pools the
24 measured requests; paired gains use the median request within each process.

| Model | Legacy generation median | Packed-reduce median | Raw gain | Paired median gain | Positive pairs | Raw wall gain | Paired wall gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6B Q8_0 | 333.5 ms | 326.5 ms | 2.10% | 4.70% | 6/8 | -0.43% | 2.57% |
| 1.7B Q8_0 | 555.0 ms | 550.5 ms | 0.81% | 4.26% | 5/8 | 2.43% | 2.24% |

The five correctness cases were `0.6B` sampled/greedy, `1.7B`
sampled/greedy/ICL. Generated-code JSON, decoder-code JSON, and WAV SHA-256
hashes matched the saved pre-change binary from `babc054` and the runtime legacy path in every
case. The CUDA regression suite passed with 11 passes and no failures, and a
separate non-CUDA Release build completed successfully. A CUDA-graphs-disabled
legacy/candidate diagnostic was also byte-identical. Raw artifacts are under
the ignored `benchmark_output/talker_frame_gather_sum` directory.

The first packed-table prototype retained the 15 sequential adds. Its four-pair
median was `+1.53%` on 0.6B and `-0.63%` on 1.7B, so it was superseded by the
packed reduction rather than retained unchanged.

A later independent 1.7B F32 ICL check against the official Python trajectory
found the missed numerical regression. With the same Python speaker embedding,
reference codes, text, and top-k-1 decode:

| C++ revision/path | Matching codes | Exact prefix | First difference |
|---|---:|---:|---|
| Historical `f06a0ba` | `1008/1008` (100%) | 63 frames | none |
| `e9c4a21`, `99ebc44`, `93867a5` | `1008/1008` (100%) | 63 frames | none |
| `fb9a9e0` packed reduction | `521/1008` (51.69%) | 24 frames | frame 24, codebook 13 |
| `fb9a9e0` with packed reduction disabled | `1008/1008` (100%) | 63 frames | none |
| Current restored sequential sum | `1008/1008` (100%) | 63 frames | none |

The packed reduction changed floating-point addition order. Its modest Q8
generation gain (raw `0.81-2.10%`, paired `4.26-4.70%`) does not justify an
autoregressive trajectory regression, so the packed tensor, graph path, and
runtime switch were removed rather than retained as an opt-in alternative.

The same controlled prompt was also checked outside the original 64-token
window. At 32 max tokens both historical and current behavior match Python for
all 31 common frames. At 96 max tokens both the exact historical `f06a0ba`
binary and the restored current binary produce the same `81.84%` match, with
73 exact prefix frames and the first difference at frame 73/codebook 6. The
long-run divergence therefore predates the performance commits; the historical
100% claim was correct for its documented 63-frame comparison, but must not be
generalized to arbitrary generation length.

A final five-request resident smoke after the removal measured `494 ms`
generation median for the restored path versus `503 ms` for the saved packed
binary on 1.7B Q8_0 (two warm-ups, 64-frame greedy request). This short probe
is informational, but it shows no measurable performance penalty that would
argue for keeping the incorrect reduction.

The broader restored-path 1.7B speaker-only matrix covered Q8_0, BF16, and F32
at maximum lengths 32, 64, and 96. Automatic and forced-legacy C++ outputs were
byte-identical in all nine cases, all first frames matched Python exactly, and
eight cases passed the token/audio/duration gates. The F32 96-frame case was
reported separately because Python stopped at 58 frames while C++ continued to
96 (`duration_ratio=1.66`); its common-prefix token match was `31.57%`. This is
an autoregressive stopping-trajectory difference, not a difference between the
optimized and legacy C++ paths. Raw reports are under the ignored
`benchmark_output/python_device_chain_validation/1p7_precision_matrix_restored`
directory.

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

### 2026-07-12: Asynchronous Device-Chain Replay

- Status: Rejected and removed.
- Hypothesis: enqueue the Code Predictor prefill and 14 replay steps without
  intermediate host waits, then synchronize and read the 15-token bridge once.
- Scope: automatic greedy 0.6B CUDA device-chain requests only; sampled, trace,
  replay-disabled, CPU, and 1.7B paths retained synchronous fallbacks.
- Correctness: automatic and synchronous outputs remained byte-identical across
  Q8_0/F16/F32 length matrices, sampled/fallback checks, the 0.6B F16 exact
  Python gate, and the historical 1.7B F32 ICL exact gate.
- Performance: across 20 independent process pairs at 64/96/128 frames, paired
  generation median was `+1.17%` (95% bootstrap CI `-0.19%` to `+2.77%`) and
  paired wall median was `+0.93%` (CI `-0.08%` to `+2.01%`). A refined 64-frame
  variant that queued the bridge D2H copy before the final wait reached only
  `+0.66%` raw wall median and `+1.05%` paired wall median (Wilcoxon `p=0.078`).
- Decision: the possible gain remained below the laptop-GPU noise floor and did
  not justify backend-split checks, asynchronous error handling, and altered
  timing semantics. Raw artifacts are under the ignored
  `benchmark_output/code_pred_async_chain` directory.

### 2026-07-12: Packed QKV in Code Predictor 2-Token Prefill

- Status: Rejected and removed.
- Hypothesis: reuse the accepted packed Code Predictor QKV tensor in prefill and
  replace three projections per layer with one, removing ten matmul nodes per
  audio frame.
- Correctness: generated codes, decoder codes, and WAVs were byte-identical in
  30 greedy cases covering both model sizes, every local Q8_0/F16/BF16/F32
  talker, and 8/32/64/96/128 frames. Twelve sampled cases covering all six
  models and two seeds were also byte-identical. Official-Python gates retained
  100% common-trajectory parity for 0.6B F16 at 32/64/96 frames and historical
  1.7B F32 ICL at 64 frames.
- Performance: the eight-pair 0.6B Q8_0 early gate used three warm-ups and eight
  resident measurements per process. Generation regressed from `507.5 ms` to
  `518.0 ms` raw median (`-2.07%`) with a `-2.19%` paired median and only 2/8
  positive pairs. Wall median regressed from `544.5 ms` to `552.0 ms`
  (`-1.38%`; paired `-1.99%`).
- Decision: the wider two-token quantized matmul selected a worse CUDA workload
  than the three established projections. Keep packed QKV limited to recurrent
  Code Predictor steps; no dtype-specific special path was retained. Raw
  artifacts are under the ignored
  `benchmark_output/code_pred_prefill_packed_qkv` directory.

| Experiment | Status | Measurement / reason |
|---|---|---|
| Asynchronous device-chain replay | Rejected | Byte-identical, but pooled 64/96/128 wall gain was only `+0.93%` with a confidence interval crossing zero; refined D2H chaining remained about 1% |
| Packed QKV in Code Predictor prefill | Rejected | Broad byte parity passed, but the primary 0.6B Q8_0 path regressed by about 2% generation and wall time |
| Pack Code Predictor FFN Gate+Up weights into one matmul | Rejected | Byte-identical, but 0.6B was effectively neutral and 1.7B regressed by roughly 1-2% in paired measurements |
| Standalone device-side greedy `argmax` | Rejected | Byte-identical, but approximately 1.9% slower on 0.6B and 5.6% slower on 1.7B; the extra kernel does not remove the synchronization by itself |
| Naive decoder graph pointer reuse after scheduler reset | Rejected | Invalid allocation lifetime caused a CUDA illegal-memory-access failure on the second decode |
| Dedicated decoder replay scheduler | Rejected | Correct and byte-identical, but neutral/slower offline; streaming graph rebuilds fell from 13 to 7 while decode time only changed from about 323 to 322 ms |
| Reuse streaming wrapper slice/output vectors | Rejected | Inspired by qwentts.cpp `d17c33d`; codes and WAVs were byte-identical at 5 and 19 chunks, but four paired 1.7B runs regressed by median 1.59% wall / 3.25% decode at 19 chunks and were wall-neutral (-0.32%) at 5 chunks. Prototype removed; persistent decoder position/mask/codebook buffers were already present. |
| Additional RMSNorm fusion patch | Inspected | GGML CUDA already fuses the relevant RMSNorm-to-Mul and RMSNorm-to-Mul-to-Add patterns |
| Packed Talker QKV on 0.6B | Rejected | Small paired regression; packed Talker QKV remains limited to models with `hidden_size > 1024` |
| Decoder pre-transformer Flash Attention | Rejected | Correct on fixed-code lengths 1, 32, 63, 64, 72, 128, and 252, but isolated CUDA medians were inconsistent: -5.0% at 63 frames, -0.5% at 64, +7.3% at 72, +2.0% at 128, and -2.1% at 252; opt-in prototype removed |
| Decoder Snake activation fusion | Inspected | The vendored CUDA graph optimizer already fuses the five-op `mul -> sin -> sqr -> mul -> add` pattern into `ggml_cuda_op_snake_fused()` when shapes and types match; no project-side operator is needed |

The Flash prototype used `QWEN3_TTS_DECODER_FLASH_ATTN=1`, converted the
existing F32 causal/window mask to the F16 format required by
`ggml_flash_attn_ext`, and passed the decoder reference and repeat-stability
checks. At 63 frames the legacy/candidate waveform comparison had maximum
absolute error `0.004707`, RMS error `0.000183`, and cosine correlation
`0.999561`; both paths remained within the decoder reference thresholds. Raw
logs and fixed-code length artifacts are under the ignored
`benchmark_output/decoder_flash_attention` directory. The candidate was still
removed because the launch and mask-conversion overhead produced no stable
end-to-end decoder win across the sliding-window boundary and long-input cases.

## Open Work Queue

### P2: Fully Device-Chained Greedy Code Predictor

Status: Implemented / automatic CUDA greedy supergraph

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
- Multiple replay graphs enqueued on one stream with device-resident bridges
  was tested on 2026-07-12 and rejected because its measured wall gain remained
  around 1% and below the noise floor.

Initial scope should be greedy only. Sampled top-k/top-p generation requires a
reproducible device sampler and is a separate project.

This is a high-risk change. Require byte-identical greedy codes, long
multi-request testing, both model sizes, CUDA-Graph on/off diagnostics, and a
clear win over the current host-sampled path.

The fully unrolled option was implemented on 2026-07-12. One graph now owns the
two-token Code Predictor prefill and all 14 recurrent steps. Each step writes
its KV rows explicitly, computes a device argmax into a persistent I32 token
bridge, and feeds that stored token into the next embedding lookup. The host
submits once and reads the 15 final codes once per audio frame. The graph has
`2533` nodes on 0.6B and `2593` on 1.7B, one scheduler split, and about `0.1 MiB`
of reported scheduler scratch. Construction and allocation were both below
`1 ms` in the measured runs.

Dispatch is automatic for `temperature=0` on CUDA for both 0.6B and 1.7B.
Sampling, debug tracing, CPU/non-CUDA backends, CoreML, and construction
failures before submission retain the established path. A compute failure
after submission fails the request instead of retrying against a mutated KV
cache. `QWEN3_TTS_CODE_PRED_SUPERGRAPH=0` is a diagnostic kill switch, not an
opt-in. A dedicated scheduler keeps CUDA-graph allocations stable across the
resident `greedy -> sampled -> greedy` mode transition.

Correctness validation covered Q8_0 0.6B and 1.7B at `32/64/96/128` requested
frames; every automatic-versus-disabled generated-code hash was byte-identical.
Seeded sampling stayed on its old path and was byte-identical with the switch
enabled or disabled. CUDA Graphs off, physical KV zeroing, and the resident
short/long/greedy/sampled switch test also passed. The reusable Python validator
now disables both optimized paths for its legacy leg and reports supergraph
dispatch explicitly. Historical exact gates remained at `100%`: 0.6B F16
top-k-1 for `32/64/96`, and 1.7B F32 ICL top-k-1 for `64`.

The robust 64-frame acceptance runs used two warm-ups and seven measured
resident requests per process. For 0.6B, eight interleaved process pairs had
`7/8` positive generation and wall results; paired median gains were `+3.10%`
generation and `+4.04%` wall. For 1.7B, all four exploratory pairs were
positive, so the gate was extended to eight pairs; all `8/8` were positive and
the final medians were `+3.46%` generation and `+3.73%` wall. The 95% lower
confidence bounds were positive for generation and wall on both model sizes
(`0.98%/1.02%` on 0.6B and `2.03%/1.95%` on 1.7B). The earlier
0.6B length probes at `32/96/128` frames also had positive medians, although
short-process measurements were noisier than the primary gate.

A 100-request resident memory run completed `102/102` warm-up plus measured
requests for both binaries. Relative to the 15-scheduler baseline, the
supergraph reduced peak working set by `53.9 MiB` and peak private memory by
about `1.16 GiB`; no growth or crash was observed. The complete Windows suite
finished with `11` pass, `0` fail, and `1` optional skip. Raw artifacts live
under the ignored `benchmark_output/supergraph-*` directories.

The older multi-graph device bridge described below remains as a tested
fallback and as historical context; the supergraph supersedes it in normal
greedy CUDA dispatch.

The first safe slice is now selected automatically for 0.6B CUDA requests with
`temperature=0` and at least 64 requested max frames. Each Code Predictor graph
writes its argmax token into a persistent 15-element I32 device bridge. The
next graph reads the previous slot directly, and the host reads the complete
codebook sequence once at the end. Replay graphs remain enabled; sampled
decoding, 1.7B, and CPU backends retain the legacy host-token bridge. The
environment setting `QWEN3_TTS_CODE_PRED_DEVICE_CHAIN=0` is retained only as a
diagnostic fallback for A/B tests; there is no opt-in path.

The initial 0.6B resident smoke used four interleaved process pairs, two
warm-ups, and four measured requests per process. Raw generation median
improved from `729.5 ms` to `710.0 ms` (`+2.67%`); paired process medians
improved by about `+1.03%` with two of four positive pairs. The 1.7B
exploratory three-pair smoke improved the raw median from `641 ms` to `637 ms`
(`+0.62%`), with two of three positive pairs. The graphs-disabled diagnostic
remained byte-identical but was slightly slower.

The length matrix explains the automatic threshold. With two process pairs,
three warm resident measurements, and the same greedy settings, the raw
generation medians were:

| Model | 8 max frames | 32 max frames | 64 max frames | 96 max frames | 128 max frames |
|---|---:|---:|---:|---:|---:|
| 0.6B | -5.08% | -2.30% | +6.82% | +5.11% | 0.00% |
| 1.7B | -11.89% | -2.13% | +1.37% | -1.46% | +2.82% |

Short requests therefore keep the legacy graph automatically. For 0.6B, the
device bridge is selected only from 64 requested frames onward; 1.7B remains
legacy because its measured gains were inconsistent. The initial automatic
dispatch matrix confirmed byte-identical outputs at every tested length for
both model sizes.

The final post-implementation validation used four interleaved process pairs,
two warm-ups, and five resident measurements per process at 64 requested
frames. The 0.6B paired median gain was `+3.25%` with three of four positive
pairs. The 1.7B result was effectively neutral at `+0.33%`, with two of four
positive pairs and high process variance; this is why automatic dispatch is
limited to the measured 1024-hidden-size profile. A 32-frame control confirmed
that automatic dispatch remains on the legacy graph.

Replay-graph reuse is keyed by the active token-bridge contract. A new resident
regression sequence exercises `32 -> 64 -> 32 -> 64` greedy requests followed
by `sampled -> greedy -> sampled` on one transformer instance. Repeated outputs
were byte-identical on 0.6B while crossing the bridge contract, and the same
sequence confirmed deterministic legacy fallback on 1.7B. This prevents a
graph built for the host-token contract from being replayed as a device-token
graph, or vice versa.

The final post-dispatch hash matrix covered 0.6B at 8, 32, 64, 96, and 128
requested frames plus 1.7B at 64 and 128. Every automatic-versus-legacy pair
produced byte-identical generated-code JSON, decoder-code JSON, and WAV SHA-256
hashes. The sampled fallback and the CUDA-graphs-off diagnostic also matched
the legacy path byte-for-byte. Raw artifacts are under the ignored
`benchmark_output/code_pred_device_chain` directory.

Official-Python parity is now part of the reusable validation workflow via
`scripts/validate_device_chain_python.ps1`. It loads the official 0.6B model in
float32, passes its speaker embedding to both C++ modes, and checks automatic
dispatch, byte identity between automatic and legacy generated codes/decoder
codes/WAVs, early and aggregate Python token agreement, sample rate, RMS, and
duration. Optional resident warm-up/run counts add generation medians to the
same JSON report; those single-process numbers remain informational and do not
replace interleaved process-pair acceptance benchmarks.

The first precision matrix covered Q8_0, F16, and F32 GGUF talkers at 32, 64,
and 96 requested frames. All nine automatic-versus-legacy comparisons were
byte-identical. Against official float32 Python, first-frame codebook agreement
was `68.75%` for Q8_0, `100%` for F16, and `87.5%` for F32. Aggregate agreement
over the tested lengths ranged from `7.60-10.69%`, `18.07-29.03%`, and
`9.71-11.90%` respectively; autoregressive divergence after near-tied logits is
therefore still expected even for F32. Every audio validity and duration gate
passed (`1.014-1.297` C++/Python duration ratio).

This matrix also caught a latent F16 correctness failure: removing the explicit
F32 cast of F16 FFN-down weights caused mostly-zero predictor codes and made
device argmax propagate an invalid token into a CUDA illegal-memory access.
The cast is now conditional on F16 weights, so Q8_0 and F32 keep their existing
native kernels while F16 restores finite logits, Python parity, and safe device
chaining.

The workflow's resident benchmark mode was smoke-tested on Q8_0 at 64 frames
with two warm-ups and five measurements per mode. Automatic generation median
was `671 ms` versus `681 ms` legacy (`+1.47%`); wall median was noisy and slower
(`748 ms` versus `733 ms`), so the result remains informational and the earlier
interleaved process-pair acceptance result remains authoritative.

### P2: Asynchronous Code Predictor I/O

Status: Partial / replay-chain variant rejected

Evaluate persistent/pinned host buffers plus asynchronous tensor upload,
compute, and readback. CPU sampling still creates a dependency between steps,
so first prove with a trace that asynchronous APIs remove a real synchronization
or overlap useful CPU work. Do not assume that replacing calls with `_async`
variants is itself an optimization. The greedy replay-chain experiment removed
the larger intermediate compute waits and still failed the acceptance gate, so
do not revisit smaller I/O-only changes without new profiling evidence or an
upstream backend change.

### P2: Stateful Streaming Decoder and Pipeline Overlap

Status: Open

The tested replay scheduler did not materially improve performance. A larger
design would keep persistent decoder state per common chunk width and overlap:

Profiling after the qwentts.cpp `d17c33d` scratch-buffer comparison confirms
that host allocation is not the limiting factor here. With a `0.25 s` chunk,
the current overlap decoder consumed `532` input/context frames to emit `57`
new frames across `19` chunks; `442/447 ms` was graph compute while graph
building was only `3 ms`. At `1.0 s`, it consumed `182` input/context frames
for the same `57` emitted frames and spent `230/234 ms` in compute. The next
prototype must therefore preserve transformer KV and causal convolution state
between chunks instead of merely caching graphs or host vectors.

```text
generate chunk N+1        decode chunk N
        GPU/stream A  ||  GPU/stream B or bounded worker queue
```

This requires explicit backpressure, bounded memory, deterministic chunk
ordering, final-tail handling, and comparison against non-streaming audio. Its
correctness gate must cover chunk sizes on both sides of the decoder sliding
window, byte-stable repeated runs where feasible, waveform error/correlation
against the current overlap implementation, and reset/reuse across resident
requests.

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

1. Design stateful streaming decoder overlap.
2. Treat continuous batching and mixed quantization as separate milestones.
3. Promote the exact Python parity fixtures and broader 1.7B voice coverage.
