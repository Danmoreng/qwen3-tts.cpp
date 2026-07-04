# Python Parity Debug Plan

Last updated: 2026-07-04

## Purpose

Use Python parity as a focused debugging tool for `qwen3-tts.cpp`.

The goal is not to promise bit-identical WAV output between PyTorch and GGML across full generations. Small numerical differences, backend kernels, BF16 rounding, sampling, and autoregressive accumulation make full end-to-end identity fragile.

The goal is narrower and more useful:

- Match prompt construction inputs where practical.
- Match the first generated codebook-0 token.
- Find and fix the first deterministic divergence in the code predictor.
- Use top-token parity for local steps as the main correctness signal.

The current highest-value target is the first divergent step after the corrected
speaker-only trace setup:

- Frame: `9`
- Codebook: `6` (code-predictor step `5`)
- Area: late-frame code-predictor numerical drift / near-tie

## Current Findings

| Area | Finding | Status |
|---|---|---|
| Python prompt extraction | Faster Qwen3 TTS matches Python prompt codes exactly when extra silence appending is disabled. | Verified |
| audio.cpp prompt extraction | Reference-code values diverge from Python despite matching frame count. | Verified divergent |
| BF16 GGUF conversion | A mostly-BF16 GGUF can be built directly from the BF16 HuggingFace checkpoint. `scripts/convert_tts_to_gguf.py` also supports repeatable `--keep-f32-regex` overrides for targeted parity model variants without runtime casts. | Implemented locally |
| Source checkpoint precision | The local `Qwen3-TTS-12Hz-1.7B-Base/model.safetensors` checkpoint stores all `480/480` tensors as BF16, including code-predictor layer-4 MLP weights. `scripts/inspect_safetensors_dtypes.py` makes this check repeatable. Targeted F32 GGUF storage cannot recover precision that is not present in the source checkpoint. | Verified |
| Speaker-only BF16 generation parity | With Python speaker embedding, corrected non-streaming/base prompt layout, `do_sample=True`, and `top_k=1`, frames `0..8` match exactly. First divergence is frame `9`, codebook `6`: Python token `517`, C++ token `9`. | Late drift |
| Python CPU BF16 trace | CPU BF16 PyTorch is not a reliable parity proxy for the current CUDA/GGUF path: it matches only `94/160` tokens against C++ over the 10-frame fixture and first diverges at frame `0`, codebook `14`. | Diagnostic only |
| Python CUDA BF16 trace | CUDA BF16 PyTorch is also not a stronger top-token oracle for the current GGUF path. Speaker-only matches `94/160` tokens and first diverges at frame `0`, codebook `14`; ICL matches `3/16` tokens and first diverges at frame `0`, codebook `3`. Both first divergences are BF16 top-logit ties in Python. | Diagnostic only |
| ICL BF16 generation parity | The old frame `0`, codebook `1` divergence was caused by an ICL prompt-layout mismatch. C++ now uses the Python non-streaming ICL layout and trims `--reference-text-file` outer whitespace. First local F32-vs-GGUF-BF16 drift is frame `0`, codebook `8`: Python token `499`, C++ token `1481`. | Late-step near-tie |
| Greedy path | Greedy currently produces invalid/repetitive output and should not be used as the main parity gate until fixed. | Known failing |
| No-debug performance guard | Alternating clean-baseline/current CUDA timing run showed no regression after debug-only trace hooks: current median `Total generate` was `872.1 ms` vs baseline `882.2 ms` for the 64-token speaker-only benchmark. | Verified |
| Trace tooling | `scripts/dump_python_trace.py` can dump external speaker-embedding traces, raw per-step code-predictor logits when supported by Transformers, and debug-only code-predictor per-layer/sub-layer tensors at AR steps. `scripts/debug_trace_report.py` labels raw vs post-warp logits. `scripts/parity_trace_summary.py` emits compact JSON first-diff summaries, near-tie margins and classifications, first-diff step trajectories with aggregate metrics, boundary tensor comparisons, code-predictor layer/sub-layer comparisons, and optional expectation checks. `scripts/run_speaker_parity_fixture.ps1` regenerates speaker-only and ICL fixtures. | Improved |
| Benchmark tooling | `scripts/benchmark_parity_smoke.ps1` runs the standard speaker-embedding parity timing smoke, captures logs, parses warm medians, records before/after `nvidia-smi` snapshots when available, and can compare against a saved baseline summary with regression thresholds. | Added |

## Guiding Rules

- Prefer deterministic `top_k=1`, `temperature=1.0`, `top_p=1.0` over the current greedy special case.
- Compare speech-code tokens before comparing audio.
- Compare logits and hidden tensors at the first divergent step before inspecting later frames.
- Use Python-provided speaker embeddings and Python-provided reference codes when testing transformer parity, so encoder differences do not contaminate the result.
- Treat full-generation parity as a stretch goal. Treat first-step top-token parity as the actionable target.

## Phase 1: Reproducible Parity Fixtures

Create or formalize small fixtures under `benchmark_output/python_parity/`:

- A short target text for speaker-only synthesis.
- A short target text for ICL synthesis.
- A reference transcript.
- A reference WAV.
- Python speaker embedding dump.
- Python reference speech-code dump.
- Python generated-code dump.
- C++ generated-code dump.

Success criteria:

- Running the fixture scripts produces a JSON report with shapes, match count, match percent, and first divergent token.
- Speaker-only and ICL reports can be regenerated without manual steps.
- The fixture uses the same text tokens, language, speaker embedding, and reference codes for Python and C++.

Speaker-only regeneration command:

```powershell
.\scripts\run_speaker_parity_fixture.ps1 `
  -PythonPath C:\Development\Qwen3TTSDev\Qwen3-TTS `
  -OutputDir benchmark_output\python_parity\speaker_fixture_current `
  -RequireAssets
```

Use `-PythonModel`, `-CppModelDir`, `-SpeakerEmbedding`, and `-CliExe` to
override local paths when the default assets are not present. Without
`-RequireAssets`, the script skips cleanly when large local model files are
missing.

ICL regeneration command:

```powershell
.\scripts\run_speaker_parity_fixture.ps1 `
  -PythonPath C:\Development\Qwen3TTSDev\Qwen3-TTS `
  -OutputDir benchmark_output\python_parity\icl_fixture_current `
  -Text "This is a short parity check for ICL voice cloning." `
  -MaxTokens 4 `
  -MaxFrames 1 `
  -ReferenceTextFile benchmark_output\parity_serveurperso_seed\ref.txt `
  -ReferenceCodes benchmark_output\python_parity\python_reference_codes.json `
  -RequireAssets
```

Local expectation gate for both fixtures:

```powershell
.\scripts\run_all_tests.ps1 `
  -ParityFixturesOnly `
  -BuildDir build-timing-current `
  -OutputDir test_output\python_parity_gate
```

`-ParityFixturesOnly` implies `-RequireParityFixtures` and avoids failing on
unrelated default model filenames from the broader regression suite. Expected
match percentages and first-diff locations are checked from
`tests/fixtures/python_parity_expectations.json`; the large trace/model assets
remain external.

Latest gate result:

- `.\scripts\run_all_tests.ps1 -ParityFixturesOnly -BuildDir build-timing-current -OutputDir test_output\python_parity_gate`
- Passed speaker-only and ICL fixtures (`PASS: 2`, `FAIL: 0`, `SKIP: 4`)

Local performance smoke command:

```powershell
.\scripts\benchmark_parity_smoke.ps1 `
  -OutputDir benchmark_output\perf_parity_smoke_current `
  -Repeat 4 `
  -RequireAssets
```

Baseline comparison command:

```powershell
.\scripts\benchmark_parity_smoke.ps1 `
  -OutputDir benchmark_output\perf_parity_smoke_current `
  -Repeat 4 `
  -RequireAssets `
  -BaselineSummary benchmark_output\perf_parity_smoke_baseline\summary.json `
  -MaxGpuUtilizationBeforePercent 20 `
  -WaitForGpuIdleSeconds 120 `
  -MaxGenerateRegressionPercent 5 `
  -MaxPipelineRegressionPercent 5 `
  -MaxRtfRegressionPercent 5
```

Use `-CliExe`, `-ModelDir`, and `-SpeakerEmbedding` to override local assets.
The script writes `summary.json` with warm medians, per-repeat details, and GPU
snapshots. When `-BaselineSummary` is provided, `summary.json` also includes
per-metric deltas for generate, talker, code predictor, pipeline total, and RTF;
the script exits non-zero if any enabled threshold is exceeded. Use
`-MaxGpuUtilizationBeforePercent` to guard against another process saturating
the GPU before inference; add `-WaitForGpuIdleSeconds` when the benchmark should
poll briefly for the GPU to become idle before writing a skipped summary.

## Phase 2: BF16 Model Path

Keep BF16 conversion available because Python CUDA commonly runs the model in BF16.

Required implementation:

- `scripts/convert_tts_to_gguf.py` supports `--type bf16`.
- `scripts/convert_tokenizer_to_gguf.py` supports `--type bf16`.
- BF16 tensors are written as `GGML_TYPE_BF16`, not NumPy `float16`.
- 1D tensors and precision-sensitive tensors remain F32 where current policy requires it.
- Use `scripts/inspect_safetensors_dtypes.py` to verify source checkpoint dtype
  before trying targeted F32 GGUF storage variants.

Success criteria:

- GGUF inspection shows a mostly-BF16 model with expected F32 exceptions.
- The C++ loader can load and run the BF16 model.
- The BF16 path reproduces the same first-divergence location as the current parity fixture.

Source dtype inspection command:

```powershell
python .\scripts\inspect_safetensors_dtypes.py `
  C:\Development\Qwen3TTSDev\audio.cpp\models\Qwen3-TTS-12Hz-1.7B-Base `
  --name-regex 'talker\.code_predictor\.model\.layers\.4\.mlp\.(down|gate|up)_proj\.weight$' `
  --expect-all-dtype BF16 `
  --expect-matched-dtype BF16
```

Current local result: `480/480` tensors are BF16; the three matched layer-4 MLP
weights are BF16.

## Phase 3: First Divergence Trace

Instrument or reuse debug trace dumps for the first divergent code-predictor step.

For the original frame `0`, codebook `1` failure, dump and compare:

- Talker hidden state passed into the code predictor.
- Codebook-0 token.
- Codebook-0 embedding.
- Code predictor prefill input tensor.
- Code predictor attention mask.
- Code predictor position IDs / M-RoPE IDs.
- Code predictor first-step logits before sampling/top-k filtering.
- Selected top token.

Success criteria:

- A report identifies the first tensor where Python and C++ materially diverge.
- If hidden state and CB0 embedding match but logits diverge, the bug is inside the code predictor graph/runtime.
- If CB0 embedding differs, investigate embedding lookup/layout.
- If hidden state differs, move the trace boundary backward into talker prefill/step.

Current result:

- Frame `0`, all codebooks now match in the corrected speaker-only fixture.
- The corrected trace shows matching input tokens, speaker embedding, prefill embedding, CB0 tokens, code-predictor prefill inputs, positions, masks, and frame `0` code-predictor tokens.
- This phase is considered complete for the speaker-only path; keep it available for ICL verification.
- For the ICL path, C++ and Python now agree on the non-streaming prefill boundary:
  - `prefill_len=141`, `trailing_len=1`
  - `prefill_embd` cosine `1.000000`, max absolute difference `0.001052`
  - frame `0` CB0 token matches

## Phase 4: Code Predictor Audit

Audit the code predictor implementation against Python for the failing step.

Checklist:

- `small_to_mtp` projection shape, transpose, bias, and dtype handling.
- Codebook embedding index and per-codebook embedding selection.
- Code predictor prefill sequence ordering.
- Code predictor KV-cache initialization and update.
- Code predictor attention mask layout and sign.
- Code predictor RoPE / M-RoPE position layout.
- RMSNorm epsilon and accumulation dtype.
- F16/BF16/F32 casts around `mul_mat`, especially `ffn_down`.
- LM head selection for codebook `1`.
- Suppression/repetition-penalty rules are not accidentally applied to code predictor logits.

Success criteria:

- Frame `0`, codebook `1` top-token matches Python in speaker-only mode.
- The same step also matches in ICL mode when using Python reference codes.

## Phase 5: Expand Local Parity

After the first divergent step is fixed, expand gradually:

- Frame `0`, all codebooks.
- Frames `0..3`, all codebooks.
- A short 32-frame speaker-only generation.
- A short 32-frame ICL generation.

Success criteria:

- Top-token parity holds for at least frame `0`.
- Later divergences are documented with exact first divergence locations.
- If divergence appears only after many autoregressive steps, classify it separately from first-step implementation bugs.

Current result:

- Speaker-only frames `0..8` match all 16 codebooks.
- Across the 10-frame speaker-only fixture, Python F32 CPU vs C++ GGUF BF16 matches `150/160` tokens (`93.75%`).
- Frame `9` matches through code-predictor step `4`; step `5` flips between close logits:
  - Python: token `517` at `22.143404`
  - C++: token `9` at `22.144558`, token `517` at `22.126083`
  - Step cosine: `0.99999982`, max absolute logit difference: `0.044123`
  - Python top-1 margin: `0.005629`; C++ top-1 margin: `0.018475`
  - Python winner token `517` ranks second in C++; C++ winner token `9` ranks second in Python
  - At this boundary, CB0 embedding, position IDs, and mask match exactly; talker hidden/input-hidden cosine is `0.99999581`, and code-predictor projected-input cosine is `0.99999800`
- Layer-level debug traces for the same frame `9`, step `5` show the AR step input is identical and the remaining drift is inside the code-predictor decoder:
  - step projected input: cosine `1.000000000`, max absolute difference `0.000000`
  - layer `0`: cosine `0.999999202`, max absolute difference `0.000579`
  - layer `1`: cosine `0.999999028`, max absolute difference `0.001333`
  - layer `2`: cosine `0.999998624`, max absolute difference `0.001819`
  - layer `3`: cosine `0.999998167`, max absolute difference `0.001705`
  - layer `4`: cosine `0.999997618`, max absolute difference `0.017359`
  - final normalized hidden: cosine `0.999997543`, max absolute difference `0.037262`
- Sub-layer traces refine this further for layer `4`:
  - attention norm input to self-attention: max absolute difference `0.010942`
  - self-attention output projection: max absolute difference `0.002479`
  - post-attention FFN norm: max absolute difference `0.016222`
  - MLP/FFN output: max absolute difference `0.019901`
  - layer residual output: max absolute difference `0.017359`
  - This points at accumulated residual/MLP numerical drift rather than a large attention-output mismatch.
- First-diff step trajectory for codebook `6`:
  - Frames `0..8` have matching top tokens at the same codebook/step.
  - The smallest earlier Python top-1 margin is frame `6` at `0.108820`; the frame `9` Python margin collapses to `0.005629`.
  - The largest earlier logit max-absolute difference at this step is frame `8` at `0.046885`, comparable to frame `9` at `0.044123`; the flip happens because the local top margin becomes tiny, not because frame `9` has uniquely large logit error.
  - The aggregate trajectory summary reports the frame `9` Python top-1 margin is `0.0517x` the smallest prior Python margin, while the frame `9` max-absolute logit difference is `0.9411x` the prior maximum.
- After the step `5` flip, subsequent codebooks in frame `9` diverge autoregressively.
- Python CPU BF16 is much less aligned than Python CPU F32 for this local setup, so use CPU F32 traces for structural debugging unless a CUDA BF16 Python trace is available.
- Python CUDA BF16 was checked after the GPU became idle and is not a better
  top-token parity oracle than CPU F32:
  - Speaker-only CUDA BF16 vs C++ matched `94/160` tokens and first diverged at
    frame `0`, codebook `14`: Python token `1213` vs C++ token `812`.
    Python's top logits for tokens `1213` and `812` were exactly tied at
    `29.375`, while C++ favored token `812` by `0.022070`.
  - ICL CUDA BF16 vs C++ matched `3/16` tokens and first diverged at frame `0`,
    codebook `3`: Python token `483` vs C++ token `1151`. Python's top logits
    for tokens `1151` and `483` were exactly tied at `18.25`, while C++ favored
    token `1151` by `0.071327`.
  - These traces are useful evidence that BF16 quantization creates hard
    near-tie/tie cases, but the CPU F32 trace remains the cleaner structural
    debugging reference.
- ICL frame `0` now matches through codebook `7`; codebook `8` flips between a tight top-3:
  - Python: token `499` at `21.591181`
  - C++: token `1481` at `21.595354`, token `499` at `21.593187`
  - Step cosine: `0.99999985`, max absolute logit difference: `0.036032`
  - Layer-level debug traces for the same step `7` show identical projected input, then the largest hidden drift appears after the final decoder layer and final norm:
    - step projected input: cosine `1.000000000`, max absolute difference `0.000000`
    - layer `4`: cosine `0.999996703`, max absolute difference `0.025833`
    - final normalized hidden: cosine `0.999997219`, max absolute difference `0.036705`
  - Sub-layer traces for ICL layer `4` mirror the speaker-only pattern:
    - attention norm input to self-attention: max absolute difference `0.018688`
    - self-attention output projection: max absolute difference `0.002907`
    - post-attention FFN norm: max absolute difference `0.022301`
    - MLP/FFN output: max absolute difference `0.025537`
    - layer residual output: max absolute difference `0.025833`

## Phase 6: Regression Gates

Add lightweight regression checks once a parity bug is fixed.

Candidate gates:

- A code-predictor first-step top-token test using checked-in small metadata and external/generated binary fixtures.
- A script-level parity check that is skipped unless required model artifacts are present.
- `scripts/test_parity_trace_summary_smoke.py` is a CI-safe synthetic trace
  smoke that validates summary parsing, tensor shapes, first-diff token IDs,
  near-tie classification, boundary/code-predictor tensor comparisons, and
  negative category expectation handling without large model artifacts.
- `scripts/test_debug_trace_report_smoke.py` is a synthetic smoke for
  `scripts/debug_trace_report.py`; it checks raw/post-warp labels and compare
  metrics in the human-readable trace report.
- `scripts/test_inspect_safetensors_dtypes_smoke.py` is a tiny synthetic
  safetensors smoke for `scripts/inspect_safetensors_dtypes.py`; it verifies
  positive BF16 expectations and negative expectation handling without the real
  checkpoint.
- `scripts/test_python_parity_expectations_smoke.py` validates the checked-in
  expectation metadata schema, required fields, category values, and basic
  numeric ranges without running full model fixtures; it also checks that a
  deliberately invalid metadata payload is rejected.
- `tests/fixtures/python_parity_expectations.json` stores the small checked-in expected first-diff metadata and first-diff classification for local full-model parity fixtures.
- `scripts/parity_trace_summary.py` is the local JSON-reporting primitive for first-diff gates and supports expected match percentage, first-diff token, cosine, and max-absolute thresholds.
- `scripts/parity_trace_summary.py` also emits `first_diff_classification`
  with `exact_tie`, `near_tie_token_swap`, `near_tie`, `token_swap`, or
  `logit_drift` categories. Defaults are `--near-tie-margin 0.02` and
  `--near-tie-rank-threshold 2`; local gates can enforce it with
  `--expect-first-diff-category` and
  `--expect-first-diff-max-abs-over-margin-at-least`.
- `scripts/benchmark_parity_smoke.ps1` is the local JSON-reporting primitive for repeat timing smokes and should be used before/after C++ hot-path parity experiments. Use `-BaselineSummary` plus `-MaxGenerateRegressionPercent`, `-MaxPipelineRegressionPercent`, and `-MaxRtfRegressionPercent` when a saved baseline is available.
  `-SelfTest` validates timing-log parsing, median calculation, and regression
  threshold handling without requiring model artifacts. The summary also
  reports warm-run min/max/range percentages and warns when fewer than
  `-MinWarmRuns` warm samples are available; prefer at least 4 total repeats so
  the default 3 warm samples are present after dropping the cold first run. Use
  `-MaxWarmGenerateRangePercent`, `-MaxWarmCodePredRangePercent`,
  `-MaxWarmPipelineRangePercent`, and `-MaxWarmRtfRangePercent` to fail noisy
  benchmark samples separately from baseline regressions. Baseline comparisons
  record workload/sampling compatibility diagnostics; add
  `-RequireComparableBaseline` to fail when the saved baseline does not match
  the current model, speaker embedding, text, language, token limit, sampling
  parameters, or seed.
- `scripts/inspect_safetensors_dtypes.py` is the local source-checkpoint dtype
  verifier for deciding whether targeted F32 GGUF storage experiments can be
  meaningful.
- `scripts/run_speaker_parity_fixture.ps1` is the current speaker-only and ICL fixture regeneration command. It writes `fixture_metadata.json` beside each generated trace summary so fixture outputs record the resolved model paths, prompt inputs, sampling settings, and expectation gates used for that run. `run_all_tests.ps1 -ParityFixturesOnly` validates those metadata sidecars after each full fixture run.
- `scripts/run_all_tests.ps1 -ParityFixturesOnly` runs both parity fixtures as a required local gate when the large Python/C++ model assets are present.
- `scripts/convert_tts_to_gguf.py --keep-f32-regex` can produce targeted model variants for parity experiments without broad runtime casting. Example candidate:

```powershell
python .\scripts\convert_tts_to_gguf.py `
  --input C:\Development\Qwen3TTSDev\audio.cpp\models\Qwen3-TTS-12Hz-1.7B-Base `
  --output benchmark_output\bf16_codepred_l4_ffn_down_f32\qwen-talker.gguf `
  --type bf16 `
  --keep-f32-regex '^code_pred\.blk\.4\.ffn_down\.weight$'
```

Success criteria:

- Normal contributors can run the lightweight tests without downloading large models.
- Full parity fixtures remain available for local debugging when model artifacts are present.
- The test failure message reports the first divergent frame/codebook/token.

## Non-Goals

- Exact full-WAV parity across PyTorch and GGML.
- Exact logit equality to the last bit.
- Publishing Python parity as a user-facing quality guarantee.
- Blocking performance work on late-frame numerical drift once first-step parity is fixed.

## Open Questions

- Should BF16 conversion remain a documented user-facing model format, or only a debug/developer format?
- Should the greedy path be fixed before or after code-predictor parity?
- Should parity fixtures live only in `benchmark_output/`, or should a small reproducible subset move under `tests/fixtures/`?
- Can we create a tiny synthetic code-predictor fixture that avoids loading the full 1.7B model?

## Recommended Next Step

The frame `9`, codebook `6` speaker-only divergence is now classified and gated
as a late `near_tie_token_swap`, and the ICL first divergence is gated as
`near_tie`. Keep the current CPU F32 trace as the structural reference and use
the parity fixtures as regression gates while pursuing any future hot-path
changes.

Do not add broad F32 casts as a parity fix unless a targeted benchmark shows the
tradeoff is acceptable. The current evidence points to a late near-tie caused by
small BF16/GGUF vs PyTorch F32 numerical differences, not a frame-0 structural
implementation bug.

Next practical step: do not pursue F32 GGUF storage variants from the current
local HF checkpoint. It is entirely BF16, so those variants cannot recover
source precision and the single `code_pred.blk.4.ffn_down.weight` test produced
identical parity summaries. CUDA BF16 Python traces were checked and are less
useful as top-token gates because they introduce frame-0 BF16 ties. The
remaining useful options are a true F32-source checkpoint experiment if such a
checkpoint exists, or accepting the current late-frame near-ties as
BF16/PyTorch-vs-GGML numerical drift.

Latest speaker performance smoke after adding the fixture runner was
current-only, no-debug, 8 process runs with the same 64-token speaker prompt.
Warm median (`runs 2..7`) was `930.7 ms` total generate, `311.55 ms` talker,
`466.65 ms` code predictor, RTF `0.257`. That was noisier and slower than the
previous alternating clean-baseline/current result.

Latest speaker performance smoke after adding the parity-only gate was
current-only, no-debug, 8 in-process repeats with the same speaker prompt and
BF16 parity model. Warm median (`runs 2..8`) was `920.3 ms` total generate,
`346.1 ms` talker, `509.5 ms` code predictor, `945.0 ms` pipeline total, and
RTF `0.241`. This is a current-only smoke; the stronger regression signal is
still the same-session clean-baseline/current benchmark below.

Latest speaker performance smoke after moving parity expectations into
`tests/fixtures/python_parity_expectations.json` was current-only, no-debug, 4
in-process repeats. Warm median (`runs 2..4`) was `915.4 ms` total generate,
`345.7 ms` talker, `511.1 ms` code predictor, `943.0 ms` pipeline total, and
RTF `0.241`. This was a script/test metadata change only.

Latest speaker performance smoke after adding near-tie diagnostics to
`scripts/parity_trace_summary.py` was current-only, no-debug, 4 in-process
repeats. Warm median (`runs 2..4`) was `931.2 ms` total generate, `353.2 ms`
talker, `524.3 ms` code predictor, `957.0 ms` pipeline total, and RTF `0.244`.
This was a Python reporting change only; no inference hot path changed.

Latest speaker performance smoke after adding the first-diff step trajectory to
`scripts/parity_trace_summary.py` was current-only, no-debug, 4 in-process
repeats. Warm median (`runs 2..4`) was `917.4 ms` total generate, `349.6 ms`
talker, `513.6 ms` code predictor, `943.0 ms` pipeline total, and RTF `0.241`.
This was also a Python reporting change only.

Latest speaker performance smoke after adding aggregate trajectory metrics was
current-only, no-debug, 4 in-process repeats, and was anomalously slower despite
this being another Python reporting-only change. First run warm median
(`runs 2..4`) was `1091.3 ms` total generate, `395.9 ms` talker, `636.1 ms`
code predictor, `1121.0 ms` pipeline total, and RTF `0.286`. A rerun remained
slow: `1062.8 ms` total generate, `389.7 ms` talker, `616.9 ms` code predictor,
`1100.0 ms` pipeline total, and RTF `0.281`. Treat this as an environment
warning, not a regression caused by the reporting patch; no C++ inference path
changed in this step.

Latest speaker performance smoke with `scripts/benchmark_parity_smoke.ps1`
verified the reusable wrapper and landed closer to the previous band: 3
in-process repeats, warm median (`runs 2..3`) `953.6 ms` total generate,
`357.3 ms` talker, `536.9 ms` code predictor, `1015.0 ms` pipeline total, and
RTF `0.2585`. This was a benchmark-script addition only.

Latest speaker performance smoke after adding baseline-summary comparison to
`scripts/benchmark_parity_smoke.ps1` used the prior wrapper summary as the
baseline with loose `50%` thresholds to verify the comparison path. The 3-repeat
current run had warm median (`runs 2..3`) `948.85 ms` total generate,
`349.95 ms` talker, `524.8 ms` code predictor, `995.5 ms` pipeline total, and
RTF `0.2535`. Compared with the saved wrapper baseline, generate was `-0.50%`,
pipeline total was `-1.92%`, and RTF was `-1.93%`; no regression threshold
failed.

Latest speaker performance smoke after adding debug-only code-predictor
per-layer trace dumps was no-debug and compared against the same saved wrapper
baseline with loose `50%` thresholds. The 3-repeat current run had warm median
(`runs 2..3`) `834.3 ms` total generate, `312.5 ms` talker, `467.05 ms` code
predictor, `864.0 ms` pipeline total, and RTF `0.2205`. Compared with the saved
wrapper baseline, generate was `-12.51%`, pipeline total was `-14.88%`, and RTF
was `-14.70%`; no regression threshold failed. This supports that the new
diagnostics are gated behind debug tracing and do not affect normal no-debug
performance.

Latest speaker performance smoke after adding debug-only code-predictor
sub-layer trace dumps was also no-debug and compared against the same saved
wrapper baseline with loose `50%` thresholds. The 3-repeat current run had warm
median (`runs 2..3`) `962.2 ms` total generate, `359.85 ms` talker, `549.05 ms`
code predictor, `1007.5 ms` pipeline total, and RTF `0.257`. Compared with the
saved wrapper baseline, generate was `+0.90%`, code predictor was `+2.26%`,
pipeline total was `-0.74%`, and RTF was `-0.58%`; no regression threshold
failed. Treat this as normal desktop benchmark noise; no-debug performance
remains effectively unchanged by the debug-only dumps.

Targeted BF16 variant experiment:

- Built `benchmark_output\bf16_codepred_l4_ffn_down_f32` with
  `--keep-f32-regex '^code_pred\.blk\.4\.ffn_down\.weight$'`.
- Converter confirmed `code_pred.blk.4.ffn_down.weight` was stored as F32.
- Follow-up safetensors metadata inspection showed the source HF checkpoint is
  entirely BF16 (`480/480` tensors), including
  `talker.code_predictor.model.layers.4.mlp.down_proj.weight`,
  `gate_proj.weight`, and `up_proj.weight`.
- Speaker-only parity was unchanged: `93.75%` token match, first diff remained
  frame `9`, codebook `6`, Python token `517` vs C++ token `9`, with the same
  logit cosine `0.999999775` and max absolute difference `0.044123`.
- ICL parity was unchanged: `56.25%` token match, first diff remained frame
  `0`, codebook `8`, Python token `499` vs C++ token `1481`, with the same
  logit cosine `0.999999849` and max absolute difference `0.036032`.
- Timing was attempted, but the GPU was already saturated by another process
  (`Trackmania.exe`, `98%` utilization in `nvidia-smi`). Both the targeted
  variant and the original BF16 baseline slowed to about `7.3 s` warm median
  total generate, so the timing sample is contaminated and not evidence of a
  model-specific regression.
- `scripts/benchmark_parity_smoke.ps1 -MaxGpuUtilizationBeforePercent` now
  catches this case before inference; add `-WaitForGpuIdleSeconds` to wait for a
  quiet GPU before skipping. A test run with threshold `50` skipped because
  pre-run GPU utilization stayed at `98%`.
- Wait-guard validation with threshold `50`, wait `2s`, poll `1s` saw the GPU
  settle to `0%` and ran a 3-repeat smoke: warm generate median `1108.05 ms`,
  pipeline median `1141.0 ms`, RTF median `0.291`.
- Follow-up no-debug timing after the CUDA BF16 trace runs used the idle-GPU
  guard with threshold `20`, wait `60s`, and 3 repeats. Warm generate median was
  `919.5 ms`, code predictor `523.35 ms`, pipeline `950.0 ms`, RTF `0.2425`;
  pre-run GPU utilization was `0%`.
- After adding structured first-diff classification, existing traces classify
  as expected: speaker F32 is `near_tie_token_swap`, ICL F32 is `near_tie`,
  and both CUDA BF16 traces are `exact_tie`.
- Follow-up no-debug timing for the classification-only reporting change used
  the idle-GPU guard with threshold `20`, wait `60s`, and 3 repeats. Warm
  generate median was `982.25 ms`, code predictor `566.0 ms`, pipeline
  `1015.5 ms`, RTF `0.259`; pre-run GPU utilization was `0%`.
- After wiring first-diff classifications into the local parity gate,
  `run_all_tests.ps1 -ParityFixturesOnly` passed speaker-only and ICL fixtures
  with category expectations enforced. Follow-up no-debug timing used the same
  idle-GPU guard and 3 repeats: warm generate median `972.25 ms`, code
  predictor `547.8 ms`, pipeline `1010.5 ms`, RTF `0.258`; pre-run GPU
  utilization was `0%`.
- After adding the synthetic CI-safe trace-summary smoke,
  `run_all_tests.ps1 -ParityFixturesOnly` passed the model-free smoke plus the
  speaker-only and ICL full fixtures (`PASS: 3`, `FAIL: 0`, `SKIP: 4`).
  Follow-up no-debug timing used the idle-GPU guard and 3 repeats: warm
  generate median `974.15 ms`, code predictor `556.8 ms`, pipeline `1012.0 ms`,
  RTF `0.2585`; pre-run GPU utilization was `0%`.
- The synthetic smoke now also asserts boundary tensor and code-predictor layer
  tensor comparison metrics, covering the report sections used to localize the
  late-frame drift. `run_all_tests.ps1 -ParityFixturesOnly` passed with
  `PASS: 3`, `FAIL: 0`, `SKIP: 4`. Follow-up no-debug timing used the
  idle-GPU guard and 3 repeats: warm generate median `956.4 ms`, code predictor
  `540.55 ms`, pipeline `999.5 ms`, RTF `0.255`; pre-run GPU utilization was
  `0%`.
- The parity gate now also asserts that first-diff logit max-absolute drift is
  at least as large as the smallest local top-1 margin, making the near-tie
  classification numeric rather than just categorical.
  `run_all_tests.ps1 -ParityFixturesOnly` passed with `PASS: 3`, `FAIL: 0`,
  `SKIP: 4`. Follow-up no-debug timing used the idle-GPU guard and 3 repeats:
  warm generate median `980.7 ms`, code predictor `559.95 ms`, pipeline
  `1027.0 ms`, RTF `0.262`; pre-run GPU utilization was `0%`.
- A synthetic safetensors dtype smoke now covers the helper used to verify that
  the local source checkpoint is BF16 before targeted F32-storage experiments.
  `run_all_tests.ps1 -ParityFixturesOnly` passed with helper smokes plus both
  full fixtures (`PASS: 4`, `FAIL: 0`, `SKIP: 4`). Follow-up no-debug timing
  used the idle-GPU guard and 3 repeats: warm generate median `853.95 ms`, code
  predictor `481.2 ms`, pipeline `883.0 ms`, RTF `0.225`; pre-run GPU
  utilization was `0%`.
- A synthetic debug trace report smoke now covers the human-readable raw vs
  post-warp diagnostic helper that is useful when inspecting trace dumps by
  hand. `run_all_tests.ps1 -ParityFixturesOnly` passed with helper smokes plus
  both full fixtures (`PASS: 5`, `FAIL: 0`, `SKIP: 4`). Follow-up no-debug
  timing used the idle-GPU guard and 3 repeats: warm generate median
  `864.75 ms`, code predictor `488.85 ms`, pipeline `893.5 ms`, RTF `0.228`;
  pre-run GPU utilization was `0%`.
- `benchmark_parity_smoke.ps1 -SelfTest` now covers the benchmark wrapper's
  timing parser and regression-threshold path without launching inference.
  `run_all_tests.ps1 -ParityFixturesOnly` passed with helper smokes plus both
  full fixtures (`PASS: 6`, `FAIL: 0`, `SKIP: 4`). Follow-up no-debug timing
  used the idle-GPU guard and 3 repeats: warm generate median `885.45 ms`, code
  predictor `496.3 ms`, pipeline `919.5 ms`, RTF `0.2345`; pre-run GPU
  utilization was `0%`.
- A model-free expectation metadata smoke now validates
  `tests/fixtures/python_parity_expectations.json` before the full parity
  fixtures consume it and verifies that invalid metadata fails schema checks.
  `run_all_tests.ps1 -ParityFixturesOnly` passed with
  helper smokes plus both full fixtures (`PASS: 7`, `FAIL: 0`, `SKIP: 4`).
  Follow-up no-debug timing used the idle-GPU guard and 3 repeats. The current
  comparison run measured warm generate median `1069.85 ms`, code predictor
  `614.8 ms`, pipeline `1104.0 ms`, RTF `0.282`, about `+22%` versus the
  previous same-session expectation-schema timing baseline (`873.65 ms`
  generate, `903.0 ms` pipeline), with no failures under the `30%` smoke
  threshold; pre-run GPU utilization was `0%`. This change touches only test
  and doc files, so treat the slower run as a timing-noise observation rather
  than a production-code regression.
- `benchmark_parity_smoke.ps1` now reports warm-run min/max/range percentages
  and warns when fewer than `-MinWarmRuns` warm samples are present. The
  self-test covers the spread math and warning path. `run_all_tests.ps1
  -ParityFixturesOnly` passed with helper smokes plus both full fixtures
  (`PASS: 7`, `FAIL: 0`, `SKIP: 4`). Follow-up no-debug timing used the
  idle-GPU guard and 4 repeats, giving 3 warm samples and no benchmark warnings:
  warm generate median `1090.7 ms`, code predictor `629.2 ms`, pipeline
  `1127.0 ms`, RTF `0.288`; warm generate range was `3.18%` of median and
  pipeline range was `3.73%`. The saved same-session expectation-schema
  baseline comparison still showed about `+25%` slower current medians with no
  failures under the `30%` smoke threshold; this change is benchmark-script/doc
  only and does not touch inference code.
- `benchmark_parity_smoke.ps1` now has opt-in warm-spread failure thresholds:
  `-MaxWarmGenerateRangePercent`, `-MaxWarmCodePredRangePercent`,
  `-MaxWarmPipelineRangePercent`, and `-MaxWarmRtfRangePercent`. These report
  `BENCHMARK UNSTABLE` separately from baseline regression failures. The
  self-test covers both passing and failing warm-range thresholds.
  `run_all_tests.ps1 -ParityFixturesOnly` passed with helper smokes plus both
  full fixtures (`PASS: 7`, `FAIL: 0`, `SKIP: 4`). Follow-up no-debug timing
  used the idle-GPU guard, 4 repeats, the saved expectation-schema baseline,
  loose `30%` regression thresholds, and `10%` warm-spread thresholds. It
  produced no benchmark warnings and no stability failures: warm generate
  median `1077.5 ms`, code predictor `622.6 ms`, pipeline `1110.0 ms`, RTF
  `0.283`; warm generate range was `1.93%`, code predictor range `2.84%`, and
  pipeline range `2.16%` of median. Baseline comparison remained slower by
  about `+23%` to `+26%`, under the loose smoke thresholds, with no C++
  inference-path changes in this step.
- `benchmark_parity_smoke.ps1` now records workload/sampling fields in
  `summary.json` (`Language`, `Temperature`, `TopK`, `TopP`, and `Seed`) and
  emits `BaselineComparison.Compatibility` diagnostics for saved baselines.
  `-RequireComparableBaseline` turns incompatibilities into
  `BASELINE INCOMPARABLE` failures, while the default remains warning-only for
  older summaries and now adds a visible `BenchmarkWarnings` entry. The self-test
  covers matching, mismatched, and missing-field baseline metadata, plus the
  non-required warning and required failure routing.
  `run_all_tests.ps1 -ParityFixturesOnly` passed with helper smokes plus both
  full fixtures (`PASS: 7`, `FAIL: 0`, `SKIP: 4`). Follow-up no-debug timing
  used the idle-GPU guard, 4 repeats, loose `30%` regression thresholds, and
  `10%` warm-spread thresholds: warm generate median
  `1052.8 ms`, code predictor `604.3 ms`, pipeline `1087.0 ms`, RTF `0.277`;
  warm generate range was `4.82%` and pipeline range `4.23%` of median. The old
  saved expectation-schema baseline remained under loose regression thresholds,
  reported missing compatibility metadata for `Language`, `Temperature`, `TopK`,
  `TopP`, and `Seed`, and now adds a visible `BenchmarkWarnings` entry pointing
  to `BaselineComparison.Compatibility.Issues`, as expected for a pre-metadata
  summary.
- `run_speaker_parity_fixture.ps1` now writes `fixture_metadata.json` beside
  each generated parity fixture. `run_all_tests.ps1 -ParityFixturesOnly` passed
  with helper smokes, both full fixtures, and both metadata sidecar checks
  (`PASS: 9`, `FAIL: 0`, `SKIP: 4`). The generated sidecars recorded the
  expected modes and gates: speaker fixture `near_tie_token_swap` with
  `MaxTokens=12`, `MaxFrames=10`; ICL fixture `near_tie` with `MaxTokens=4`,
  `MaxFrames=1`. Follow-up no-debug timing used the idle-GPU guard, 4 repeats,
  loose `30%` regression thresholds, and `10%` warm-spread thresholds: warm
  generate median `892.8 ms`, code predictor `501.2 ms`, pipeline `917.0 ms`,
  RTF `0.234`; no stability failures and the old pre-metadata baseline warning
  remained visible as expected.

Latest ICL performance smoke after the non-streaming prefill fix was
current-only, no-debug, 5 process runs with the same 64-token ICL prompt.
Median was `1294.1 ms` total generate, `50.0 ms` prefill, `415.9 ms` talker,
`641.1 ms` code predictor, RTF `0.268`.

Same-session clean-baseline/current ICL benchmark for the non-streaming prefill
fix used 5 alternating runs per binary with inline stripped reference text. The
corrected current path was `1276.0 ms` median total generate vs `1266.6 ms`
baseline (`+0.74%`). Prefill was `50.5 ms` vs `47.1 ms` (`+3.4 ms`,
`+7.22%`), talker was `409.9 ms` vs `415.1 ms` (`-1.25%`), code predictor was
`631.1 ms` vs `621.8 ms` (`+1.50%`), and RTF was `0.265` vs `0.263`
(`+0.76%`).
