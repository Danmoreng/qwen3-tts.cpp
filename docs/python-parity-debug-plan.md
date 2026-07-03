# Python Parity Debug Plan

Last updated: 2026-07-03

## Purpose

Use Python parity as a focused debugging tool for `qwen3-tts.cpp`.

The goal is not to promise bit-identical WAV output between PyTorch and GGML across full generations. Small numerical differences, backend kernels, BF16 rounding, sampling, and autoregressive accumulation make full end-to-end identity fragile.

The goal is narrower and more useful:

- Match prompt construction inputs where practical.
- Match the first generated codebook-0 token.
- Find and fix the first deterministic divergence in the code predictor.
- Use top-token parity for local steps as the main correctness signal.

The current highest-value target is the first divergent step:

- Frame: `0`
- Codebook: `1`
- Area: code predictor / subtalker path

## Current Findings

| Area | Finding | Status |
|---|---|---|
| Python prompt extraction | Faster Qwen3 TTS matches Python prompt codes exactly when extra silence appending is disabled. | Verified |
| audio.cpp prompt extraction | Reference-code values diverge from Python despite matching frame count. | Verified divergent |
| BF16 GGUF conversion | A mostly-BF16 GGUF can be built directly from the BF16 HuggingFace checkpoint. | Implemented locally |
| Speaker-only BF16 generation parity | C++ diverges from Python at frame `0`, codebook `1`. | Failing |
| ICL BF16 generation parity | C++ diverges from Python at frame `0`, codebook `1`. | Failing |
| Greedy path | Greedy currently produces invalid/repetitive output and should not be used as the main parity gate until fixed. | Known failing |

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

## Phase 2: BF16 Model Path

Keep BF16 conversion available because Python CUDA commonly runs the model in BF16.

Required implementation:

- `scripts/convert_tts_to_gguf.py` supports `--type bf16`.
- `scripts/convert_tokenizer_to_gguf.py` supports `--type bf16`.
- BF16 tensors are written as `GGML_TYPE_BF16`, not NumPy `float16`.
- 1D tensors and precision-sensitive tensors remain F32 where current policy requires it.

Success criteria:

- GGUF inspection shows a mostly-BF16 model with expected F32 exceptions.
- The C++ loader can load and run the BF16 model.
- The BF16 path reproduces the same first-divergence location as the current parity fixture.

## Phase 3: First Divergence Trace

Instrument or reuse debug trace dumps for the first divergent code-predictor step.

For frame `0`, codebook `1`, dump and compare:

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

## Phase 6: Regression Gates

Add lightweight regression checks once a parity bug is fixed.

Candidate gates:

- A code-predictor first-step top-token test using checked-in small metadata and external/generated binary fixtures.
- A script-level parity check that is skipped unless required model artifacts are present.
- A CI-safe smoke test that validates tensor shapes and known first-step token IDs.

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

Start with Phase 3.

Dump Python and C++ tensors for frame `0`, codebook `1`, using:

- Python speaker embedding
- Python reference codes for ICL
- `top_k=1`
- `temperature=1.0`
- BF16 model path where available

Then identify whether the first material difference is before or inside the code predictor graph.
