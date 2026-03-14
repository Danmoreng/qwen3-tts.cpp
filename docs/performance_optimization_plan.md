# Performance Optimization Plan

## Goal

Bring `qwen3-tts.cpp` closer to or beyond the runtime of `faster-qwen3-tts` by attacking the actual hot path in this codebase:

1. Repeated graph build/allocation in the talker and code predictor decode loops
2. Repeated host/backend data movement for masks, positions, logits, and embedding rows
3. CPU fallback and thread configuration gaps
4. Smaller per-frame allocation and sampling overhead

This plan is intentionally staged. Each phase should land independently, keep tests passing, and produce a measurable speedup before moving to the next phase.

## Guiding Rules

- Measure before and after every phase.
- Do not combine multiple large runtime changes in one commit.
- Preserve deterministic behavior for greedy decoding unless a phase explicitly changes numeric behavior.
- Keep the Python reference and existing component tests usable throughout.
- Prefer changes that reduce work in the inner decode loop over broad refactors.

## Baseline And Instrumentation

### Step 0.1: Establish a repeatable benchmark

Create a fixed benchmark script and benchmark inputs for:

- Short text
- Medium text
- Long text
- Voice-clone path
- Plain synthesis path

Use the same:

- Model
- Backend
- Build type
- Thread count
- Sampling settings
- Prompt set

Record at minimum:

- End-to-end wall time
- `t_generate_ms`
- Frames per second
- Prefill time
- Talker step time
- Code predictor time
- Decoder time

Artifacts:

- `docs/benchmark_pytorch_vs_cpp.json` already exists and can be extended or complemented
- Add a dedicated benchmark command doc if needed

Acceptance criteria:

- We can reproduce a baseline run on demand
- We can compare phases without changing benchmark conditions

### Step 0.2: Tighten timing visibility

Review and extend timing output so it clearly separates:

- Graph build
- Graph alloc
- Tensor upload
- Compute
- Tensor readback

Files to inspect first:

- `src/transformer/transformer_generate.cpp`
- `src/transformer/transformer_runtime.cpp`
- `src/transformer/transformer_runtime_code_pred.cpp`
- `src/pipeline/pipeline_synthesize.cpp`

Notes:

- The timing hooks are already useful, but we should keep them stable while optimizing so regressions are obvious.
- Remove noisy unconditional logging from hot paths once baseline data is captured.

Acceptance criteria:

- Timing output makes it obvious which substep moved after each phase

## Phase 1: Low-Risk Runtime Fixes

These changes are small, isolated, and should land first.

### Step 1.1: Actually apply `n_threads`

Problem:

- `tts_params.n_threads` is exposed in the public API, but it is not currently wired into the GGML backends.

Work:

- Add a shared helper to apply thread count to CPU-capable backends.
- Apply it in:
  - transformer loader
  - encoder loader
  - decoder loader
- If a backend-specific `set_n_threads` symbol exists, use it.
- At minimum, configure CPU backend and CPU fallback backend.

Likely files:

- `src/qwen3_tts.h`
- `src/pipeline/pipeline_models.cpp`
- `src/transformer/transformer_loader.cpp`
- `src/encoder/encoder_loader.cpp`
- `src/decoder/decoder_loader.cpp`
- `src/gguf_loader.{h,cpp}`

Acceptance criteria:

- Running with different `--threads` values changes effective runtime on CPU paths
- No regression on GPU paths

### Step 1.2: Remove avoidable hot-path logging

Problem:

- There is unconditional debug-style logging in runtime-adjacent code.

Work:

- Remove or gate any unconditional `fprintf` in synthesis hot paths behind env flags or timing/debug flags.

Likely files:

- `src/transformer/transformer_embeddings.cpp`
- Any runtime file with unconditional per-run or per-frame logs

Acceptance criteria:

- No unexpected logging during standard synthesis
- Benchmark variance is lower

### Step 1.3: Reuse sampling scratch buffers

Problem:

- Top-k sampling currently allocates temporary vectors inside inner loops.

Work:

- Move temporary storage for:
  - `std::vector<std::pair<float, int32_t>>`
  - probability buffers
  - logits scratch
into reusable transformer state or reusable local buffers created once per generate call.

Likely files:

- `src/transformer/transformer_generate.cpp`
- `src/transformer/transformer_runtime_code_pred.cpp`
- `src/transformer/transformer_state_internal.h`

Acceptance criteria:

- No behavioral change
- Reduced allocation churn in profiler

## Phase 2: Remove Inner-Loop CPU/Backend Round-Trips

This is the first major speed phase.

### Step 2.1: Stop per-token embedding row fetches from backend memory

Problem:

- `lookup_single_embedding_row()` reads one row at a time from backend tensors.
- Generation performs repeated row fetches for:
  - CB0 codec embedding
  - code predictor embeddings for CB1-15

This is especially costly when the tensor lives on GPU and every row requires a device-to-host transfer.

Work:

- Add a CPU mirror path for frequently accessed embedding tables used in decode:
  - `codec_embd`
  - `code_pred_embd[*]`
- Populate mirrors once after model load, or lazily on first use.
- Change step embedding composition to read from CPU memory directly.

Alternative:

- Build a small device-side graph that gathers all needed embeddings in one operation per frame.

Preferred order:

1. CPU mirror for lowest implementation risk
2. Device-side gather only if the mirror approach is still too slow on GPU

Likely files:

- `src/transformer/transformer_embeddings.cpp`
- `src/transformer/transformer_loader.cpp`
- `src/transformer/transformer_loader_tensors.cpp`
- `src/transformer/transformer_state_internal.h`
- `src/transformer/transformer_generate.cpp`

Acceptance criteria:

- Same generated output for greedy decoding
- Clear reduction in `t_embed_lookup_ms`

### Step 2.2: Precompute constant token projections used by prefill

Problem:

- Special token projection work is repeated during prefill construction.

Work:

- Cache projected representations for stable special tokens and any constant prompt fragments used in prefill.
- Avoid rebuilding these vectors on every synthesis request.

Likely files:

- `src/transformer/transformer_embeddings.cpp`
- `src/transformer/transformer_state_internal.h`

Acceptance criteria:

- Prefill construction time decreases
- No change in emitted embeddings

## Phase 3: Eliminate Repeated Graph Build/Alloc In Decode

This is the core optimization phase and likely the largest single win.

### Step 3.1: Introduce persistent decode graph state for the talker

Problem:

- `forward_step()` rebuilds the talker step graph and allocates it for every frame.

Work:

- Add a persistent graph object for talker single-step decode.
- Build it once for a given context size.
- Store direct tensor handles for:
  - `inp_step_embd`
  - `inp_pos`
  - `inp_mrope_pos`
  - `inp_mask`
  - `hidden_states`
  - `logits`
- Recompute by updating tensor contents only.

Design notes:

- Rebuild only when context capacity changes.
- Keep cache reset semantics intact.
- Ensure the scheduler reservation and persistent graph strategy do not conflict.

Likely files:

- `src/transformer/transformer_runtime.cpp`
- `src/transformer/transformer_graph_talker.cpp`
- `src/transformer/transformer_state_internal.h`
- `src/transformer/transformer_cache.cpp`

Acceptance criteria:

- `t_talker_graph_build_ms` and `t_talker_graph_alloc_ms` go near zero during steady-state decode
- Runtime remains correct across variable prompt lengths

### Step 3.2: Introduce persistent graphs for code predictor prefill and steps

Problem:

- The code predictor currently pays graph build and alloc once for the prefill and once for each of the 14 autoregressive steps per frame.

Work:

- Create persistent graph state for:
  - code predictor prefill graph
  - code predictor step graph for each generation step 1..14
- Store direct handles to inputs and outputs.
- Rebuild only when required tensor shapes or context capacity change.

Likely files:

- `src/transformer/transformer_runtime_code_pred.cpp`
- `src/transformer/transformer_graph_code_pred.cpp`
- `src/transformer/transformer_state_internal.h`
- `src/transformer/transformer_cache.cpp`

Acceptance criteria:

- `t_code_pred_graph_build_ms` and `t_code_pred_graph_alloc_ms` drop sharply
- The code predictor remains numerically stable compared with previous greedy outputs

### Step 3.3: Reevaluate scheduler reserve warmup

Problem:

- `maybe_reserve_scheduler_graphs()` is useful today, but once graphs become persistent its role may need to change.

Work:

- Keep reserve warmup only if it still provides value.
- If persistent graphs fully replace the dynamic path, simplify reserve logic and avoid duplicate complexity.

Likely files:

- `src/transformer/transformer_cache.cpp`
- `src/transformer/transformer_runtime.cpp`
- `src/transformer/transformer_runtime_code_pred.cpp`

Acceptance criteria:

- Reserve logic is either simplified or justified by measurement

## Phase 4: Remove Repeated Mask And Position Materialization

### Step 4.1: Replace full mask rebuilds with reusable mask buffers

Problem:

- The talker and code predictor rebuild and upload whole masks repeatedly.

Work:

- Keep a persistent mask buffer in state.
- Initialize it once for the maximum context.
- Update only the newly unlocked position when advancing decode.
- If possible, avoid uploading unchanged regions.

Likely files:

- `src/transformer/transformer_runtime.cpp`
- `src/transformer/transformer_runtime_code_pred.cpp`
- `src/transformer/transformer_state_internal.h`

Acceptance criteria:

- `t_talker_data_ms` and `t_code_pred_data_ms` fall measurably

### Step 4.2: Reuse position buffers

Problem:

- Position arrays for prefill and mRoPE are rebuilt into temporary vectors repeatedly.

Work:

- Add reusable position buffers in state.
- Fill only the active slice needed for current token counts.

Likely files:

- `src/transformer/transformer_runtime.cpp`
- `src/transformer/transformer_state_internal.h`

Acceptance criteria:

- Lower allocation count during prefill and decode

## Phase 5: Reduce Tensor Readback Overhead

### Step 5.1: Minimize output fetches from backend

Problem:

- We currently fetch full logits or hidden vectors to host as separate operations.

Work:

- Audit where host readback is truly needed.
- For talker step:
  - we need logits
  - we need last hidden state for code predictor
- For code predictor:
  - we need logits for sampling
- Avoid any redundant hidden or logits fetch.

Potential improvements:

- Fuse readbacks where practical
- Keep last hidden in reusable host buffer
- Avoid fetching full hidden sequences when only the last row is needed

Likely files:

- `src/transformer/transformer_runtime.cpp`
- `src/transformer/transformer_runtime_code_pred.cpp`

Acceptance criteria:

- Reduced `*_data_ms` without changing outputs

### Step 5.2: Special-case greedy decode

Problem:

- Greedy decode does not need the same sampling machinery as stochastic decode.

Work:

- Fast-path `temperature <= 0` so sampling overhead is minimal.
- Keep host-side argmax efficient and allocation-free.

Likely files:

- `src/transformer/transformer_generate.cpp`
- `src/transformer/transformer_runtime_code_pred.cpp`

Acceptance criteria:

- Greedy runs become faster than sampling runs by a clear margin

## Phase 6: Optional Model-Level And Backend-Level Improvements

These are meaningful, but they should come after the decode loop is fixed.

### Step 6.1: Quantized model profiles

Work:

- Benchmark F16 vs Q8_0 vs Q5_K vs Q4_K for the transformer model.
- Focus on:
  - talker blocks
  - code predictor blocks
- Confirm quality and determinism tradeoffs.

Likely files:

- `src/qwen3_tts_quantize.cpp`
- benchmark scripts and docs

Acceptance criteria:

- We have a recommended performance/quality quantization profile

### Step 6.2: Backend-specific tuning

Work:

- Confirm CUDA builds are used when expected.
- Compare:
  - CPU only
  - GPU with CPU fallback
  - CUDA-enabled Release
- Check whether fallback placement is introducing avoidable transfers.

Likely files:

- `CMakeLists.txt`
- `build.ps1`
- loader files for transformer, encoder, decoder

Acceptance criteria:

- We know which backend configuration is fastest on Windows and macOS

### Step 6.3: Investigate batched or fused code predictor execution

Problem:

- The code predictor is structurally sequential, but there may still be room to reduce launch overhead further.

Work:

- Investigate whether multiple code predictor steps can share more state or use a more fused execution path.
- Treat this as a research phase, not an initial implementation target.

Acceptance criteria:

- A short design note exists before implementation begins

## Validation Strategy

Every phase should run:

- Transformer component tests
- Decoder component tests if tensor shapes or output layout changed
- Existing deterministic/reference tests where applicable
- A fixed benchmark comparison against the previous commit

For runtime-sensitive phases, capture:

- before/after timing table
- backend used
- model used
- threads used
- exact command line

## Suggested Commit Sequence

Recommended commit order:

1. `docs(perf): add staged performance optimization plan`
2. `fix(runtime): apply configured thread counts to ggml backends`
3. `perf(runtime): remove hot-path allocation churn in sampling`
4. `perf(transformer): cache decode embedding rows on cpu`
5. `perf(transformer): persist talker decode graph`
6. `perf(transformer): persist code predictor decode graphs`
7. `perf(transformer): reuse masks and position buffers`
8. `perf(benchmark): add repeatable performance benchmark workflow`

## Expected Impact

Rough expectation by phase:

- Phase 1: small but immediate gain
- Phase 2: moderate gain, especially on GPU-backed runs
- Phase 3: largest gain
- Phase 4: moderate additive gain
- Phase 5: smaller additive gain
- Phase 6: hardware- and model-dependent gain

The most likely path to closing the gap with `faster-qwen3-tts` is:

1. Make decode graphs persistent
2. Remove per-frame device/host embedding fetches
3. Reuse masks and positions
4. Ensure backend/thread configuration is actually optimal

## Risks

- Persistent graph state can become invalid when context size changes unless rebuild rules are explicit.
- CPU embedding mirrors increase memory usage.
- Quantization can change generation trajectories and subjective audio quality.
- Backend-specific improvements may help one platform and hurt another if not benchmarked separately.

## Exit Criteria

We consider this effort successful when:

- The benchmark suite shows a clear and repeatable throughput gain
- The code predictor no longer dominates due to graph build/allocation overhead
- Greedy synthesis remains correct against current reference expectations
- The repo has a documented benchmark workflow and a documented recommended runtime configuration
