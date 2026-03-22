# Tasks: Reference-Text Voice Cloning

**Input**: Design documents from `/specs/001-reftext-clone-prompt/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are REQUIRED when production behavior changes. Every task list
MUST include the verification work needed by the constitution, even if that means
updating deterministic references, component tests, benchmarks, or docs rather
than adding a brand-new test binary.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Support tooling**: `scripts/`, `reference/`, `.github/workflows/`, `README.md`
- Paths shown below use the current native inference project structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Align the existing implementation surface with the new feature plan before shared prompt work begins

- [X] T001 Review the current voice-clone entry points and validation logic in `/home/tak/code/qwen3-tts.cpp/src/main.cpp`, `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts.h`, and `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts_c.h`
- [X] T002 Review current reference/parity helpers for richer prompt signals in `/home/tak/code/qwen3-tts.cpp/scripts/generate_deterministic_reference.py`, `/home/tak/code/qwen3-tts.cpp/scripts/generate_reference_outputs.py`, and `/home/tak/code/qwen3-tts.cpp/scripts/compare_e2e.py`
- [X] T003 [P] Review Windows regression and CI entry points in `/home/tak/code/qwen3-tts.cpp/scripts/run_all_tests.ps1`, `/home/tak/code/qwen3-tts.cpp/scripts/run_all_tests.sh`, and `/home/tak/code/qwen3-tts.cpp/.github/workflows/windows-tests.yml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared runtime and artifact foundations that MUST exist before any user story can be completed

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Define the reusable voice-clone prompt asset types and file I/O helpers in `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts.h` and `/home/tak/code/qwen3-tts.cpp/src/common/speaker_embedding_io.cpp`
- [X] T005 [P] Extend the native C API contract for prompt assets in `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts_c.h` and `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts_c.cpp`
- [ ] T006 [P] Add shared prompt-asset validation and model-compatibility checks in `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts.cpp` and `/home/tak/code/qwen3-tts.cpp/src/pipeline/pipeline_models.cpp`
- [ ] T007 Update shared transformer/runtime types for reference-aware prompt mode in `/home/tak/code/qwen3-tts.cpp/src/tts_transformer.h` and `/home/tak/code/qwen3-tts.cpp/src/transformer/transformer_embeddings.cpp`
- [ ] T008 [P] Add shared deterministic reference metadata for prompt modes in `/home/tak/code/qwen3-tts.cpp/scripts/generate_deterministic_reference.py` and `/home/tak/code/qwen3-tts.cpp/reference/det_metadata.json`
- [ ] T009 [P] Add shared regression harness hooks for prompt assets in `/home/tak/code/qwen3-tts.cpp/scripts/run_all_tests.sh`, `/home/tak/code/qwen3-tts.cpp/scripts/run_all_tests.ps1`, and `/home/tak/code/qwen3-tts.cpp/.github/workflows/windows-tests.yml`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Create a Reference-Aware Voice Profile (Priority: P1) 🎯 MVP

**Goal**: Let users create and persist a reusable prompt asset from reference audio plus matching reference text

**Independent Test**: Run the CLI with `--reference` plus reference text input, write a prompt asset file, and verify the file records `reference_aware` mode with compatible metadata and explicit validation failures for unusable text

### Tests for User Story 1 ⚠️

> **NOTE: Write or update these checks FIRST, and ensure they fail or clearly
> demonstrate the missing coverage before implementation**

- [ ] T010 [P] [US1] Update richer prompt creation fixtures in `/home/tak/code/qwen3-tts.cpp/scripts/generate_reference_outputs.py` and `/home/tak/code/qwen3-tts.cpp/reference/det_metadata.json`
- [ ] T011 [P] [US1] Add regression coverage for prompt creation and validation failures in `/home/tak/code/qwen3-tts.cpp/tests/test_transformer.cpp`
- [ ] T012 [P] [US1] Add CLI-level creation coverage in `/home/tak/code/qwen3-tts.cpp/scripts/run_all_tests.sh` and `/home/tak/code/qwen3-tts.cpp/scripts/run_all_tests.ps1`

### Implementation for User Story 1

- [X] T013 [P] [US1] Implement reference-aware prompt asset creation in `/home/tak/code/qwen3-tts.cpp/src/pipeline/pipeline_synthesize.cpp` and `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts.cpp`
- [ ] T014 [P] [US1] Implement transformer prefill support for reference-aware prompt creation in `/home/tak/code/qwen3-tts.cpp/src/tts_transformer.h`, `/home/tak/code/qwen3-tts.cpp/src/transformer/transformer_generate.cpp`, and `/home/tak/code/qwen3-tts.cpp/src/transformer/transformer_embeddings.cpp`
- [X] T015 [US1] Add CLI flags for reference text input and prompt asset dump in `/home/tak/code/qwen3-tts.cpp/src/main.cpp`
- [ ] T016 [US1] Add C and JNI creation surfaces for richer prompt assets in `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts_c.h`, `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts_c.cpp`, and `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts_jni.cpp`
- [X] T017 [US1] Document prompt asset creation and validation behavior in `/home/tak/code/qwen3-tts.cpp/README.md` and `/home/tak/code/qwen3-tts.cpp/specs/001-reftext-clone-prompt/quickstart.md`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Reuse the Voice Profile for New Speech (Priority: P1)

**Goal**: Let users load a previously created prompt asset and synthesize new speech without re-supplying the original reference inputs

**Independent Test**: Create one prompt asset, synthesize at least two utterances from it through the CLI and native API surfaces, and verify the original reference audio/text are not required again

### Tests for User Story 2 ⚠️

- [ ] T018 [P] [US2] Update deterministic reference generation for prompt reuse and decode behavior in `/home/tak/code/qwen3-tts.cpp/scripts/generate_deterministic_reference.py` and `/home/tak/code/qwen3-tts.cpp/reference/det_metadata.json`
- [ ] T019 [P] [US2] Add regression coverage for prompt asset reuse in `/home/tak/code/qwen3-tts.cpp/tests/test_transformer.cpp`
- [ ] T020 [P] [US2] Add end-to-end reuse/parity coverage in `/home/tak/code/qwen3-tts.cpp/scripts/compare_e2e.py`

### Implementation for User Story 2

- [X] T021 [P] [US2] Implement synthesis from reusable prompt assets in `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts.h`, `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts.cpp`, and `/home/tak/code/qwen3-tts.cpp/src/pipeline/pipeline_synthesize.cpp`
- [X] T022 [P] [US2] Implement reference-aware decode and prompt reuse handling in `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts.cpp` and `/home/tak/code/qwen3-tts.cpp/src/pipeline/pipeline_synthesize.cpp`
- [X] T023 [US2] Add CLI and native API reuse surfaces for `--voice-clone-prompt` in `/home/tak/code/qwen3-tts.cpp/src/main.cpp`, `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts_c.h`, and `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts_c.cpp`
- [X] T024 [US2] Update reuse examples and asset semantics in `/home/tak/code/qwen3-tts.cpp/README.md` and `/home/tak/code/qwen3-tts.cpp/specs/001-reftext-clone-prompt/contracts/voice-clone-prompt.md`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Preserve the Existing Audio-Only Workflow (Priority: P2)

**Goal**: Keep the current audio-only speaker-embedding path intact and clearly distinguish it from the new richer prompt mode

**Independent Test**: Run the existing audio-only cloning flow and saved speaker-embedding reuse flow after the richer prompt changes, then validate that malformed or incompatible prompt assets fail cleanly without breaking the existing fast path

### Tests for User Story 3 ⚠️

- [ ] T025 [P] [US3] Add regression coverage for audio-only non-regression and prompt-mode compatibility failures in `/home/tak/code/qwen3-tts.cpp/tests/test_transformer.cpp` and `/home/tak/code/qwen3-tts.cpp/tests/test_encoder.cpp`
- [ ] T026 [P] [US3] Update shell and PowerShell regression harnesses for audio-only and richer-mode coexistence in `/home/tak/code/qwen3-tts.cpp/scripts/run_all_tests.sh` and `/home/tak/code/qwen3-tts.cpp/scripts/run_all_tests.ps1`
- [ ] T027 [P] [US3] Update Windows CI prerequisite checks and regression steps for the new mode in `/home/tak/code/qwen3-tts.cpp/.github/workflows/windows-tests.yml`

### Implementation for User Story 3

- [X] T028 [P] [US3] Preserve and harden audio-only compatibility logic in `/home/tak/code/qwen3-tts.cpp/src/main.cpp`, `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts.h`, and `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts.cpp`
- [X] T029 [US3] Implement explicit prompt-mode compatibility diagnostics in `/home/tak/code/qwen3-tts.cpp/src/qwen3_tts_c.cpp`, `/home/tak/code/qwen3-tts.cpp/src/common/speaker_embedding_io.cpp`, and `/home/tak/code/qwen3-tts.cpp/src/pipeline/pipeline_synthesize.cpp`
- [X] T030 [US3] Update docs to distinguish audio-only and reference-aware workflows in `/home/tak/code/qwen3-tts.cpp/README.md` and `/home/tak/code/qwen3-tts.cpp/specs/001-reftext-clone-prompt/quickstart.md`

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T031 [P] Refresh benchmark or parity helper expectations for both prompt modes in `/home/tak/code/qwen3-tts.cpp/scripts/benchmark_pytorch_vs_cpp.py` and `/home/tak/code/qwen3-tts.cpp/scripts/benchmark_python_vs_cpp.ps1`
- [ ] T032 Review and tighten prompt asset versioning and compatibility messages in `/home/tak/code/qwen3-tts.cpp/src/common/speaker_embedding_io.cpp` and `/home/tak/code/qwen3-tts.cpp/specs/001-reftext-clone-prompt/contracts/voice-clone-prompt.md`
- [ ] T033 [P] Run the quickstart validation pass and align examples with final CLI/API names in `/home/tak/code/qwen3-tts.cpp/specs/001-reftext-clone-prompt/quickstart.md` and `/home/tak/code/qwen3-tts.cpp/README.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel where dependencies allow
  - Recommended order is P1 US1 → P1 US2 → P2 US3 because reuse depends on asset creation and compatibility work depends on both modes existing
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Depends on User Story 1 prompt asset creation contract, but should be independently testable once that contract exists
- **User Story 3 (P2)**: Depends on User Stories 1 and 2 so compatibility rules can be validated against both prompt modes

### Within Each User Story

- Verification tasks MUST be written or updated before implementation
- Public contract and serialization changes before broader CLI/API exposure
- Core runtime implementation before harness and doc finalization
- Story complete before moving to the next dependent story

### Parallel Opportunities

- T003 can run in parallel with T001-T002
- T005, T006, T008, and T009 can run in parallel after T004
- Within US1, T010-T012 can run in parallel, and T013-T014 can run in parallel once foundational tasks are done
- Within US2, T018-T020 can run in parallel, and T021-T022 can run in parallel before T023
- Within US3, T025-T027 can run in parallel, and T028 can proceed in parallel with T029
- T031 and T033 can run in parallel during polish

---

## Parallel Example: User Story 1

```bash
# Launch all verification work for User Story 1 together:
Task: "Update richer prompt creation fixtures in /home/tak/code/qwen3-tts.cpp/scripts/generate_reference_outputs.py and /home/tak/code/qwen3-tts.cpp/reference/det_metadata.json"
Task: "Add regression coverage for prompt creation and validation failures in /home/tak/code/qwen3-tts.cpp/tests/test_transformer.cpp"
Task: "Add CLI-level creation coverage in /home/tak/code/qwen3-tts.cpp/scripts/run_all_tests.sh and /home/tak/code/qwen3-tts.cpp/scripts/run_all_tests.ps1"

# Launch independent implementation work for User Story 1 together:
Task: "Implement reference-aware prompt asset creation in /home/tak/code/qwen3-tts.cpp/src/pipeline/pipeline_synthesize.cpp and /home/tak/code/qwen3-tts.cpp/src/qwen3_tts.cpp"
Task: "Implement transformer prefill support for reference-aware prompt creation in /home/tak/code/qwen3-tts.cpp/src/tts_transformer.h, /home/tak/code/qwen3-tts.cpp/src/transformer/transformer_generate.cpp, and /home/tak/code/qwen3-tts.cpp/src/transformer/transformer_embeddings.cpp"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Create a reference-aware prompt asset from one audio sample plus matching text and verify validation failures for bad text
5. Share the created asset contract and CLI/API behavior for review

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Share prompt creation outputs (MVP)
3. Add User Story 2 → Test independently → Share prompt reuse outputs
4. Add User Story 3 → Test independently → Share compatibility and non-regression results
5. Finish polish items and re-run quickstart plus regression harnesses

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 runtime and CLI creation path
   - Developer B: User Story 2 reuse path and parity/reference updates
   - Developer C: User Story 3 compatibility, docs, and CI alignment after US1/US2 contracts stabilize
3. Integrate stories in priority order and finish polish together

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently testable from the feature spec acceptance criteria
- Verification tasks are included for every story because production behavior changes
- All tasks include explicit file paths so they are directly executable by an LLM or engineer
