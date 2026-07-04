#!/usr/bin/env python3
"""
CI-safe schema smoke for run_speaker_parity_fixture.ps1 fixture metadata.

Full parity fixture generation needs large local model artifacts. This check
uses tiny synthetic metadata files to validate the sidecar shape and the same
expectation fields consumed by the full fixture gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {"SchemaVersion", "FixtureMode", "Python", "Cpp", "Inputs", "Expectations", "Outputs"}
ALLOWED_MODES = {"speaker", "icl"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic parity fixture metadata sidecars")
    parser.add_argument(
        "--expectations",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "python_parity_expectations.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expectations = json.loads(args.expectations.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    speaker_summary = args.output_dir / "speaker_summary.json"
    speaker_summary.write_text("{}\n", encoding="utf-8")
    icl_summary = args.output_dir / "icl_summary.json"
    icl_summary.write_text("{}\n", encoding="utf-8")

    speaker_payload = make_payload("speaker", expectations["fixtures"]["speaker_only"], speaker_summary)
    icl_payload = make_payload("icl", expectations["fixtures"]["icl"], icl_summary)

    failures = validate_payload(speaker_payload, "speaker", expectations["fixtures"]["speaker_only"])
    failures += validate_payload(icl_payload, "icl", expectations["fixtures"]["icl"])
    if failures:
        for failure in failures:
            print(f"FIXTURE METADATA FAILED: {failure}")
        raise SystemExit(1)

    invalid_failures = validate_payload(make_invalid_payload(), "speaker", expectations["fixtures"]["speaker_only"])
    if not invalid_failures:
        raise SystemExit("negative fixture metadata check unexpectedly passed")

    (args.output_dir / "speaker_fixture_metadata.json").write_text(json.dumps(speaker_payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "icl_fixture_metadata.json").write_text(json.dumps(icl_payload, indent=2) + "\n", encoding="utf-8")
    print("Parity fixture metadata smoke passed.")
    print("Negative fixture metadata check failed as expected.")


def make_payload(mode: str, fixture: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    expect = fixture["expect"]
    return {
        "SchemaVersion": 1,
        "FixtureMode": mode,
        "Python": {
            "Exe": "python",
            "Model": "synthetic-python-model",
            "Path": "synthetic-python-path",
            "Device": "cpu",
            "DType": "float32",
            "Language": "English",
            "DoSample": True,
        },
        "Cpp": {
            "CliExe": "qwen3-tts-cli.exe",
            "ModelDir": "synthetic-cpp-modeldir",
            "Language": "en",
            "Temperature": 1.0,
            "TopK": 1,
            "TopP": 1.0,
            "Seed": 0,
        },
        "Inputs": {
            "Text": fixture["text"],
            "SpeakerEmbedding": "synthetic-speaker.json",
            "ReferenceText": None,
            "ReferenceTextFile": "synthetic-ref.txt" if mode == "icl" else None,
            "ReferenceCodes": "synthetic-codes.json" if mode == "icl" else None,
            "MaxTokens": fixture["max_tokens"],
            "MaxFrames": fixture["max_frames"],
        },
        "Expectations": {
            "MatchPercentAtLeast": expect["match_percent_at_least"],
            "FirstDiffFrame": expect["first_diff_frame"],
            "FirstDiffCodebook": expect["first_diff_codebook"],
            "FirstDiffTokenA": expect["first_diff_token_a"],
            "FirstDiffTokenB": expect["first_diff_token_b"],
            "FirstDiffCosineAtLeast": expect["first_diff_cosine_at_least"],
            "FirstDiffMaxAbsAtMost": expect["first_diff_max_abs_at_most"],
            "FirstDiffCategory": expect["first_diff_category"],
            "FirstDiffMaxAbsOverMarginAtLeast": expect["first_diff_max_abs_over_margin_at_least"],
        },
        "Outputs": {
            "PythonTraceDir": "synthetic-python-trace",
            "CppTraceDir": "synthetic-cpp-trace",
            "Summary": str(summary_path),
        },
    }


def make_invalid_payload() -> dict[str, Any]:
    return {
        "SchemaVersion": 2,
        "FixtureMode": "bad-mode",
        "Inputs": {
            "Text": "",
            "MaxTokens": 0,
            "MaxFrames": -1,
        },
        "Expectations": {
            "FirstDiffCategory": "logit_drift",
            "FirstDiffFrame": -1,
            "FirstDiffCodebook": 99,
        },
        "Outputs": {
            "Summary": "missing-summary.json",
        },
    }


def validate_payload(payload: dict[str, Any], expected_mode: str, fixture: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    missing = REQUIRED_TOP_LEVEL - set(payload)
    for key in sorted(missing):
        failures.append(f"{key}: missing")

    if payload.get("SchemaVersion") != 1:
        failures.append(f"SchemaVersion: expected 1, got {payload.get('SchemaVersion')!r}")

    mode = payload.get("FixtureMode")
    if mode not in ALLOWED_MODES:
        failures.append(f"FixtureMode: expected one of {sorted(ALLOWED_MODES)}, got {mode!r}")
    elif mode != expected_mode:
        failures.append(f"FixtureMode: expected {expected_mode!r}, got {mode!r}")

    inputs = payload.get("Inputs")
    if not isinstance(inputs, dict):
        failures.append("Inputs: expected object")
    else:
        if inputs.get("Text") != fixture["text"]:
            failures.append("Inputs.Text: did not match fixture text")
        if inputs.get("MaxTokens") != fixture["max_tokens"]:
            failures.append(f"Inputs.MaxTokens: expected {fixture['max_tokens']}, got {inputs.get('MaxTokens')!r}")
        if inputs.get("MaxFrames") != fixture["max_frames"]:
            failures.append(f"Inputs.MaxFrames: expected {fixture['max_frames']}, got {inputs.get('MaxFrames')!r}")

    python = payload.get("Python")
    if not isinstance(python, dict):
        failures.append("Python: expected object")
    else:
        if python.get("DoSample") is not True:
            failures.append(f"Python.DoSample: expected True, got {python.get('DoSample')!r}")
        if python.get("DType") != "float32":
            failures.append(f"Python.DType: expected 'float32', got {python.get('DType')!r}")

    cpp = payload.get("Cpp")
    if not isinstance(cpp, dict):
        failures.append("Cpp: expected object")
    else:
        expected_cpp = {
            "Temperature": 1.0,
            "TopK": 1,
            "TopP": 1.0,
            "Seed": 0,
        }
        for field, expected_value in expected_cpp.items():
            if cpp.get(field) != expected_value:
                failures.append(f"Cpp.{field}: expected {expected_value!r}, got {cpp.get(field)!r}")

    expect = fixture["expect"]
    metadata_expect = payload.get("Expectations")
    if not isinstance(metadata_expect, dict):
        failures.append("Expectations: expected object")
    else:
        expected_fields = {
            "FirstDiffCategory": expect["first_diff_category"],
            "FirstDiffFrame": expect["first_diff_frame"],
            "FirstDiffCodebook": expect["first_diff_codebook"],
            "FirstDiffTokenA": expect["first_diff_token_a"],
            "FirstDiffTokenB": expect["first_diff_token_b"],
        }
        for field, expected_value in expected_fields.items():
            if metadata_expect.get(field) != expected_value:
                failures.append(f"Expectations.{field}: expected {expected_value!r}, got {metadata_expect.get(field)!r}")

    outputs = payload.get("Outputs")
    if not isinstance(outputs, dict):
        failures.append("Outputs: expected object")
    else:
        summary = outputs.get("Summary")
        if not isinstance(summary, str) or not summary:
            failures.append("Outputs.Summary: expected non-empty path")
        elif not Path(summary).exists():
            failures.append(f"Outputs.Summary: does not exist: {summary}")

    return failures


if __name__ == "__main__":
    main()
