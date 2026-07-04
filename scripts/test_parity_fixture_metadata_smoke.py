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
    icl_summary = args.output_dir / "icl_summary.json"

    speaker_payload = make_payload("speaker", expectations["fixtures"]["speaker_only"], speaker_summary)
    icl_payload = make_payload("icl", expectations["fixtures"]["icl"], icl_summary)
    speaker_summary.write_text(
        json.dumps(make_summary(expectations["fixtures"]["speaker_only"]), indent=2) + "\n",
        encoding="utf-8",
    )
    icl_summary.write_text(
        json.dumps(make_summary(expectations["fixtures"]["icl"]), indent=2) + "\n",
        encoding="utf-8",
    )

    failures = validate_payload(speaker_payload, "speaker", expectations["fixtures"]["speaker_only"])
    failures += validate_payload(icl_payload, "icl", expectations["fixtures"]["icl"])
    if failures:
        for failure in failures:
            print(f"FIXTURE METADATA FAILED: {failure}")
        raise SystemExit(1)

    invalid_failures = validate_payload(make_invalid_payload(), "speaker", expectations["fixtures"]["speaker_only"])
    if not invalid_failures:
        raise SystemExit("negative fixture metadata check unexpectedly passed")

    bad_summary = args.output_dir / "bad_summary.json"
    bad_payload = make_payload("speaker", expectations["fixtures"]["speaker_only"], bad_summary)
    bad_summary_payload = make_summary(expectations["fixtures"]["speaker_only"])
    bad_summary_payload["tokens"]["first_diff"]["token_a"] = -1
    bad_summary.write_text(json.dumps(bad_summary_payload, indent=2) + "\n", encoding="utf-8")
    bad_summary_failures = validate_payload(bad_payload, "speaker", expectations["fixtures"]["speaker_only"])
    if not any("summary.tokens.first_diff.token_a" in failure for failure in bad_summary_failures):
        raise SystemExit("negative fixture summary check unexpectedly passed")

    (args.output_dir / "speaker_fixture_metadata.json").write_text(json.dumps(speaker_payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "icl_fixture_metadata.json").write_text(json.dumps(icl_payload, indent=2) + "\n", encoding="utf-8")
    print("Parity fixture metadata smoke passed.")
    print("Negative fixture metadata check failed as expected.")
    print("Negative fixture summary check failed as expected.")


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


def make_summary(fixture: dict[str, Any]) -> dict[str, Any]:
    expect = fixture["expect"]
    return {
        "tokens": {
            "frames_compared": fixture["max_frames"],
            "tokens_compared": fixture["max_frames"] * 16,
            "tokens_matching": int(fixture["max_frames"] * 16 * expect["match_percent_at_least"] / 100),
            "match_percent": expect["match_percent_at_least"],
            "first_diff": {
                "frame": expect["first_diff_frame"],
                "codebook": expect["first_diff_codebook"],
                "token_a": expect["first_diff_token_a"],
                "token_b": expect["first_diff_token_b"],
            },
        },
        "logits_at_first_diff": {
            "cosine": expect["first_diff_cosine_at_least"],
            "max_abs": expect["first_diff_max_abs_at_most"],
        },
        "first_diff_classification": {
            "category": expect["first_diff_category"],
            "max_abs_over_min_top1_margin": expect["first_diff_max_abs_over_margin_at_least"],
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
        else:
            summary_payload = json.loads(Path(summary).read_text(encoding="utf-8"))
            failures.extend(validate_summary(summary_payload, fixture))

    return failures


def validate_summary(summary: dict[str, Any], fixture: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expect = fixture["expect"]

    tokens = summary.get("tokens")
    if not isinstance(tokens, dict):
        failures.append("summary.tokens: expected object")
    else:
        if tokens.get("match_percent") is None or tokens["match_percent"] < expect["match_percent_at_least"]:
            failures.append(
                "summary.tokens.match_percent: "
                f"expected >= {expect['match_percent_at_least']}, got {tokens.get('match_percent')!r}"
            )
        first_diff = tokens.get("first_diff")
        if not isinstance(first_diff, dict):
            failures.append("summary.tokens.first_diff: expected object")
        else:
            expected_fields = {
                "frame": expect["first_diff_frame"],
                "codebook": expect["first_diff_codebook"],
                "token_a": expect["first_diff_token_a"],
                "token_b": expect["first_diff_token_b"],
            }
            for field, expected_value in expected_fields.items():
                if first_diff.get(field) != expected_value:
                    failures.append(
                        f"summary.tokens.first_diff.{field}: expected {expected_value!r}, got {first_diff.get(field)!r}"
                    )

    logits = summary.get("logits_at_first_diff")
    if not isinstance(logits, dict):
        failures.append("summary.logits_at_first_diff: expected object")
    else:
        if logits.get("cosine") is None or logits["cosine"] < expect["first_diff_cosine_at_least"]:
            failures.append(
                "summary.logits_at_first_diff.cosine: "
                f"expected >= {expect['first_diff_cosine_at_least']}, got {logits.get('cosine')!r}"
            )
        if logits.get("max_abs") is None or logits["max_abs"] > expect["first_diff_max_abs_at_most"]:
            failures.append(
                "summary.logits_at_first_diff.max_abs: "
                f"expected <= {expect['first_diff_max_abs_at_most']}, got {logits.get('max_abs')!r}"
            )

    classification = summary.get("first_diff_classification")
    if not isinstance(classification, dict):
        failures.append("summary.first_diff_classification: expected object")
    else:
        if classification.get("category") != expect["first_diff_category"]:
            failures.append(
                "summary.first_diff_classification.category: "
                f"expected {expect['first_diff_category']!r}, got {classification.get('category')!r}"
            )
        ratio = classification.get("max_abs_over_min_top1_margin")
        if ratio is None or ratio < expect["first_diff_max_abs_over_margin_at_least"]:
            failures.append(
                "summary.first_diff_classification.max_abs_over_min_top1_margin: "
                f"expected >= {expect['first_diff_max_abs_over_margin_at_least']}, got {ratio!r}"
            )

    return failures


if __name__ == "__main__":
    main()
