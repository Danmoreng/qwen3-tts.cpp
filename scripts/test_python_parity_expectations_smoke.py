#!/usr/bin/env python3
"""
CI-safe schema smoke for tests/fixtures/python_parity_expectations.json.

The full parity fixtures need large local model artifacts. This check validates
the small checked-in expectation metadata so obvious schema mistakes fail before
the expensive local fixture regeneration starts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_CATEGORIES = {"exact_tie", "near_tie_token_swap", "near_tie", "token_swap", "logit_drift"}
REQUIRED_FIXTURES = ("speaker_only", "icl")
REQUIRED_EXPECT_FIELDS = {
    "match_percent_at_least": (int, float),
    "first_diff_frame": int,
    "first_diff_codebook": int,
    "first_diff_token_a": int,
    "first_diff_token_b": int,
    "first_diff_cosine_at_least": (int, float),
    "first_diff_max_abs_at_most": (int, float),
    "first_diff_category": str,
    "first_diff_max_abs_over_margin_at_least": (int, float),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Python parity expectation metadata")
    parser.add_argument(
        "--expectations",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "python_parity_expectations.json",
    )
    args = parser.parse_args()

    payload = json.loads(args.expectations.read_text(encoding="utf-8"))
    failures = validate_payload(payload)
    if failures:
        for failure in failures:
            print(f"EXPECTATION SCHEMA FAILED: {failure}")
        raise SystemExit(1)

    negative_failures = validate_payload(make_invalid_payload())
    if not negative_failures:
        raise SystemExit("negative expectation schema check unexpectedly passed")

    print(f"Python parity expectation schema passed: {args.expectations}")
    print("Negative expectation schema check failed as expected.")


def validate_payload(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("version") != 1:
        failures.append(f"version: expected 1, got {payload.get('version')!r}")

    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, dict):
        return failures + ["fixtures: expected object"]

    for fixture_name in REQUIRED_FIXTURES:
        fixture = fixtures.get(fixture_name)
        if not isinstance(fixture, dict):
            failures.append(f"fixtures.{fixture_name}: missing or not an object")
            continue

        text = fixture.get("text")
        if not isinstance(text, str) or not text.strip():
            failures.append(f"fixtures.{fixture_name}.text: expected non-empty string")

        for key in ("max_tokens", "max_frames"):
            value = fixture.get(key)
            if not isinstance(value, int) or value <= 0:
                failures.append(f"fixtures.{fixture_name}.{key}: expected positive integer")

        expect = fixture.get("expect")
        if not isinstance(expect, dict):
            failures.append(f"fixtures.{fixture_name}.expect: expected object")
            continue

        for field, expected_type in REQUIRED_EXPECT_FIELDS.items():
            if field not in expect:
                failures.append(f"fixtures.{fixture_name}.expect.{field}: missing")
                continue
            if not isinstance(expect[field], expected_type):
                failures.append(f"fixtures.{fixture_name}.expect.{field}: expected {expected_type}, got {type(expect[field]).__name__}")

        if isinstance(expect.get("match_percent_at_least"), (int, float)):
            value = float(expect["match_percent_at_least"])
            if value < 0.0 or value > 100.0:
                failures.append(f"fixtures.{fixture_name}.expect.match_percent_at_least: expected 0..100")

        if isinstance(expect.get("first_diff_codebook"), int):
            value = int(expect["first_diff_codebook"])
            has_first_diff = isinstance(expect.get("first_diff_frame"), int) and int(expect["first_diff_frame"]) >= 0
            if has_first_diff and (value < 0 or value > 15):
                failures.append(f"fixtures.{fixture_name}.expect.first_diff_codebook: expected 0..15")
            if not has_first_diff and value != -1:
                failures.append(f"fixtures.{fixture_name}.expect.first_diff_codebook: expected -1 when first_diff_frame is -1")

        has_first_diff = isinstance(expect.get("first_diff_frame"), int) and int(expect["first_diff_frame"]) >= 0
        if has_first_diff and expect.get("first_diff_category") not in ALLOWED_CATEGORIES:
            failures.append(
                f"fixtures.{fixture_name}.expect.first_diff_category: "
                f"expected one of {sorted(ALLOWED_CATEGORIES)}, got {expect.get('first_diff_category')!r}"
            )
        if not has_first_diff and expect.get("first_diff_category") != "":
            failures.append(
                f"fixtures.{fixture_name}.expect.first_diff_category: expected empty string when first_diff_frame is -1"
            )

        if not has_first_diff:
            for field in ("first_diff_token_a", "first_diff_token_b"):
                if expect.get(field) != -1:
                    failures.append(f"fixtures.{fixture_name}.expect.{field}: expected -1 when first_diff_frame is -1")
            for field in ("first_diff_cosine_at_least", "first_diff_max_abs_at_most", "first_diff_max_abs_over_margin_at_least"):
                if expect.get(field) != -1.0:
                    failures.append(f"fixtures.{fixture_name}.expect.{field}: expected -1.0 when first_diff_frame is -1")

    return failures


def make_invalid_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "fixtures": {
            "speaker_only": {
                "text": "Synthetic bad fixture",
                "max_tokens": 1,
                "max_frames": 1,
                "expect": {
                    "match_percent_at_least": 101.0,
                    "first_diff_frame": 0,
                    "first_diff_codebook": 99,
                    "first_diff_token_a": 1,
                    "first_diff_token_b": 2,
                    "first_diff_cosine_at_least": 0.0,
                    "first_diff_max_abs_at_most": 1.0,
                    "first_diff_category": "not_a_category",
                    "first_diff_max_abs_over_margin_at_least": 1.0,
                },
            },
            "icl": {
                "text": "",
                "max_tokens": 0,
                "max_frames": 0,
                "expect": {},
            },
        },
    }


if __name__ == "__main__":
    main()
