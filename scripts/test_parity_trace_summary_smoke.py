#!/usr/bin/env python3
"""
CI-safe smoke test for parity_trace_summary.py.

The full Python/C++ parity fixtures need large local model artifacts. This test
creates tiny synthetic trace directories at runtime and verifies that the
summary CLI reports the expected first-diff token metadata and near-tie
classification.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def write_trace_entry(trace_dir: Path, rows: list[str], name: str, dtype: str, array: np.ndarray) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    array.reshape(-1).tofile(trace_dir / name)
    shape_text = "x".join(str(dim) for dim in array.shape)
    rows.append(f"{name}\t{dtype}\t{array.size}\t{shape_text}\n")


def write_manifest(trace_dir: Path, rows: list[str]) -> None:
    (trace_dir / "manifest.tsv").write_text("name\tdtype\tcount\tshape\n" + "".join(rows), encoding="utf-8")


def make_synthetic_traces(root: Path) -> tuple[Path, Path]:
    trace_a = root / "trace_a"
    trace_b = root / "trace_b"
    rows_a: list[str] = []
    rows_b: list[str] = []

    tokens_a = np.arange(16, dtype=np.int32)
    tokens_b = tokens_a.copy()
    tokens_a[6] = 10
    tokens_b[6] = 11
    write_trace_entry(trace_a, rows_a, "frame000_codec_tokens_cb0_15.i32.bin", "i32", tokens_a)
    write_trace_entry(trace_b, rows_b, "frame000_codec_tokens_cb0_15.i32.bin", "i32", tokens_b)

    logits_a = np.zeros(16, dtype=np.float32)
    logits_b = np.zeros(16, dtype=np.float32)
    logits_a[10] = 2.00
    logits_a[11] = 1.99
    logits_b[10] = 2.00
    logits_b[11] = 2.01
    write_trace_entry(trace_a, rows_a, "frame000_codepred_logits_step05.f32.bin", "f32", logits_a)
    write_trace_entry(trace_b, rows_b, "frame000_codepred_logits_step05.f32.bin", "f32", logits_b)

    write_manifest(trace_a, rows_a)
    write_manifest(trace_b, rows_b)
    return trace_a, trace_b


def run_summary(script: Path, trace_a: Path, trace_b: Path, output: Path, category: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--trace-a",
            str(trace_a),
            "--trace-b",
            str(trace_b),
            "--label-a",
            "synthetic_a",
            "--label-b",
            "synthetic_b",
            "--output",
            str(output),
            "--expect-match-percent-at-least",
            "93.0",
            "--expect-first-diff-frame",
            "0",
            "--expect-first-diff-codebook",
            "6",
            "--expect-first-diff-token-a",
            "10",
            "--expect-first-diff-token-b",
            "11",
            "--expect-first-diff-category",
            category,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test parity trace summary on synthetic traces")
    parser.add_argument("--summary-script", type=Path, default=Path(__file__).with_name("parity_trace_summary.py"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="qwen3_tts_parity_smoke_") as tmp:
            run_smoke(args.summary_script, Path(tmp))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_smoke(args.summary_script, args.output_dir)


def run_smoke(summary_script: Path, output_dir: Path) -> None:
    trace_a, trace_b = make_synthetic_traces(output_dir)
    summary_path = output_dir / "summary.json"

    ok = run_summary(summary_script, trace_a, trace_b, summary_path, "near_tie_token_swap")
    if ok.returncode != 0:
        print(ok.stdout)
        raise SystemExit(ok.returncode)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    category = summary["first_diff_classification"]["category"]
    first_diff = summary["tokens"]["first_diff"]
    print(
        "Synthetic parity summary: "
        f"frame={first_diff['frame']} codebook={first_diff['codebook']} "
        f"token_a={first_diff['token_a']} token_b={first_diff['token_b']} "
        f"category={category}"
    )

    bad = run_summary(summary_script, trace_a, trace_b, output_dir / "bad_summary.json", "logit_drift")
    if bad.returncode == 0:
        print(bad.stdout)
        raise SystemExit("negative category expectation unexpectedly passed")
    print("Negative category expectation failed as expected.")


if __name__ == "__main__":
    main()
