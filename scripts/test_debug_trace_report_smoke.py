#!/usr/bin/env python3
"""
CI-safe smoke test for debug_trace_report.py.

The report is a human-facing diagnostic, so this test builds tiny synthetic
trace directories and verifies that raw/post-warp logit labels plus compare
metrics appear in the output.
"""

from __future__ import annotations

import argparse
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


def make_trace(root: Path, label: str, offset: float) -> Path:
    trace = root / label
    rows: list[str] = []

    cb0_token = np.array([7], dtype=np.int32)
    cb0_raw = np.array([0.1 + offset, 1.0, 0.2, 0.0], dtype=np.float32)
    cb0_post = np.array([0.1, 0.3 + offset, 1.2, 0.0], dtype=np.float32)
    codepred_tokens = np.array([2, 3], dtype=np.int32)
    codepred_raw = np.array([0.0, 0.5 + offset, 1.5, 0.1], dtype=np.float32)
    codepred_post = np.array([0.0, 0.2, 0.4 + offset, 1.1], dtype=np.float32)

    write_trace_entry(trace, rows, "frame000_cb0_token.i32.bin", "i32", cb0_token)
    write_trace_entry(trace, rows, "frame000_cb0_logits_raw.f32.bin", "f32", cb0_raw)
    write_trace_entry(trace, rows, "frame000_cb0_logits_post_rules.f32.bin", "f32", cb0_post)
    write_trace_entry(trace, rows, "frame000_codepred_tokens_cb1_15.i32.bin", "i32", codepred_tokens)
    write_trace_entry(trace, rows, "frame000_codepred_logits_step00.f32.bin", "f32", codepred_raw)
    write_trace_entry(trace, rows, "frame000_codepred_logits_step00_post_warp.f32.bin", "f32", codepred_post)
    write_manifest(trace, rows)
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test debug trace reporting on synthetic traces")
    parser.add_argument("--report-script", type=Path, default=Path(__file__).with_name("debug_trace_report.py"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="qwen3_tts_debug_report_smoke_") as tmp:
            run_smoke(args.report_script, Path(tmp))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_smoke(args.report_script, args.output_dir)


def run_smoke(report_script: Path, output_dir: Path) -> None:
    trace_a = make_trace(output_dir, "trace_a", 0.0)
    trace_b = make_trace(output_dir, "trace_b", 0.05)
    result = subprocess.run(
        [
            sys.executable,
            str(report_script),
            "--trace-a",
            str(trace_a),
            "--trace-b",
            str(trace_b),
            "--top-k",
            "2",
            "--max-frames",
            "1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        print(result.stdout)
        raise SystemExit(result.returncode)

    required = [
        "cb0 top-2 post-rules",
        "codepred step 00 raw top-2",
        "codepred step 00 post-warp top-2",
        "frame000_cb0_logits_raw.f32.bin: cosine=",
        "frame000_codepred_tokens_cb1_15.i32.bin: exact=True",
    ]
    missing = [needle for needle in required if needle not in result.stdout]
    if missing:
        print(result.stdout)
        raise SystemExit(f"debug trace report smoke missing output: {missing}")
    print("Synthetic debug trace report smoke passed.")


if __name__ == "__main__":
    main()
