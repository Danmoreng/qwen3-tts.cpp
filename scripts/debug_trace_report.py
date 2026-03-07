#!/usr/bin/env python3
"""
Summarize and compare GGML debug trace dumps from TTSTransformer.

Usage:
  python scripts/debug_trace_report.py --trace-a trace_1p7
  python scripts/debug_trace_report.py --trace-a trace_cpp --trace-b trace_py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_shape(shape_text: str) -> tuple[int, ...]:
    shape_text = shape_text.strip()
    if not shape_text:
        return ()
    return tuple(int(x) for x in shape_text.split("x") if x)


def load_manifest(trace_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = trace_dir / "manifest.tsv"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    entries: dict[str, dict[str, Any]] = {}
    lines = manifest.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        name, dtype, count_text, shape_text = parts
        entries[name] = {
            "dtype": dtype,
            "count": int(count_text),
            "shape": parse_shape(shape_text),
        }
    return entries


def dtype_to_np(dtype: str) -> np.dtype:
    if dtype == "f32":
        return np.float32
    if dtype == "i32":
        return np.int32
    raise ValueError(f"unsupported dtype: {dtype}")


def load_entry(trace_dir: Path, name: str, meta: dict[str, Any]) -> np.ndarray:
    path = trace_dir / name
    arr = np.fromfile(path, dtype=dtype_to_np(meta["dtype"]))
    expected = int(meta["count"])
    if arr.size != expected:
        raise ValueError(f"{name}: expected {expected} values, got {arr.size}")
    shape = meta["shape"]
    if shape:
        arr = arr.reshape(shape)
    return arr


def topk(arr: np.ndarray, k: int) -> list[tuple[int, float]]:
    flat = arr.reshape(-1)
    if flat.size == 0:
        return []
    k = max(1, min(k, int(flat.size)))
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    return [(int(i), float(flat[i])) for i in idx]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    finite = np.isfinite(aa) & np.isfinite(bb)
    if not np.any(finite):
        return float("nan")
    aa = aa[finite]
    bb = bb[finite]
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    if den == 0:
        return 0.0
    return float(np.dot(aa, bb) / den)


def summarize_trace(trace_dir: Path, entries: dict[str, dict[str, Any]], top_k_n: int, max_frames: int) -> None:
    print(f"\n=== Trace Summary: {trace_dir} ===")
    frame_re = re.compile(r"^frame(\d+)_")
    frames = sorted(
        {
            int(m.group(1))
            for name in entries
            for m in [frame_re.match(name)]
            if m is not None
        }
    )
    if not frames:
        print("No frame-level files found.")
        return

    if max_frames > 0:
        frames = frames[:max_frames]

    for frame in frames:
        cb0_post_name = f"frame{frame:03d}_cb0_logits_post_rules.f32.bin"
        cb0_raw_name = f"frame{frame:03d}_cb0_logits_raw.f32.bin"
        cb0_tok_name = f"frame{frame:03d}_cb0_token.i32.bin"
        cp_toks_name = f"frame{frame:03d}_codepred_tokens_cb1_15.i32.bin"

        print(f"\n[frame {frame}]")
        if cb0_tok_name in entries:
            cb0_tok = int(load_entry(trace_dir, cb0_tok_name, entries[cb0_tok_name]).reshape(-1)[0])
            print(f"  cb0 token: {cb0_tok}")

        if cb0_post_name in entries:
            logits = load_entry(trace_dir, cb0_post_name, entries[cb0_post_name])
            print(f"  cb0 top-{top_k_n} post-rules: {topk(logits, top_k_n)}")
        elif cb0_raw_name in entries:
            logits = load_entry(trace_dir, cb0_raw_name, entries[cb0_raw_name])
            print(f"  cb0 top-{top_k_n} raw: {topk(logits, top_k_n)}")

        code_tokens: np.ndarray | None = None
        if cp_toks_name in entries:
            code_tokens = load_entry(trace_dir, cp_toks_name, entries[cp_toks_name]).reshape(-1)

        step_re = re.compile(rf"^frame{frame:03d}_codepred_logits_step(\d+)\.f32\.bin$")
        steps = sorted(
            int(m.group(1))
            for name in entries
            for m in [step_re.match(name)]
            if m is not None
        )

        for step in steps:
            name = f"frame{frame:03d}_codepred_logits_step{step:02d}.f32.bin"
            logits = load_entry(trace_dir, name, entries[name])
            selected = None
            if code_tokens is not None and step < code_tokens.size:
                selected = int(code_tokens[step])
            info = f"  codepred step {step:02d} top-{top_k_n}: {topk(logits, top_k_n)}"
            if selected is not None:
                info += f"  selected={selected}"
            print(info)


def compare_traces(
    trace_a: Path,
    entries_a: dict[str, dict[str, Any]],
    trace_b: Path,
    entries_b: dict[str, dict[str, Any]],
) -> None:
    print(f"\n=== Trace Compare ===")
    print(f"A: {trace_a}")
    print(f"B: {trace_b}")

    shared = sorted(set(entries_a.keys()) & set(entries_b.keys()))
    if not shared:
        print("No shared files.")
        return

    compared = 0
    for name in shared:
        ma = entries_a[name]
        mb = entries_b[name]
        if ma["dtype"] != mb["dtype"] or ma["shape"] != mb["shape"] or ma["count"] != mb["count"]:
            continue

        a = load_entry(trace_a, name, ma)
        b = load_entry(trace_b, name, mb)
        if ma["dtype"] == "f32":
            c = cosine(a, b)
            finite = np.isfinite(a) & np.isfinite(b)
            if np.any(finite):
                max_abs = float(np.max(np.abs(a[finite] - b[finite])))
            else:
                max_abs = float("nan")
            argmax_match = int(np.argmax(a.reshape(-1))) == int(np.argmax(b.reshape(-1)))
            print(
                f"{name}: cosine={c:.6f}, max_abs={max_abs:.6e}, argmax_match={argmax_match}"
            )
            compared += 1
        elif ma["dtype"] == "i32":
            exact = bool(np.array_equal(a, b))
            mismatch = int(np.sum(a.reshape(-1) != b.reshape(-1)))
            print(f"{name}: exact={exact}, mismatches={mismatch}")
            compared += 1

    if compared == 0:
        print("No comparable files with matching dtype/shape.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect/compare qwen3-tts debug traces")
    parser.add_argument("--trace-a", required=True, type=Path, help="Primary trace directory")
    parser.add_argument("--trace-b", type=Path, default=None, help="Optional second trace directory")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k entries to print")
    parser.add_argument("--max-frames", type=int, default=8, help="Max frames to print (0=all)")
    args = parser.parse_args()

    entries_a = load_manifest(args.trace_a)
    summarize_trace(args.trace_a, entries_a, top_k_n=args.top_k, max_frames=args.max_frames)

    if args.trace_b is not None:
        entries_b = load_manifest(args.trace_b)
        summarize_trace(args.trace_b, entries_b, top_k_n=args.top_k, max_frames=args.max_frames)
        compare_traces(args.trace_a, entries_a, args.trace_b, entries_b)


if __name__ == "__main__":
    main()
