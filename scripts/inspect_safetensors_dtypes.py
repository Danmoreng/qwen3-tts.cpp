#!/usr/bin/env python3
"""
Inspect safetensors checkpoint dtypes without loading tensor data.

This is useful for parity work where GGUF storage choices only help if the
source checkpoint actually contains higher-precision tensors.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from safetensors import safe_open


def iter_safetensors(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.safetensors")))
        else:
            files.append(path)
    return files


def inspect_files(files: list[Path], name_regex: str | None) -> dict[str, Any]:
    pattern = re.compile(name_regex) if name_regex else None
    dtype_counts: Counter[str] = Counter()
    matched_dtype_counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    matched: list[dict[str, Any]] = []
    tensor_count = 0

    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor_count += 1
                tensor_slice = handle.get_slice(name)
                dtype = str(tensor_slice.get_dtype())
                shape = [int(dim) for dim in tensor_slice.get_shape()]
                dtype_counts[dtype] += 1
                examples.setdefault(dtype, name)
                if pattern and pattern.search(name):
                    matched_dtype_counts[dtype] += 1
                    matched.append(
                        {
                            "file": str(file_path),
                            "name": name,
                            "dtype": dtype,
                            "shape": shape,
                        }
                    )

    return {
        "files": [str(path) for path in files],
        "tensor_count": tensor_count,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "examples": examples,
        "filter_regex": name_regex,
        "matched_count": len(matched),
        "matched_dtype_counts": dict(sorted(matched_dtype_counts.items())),
        "matched": matched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect safetensors checkpoint dtypes")
    parser.add_argument("paths", nargs="+", type=Path, help="Safetensors files or directories")
    parser.add_argument("--name-regex", default=None, help="Optional tensor-name regex to report matched tensors")
    parser.add_argument("--expect-all-dtype", default=None, help="Fail if any tensor does not have this dtype")
    parser.add_argument("--expect-matched-dtype", default=None, help="Fail if any matched tensor does not have this dtype")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    files = iter_safetensors(args.paths)
    if not files:
        raise SystemExit("No safetensors files found")

    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise SystemExit(f"Missing safetensors file(s): {', '.join(missing)}")

    result = inspect_files(files, args.name_regex)

    failures: list[str] = []
    if args.expect_all_dtype is not None:
        bad = {
            dtype: count
            for dtype, count in result["dtype_counts"].items()
            if dtype != args.expect_all_dtype
        }
        if bad:
            failures.append(f"Expected all tensors to be {args.expect_all_dtype}, found {bad}")

    if args.expect_matched_dtype is not None:
        if result["matched_count"] == 0:
            failures.append("Expected matched tensors, found none")
        bad_matches = [
            item
            for item in result["matched"]
            if item["dtype"] != args.expect_matched_dtype
        ]
        if bad_matches:
            failures.append(
                f"Expected matched tensors to be {args.expect_matched_dtype}, "
                f"found {len(bad_matches)} mismatch(es)"
            )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"files: {len(result['files'])}")
        print(f"tensors: {result['tensor_count']}")
        print(f"dtype_counts: {result['dtype_counts']}")
        if args.name_regex:
            print(f"matched_count: {result['matched_count']}")
            print(f"matched_dtype_counts: {result['matched_dtype_counts']}")
            for item in result["matched"]:
                print(f"  {item['name']}: {item['dtype']} {item['shape']}")

    if failures:
        for failure in failures:
            print(f"EXPECTATION FAILED: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
