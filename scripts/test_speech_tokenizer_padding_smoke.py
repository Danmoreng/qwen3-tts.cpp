#!/usr/bin/env python3
"""
CI-safe smoke test for Qwen3-TTS speech-tokenizer MimiConv1d padding.

The full speech-tokenizer parity check needs large model artifacts. This test
keeps the small shape invariant that fixed the local ICL prompt extraction bug:
the final downsample convolution must add Mimi's dynamic right padding, turning
the local 149-frame hidden sequence into 75 projected frames instead of the old
74-frame result.
"""

from __future__ import annotations

import argparse
import math


def python_mimi_extra_padding(length: int, kernel_size: int, stride: int, padding_total: int) -> int:
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return int(ideal_length - length)


def cpp_mimi_extra_padding(length: int, kernel_size: int, stride: int, padding_total: int) -> int:
    numerator = length - kernel_size + padding_total
    n_frames_minus_one = math.ceil(numerator / stride)
    ideal_length = n_frames_minus_one * stride + kernel_size - padding_total
    return max(ideal_length - length, 0)


def conv1d_output_length(length: int, kernel_size: int, stride: int, left_pad: int, right_pad: int) -> int:
    padded_length = length + left_pad + right_pad
    return math.floor((padded_length - kernel_size) / stride) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test speech-tokenizer MimiConv1d padding math")
    parser.parse_args()

    cases = [
        (149, 4, 2, 2, 1, 75),
        (148, 4, 2, 2, 0, 74),
        (150, 4, 2, 2, 0, 75),
        (151, 4, 2, 2, 1, 76),
        (24000, 8, 4, 4, 0, 6000),
        (24001, 8, 4, 4, 3, 6001),
    ]

    for length, kernel_size, stride, padding_total, expected_extra, expected_output in cases:
        python_extra = python_mimi_extra_padding(length, kernel_size, stride, padding_total)
        cpp_extra = cpp_mimi_extra_padding(length, kernel_size, stride, padding_total)
        if python_extra != expected_extra or cpp_extra != expected_extra:
            raise SystemExit(
                "extra padding mismatch: "
                f"length={length} kernel={kernel_size} stride={stride} padding={padding_total} "
                f"python={python_extra} cpp={cpp_extra} expected={expected_extra}"
            )

        output = conv1d_output_length(length, kernel_size, stride, padding_total, expected_extra)
        if output != expected_output:
            raise SystemExit(
                "output length mismatch: "
                f"length={length} kernel={kernel_size} stride={stride} padding={padding_total} "
                f"extra={expected_extra} output={output} expected={expected_output}"
            )

    old_output = conv1d_output_length(149, 4, 2, 2, 0)
    if old_output != 74:
        raise SystemExit(f"negative regression check expected old output 74, got {old_output}")

    fixed_output = conv1d_output_length(149, 4, 2, 2, cpp_mimi_extra_padding(149, 4, 2, 2))
    if fixed_output != 75:
        raise SystemExit(f"fixed regression check expected output 75, got {fixed_output}")

    print("Speech tokenizer Mimi padding smoke passed.")
    print("Local downsample case: 149 input frames -> 75 output frames with extra_padding=1.")


if __name__ == "__main__":
    main()
