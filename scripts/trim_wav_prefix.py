#!/usr/bin/env python3
"""Copy the first N seconds of a PCM WAV file.

This helper is intentionally small and dependency-free so benchmark scripts can
pin the exact reference-audio duration given to every framework.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()

    if args.seconds <= 0.0:
        raise ValueError("--seconds must be positive")

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(src), "rb") as inp:
        params = inp.getparams()
        max_frames = max(1, int(round(args.seconds * params.framerate)))
        frames = inp.readframes(min(max_frames, params.nframes))

    with wave.open(str(dst), "wb") as out:
        out.setparams(params)
        out.writeframes(frames)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
