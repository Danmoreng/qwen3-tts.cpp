#!/usr/bin/env python3
"""
CI-safe smoke test for inspect_safetensors_dtypes.py.

The real parity check inspects a large HuggingFace checkpoint. This test writes
a tiny synthetic safetensors checkpoint and verifies the dtype inspector's
positive and negative expectation paths without shipping model artifacts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


try:
    import torch
    from safetensors.torch import save_file
except ImportError as exc:
    print(f"SKIP: missing Python dependency: {exc.name}")
    raise SystemExit(77)


def make_checkpoint(path: Path) -> None:
    tensors = {
        "talker.code_predictor.model.layers.4.mlp.down_proj.weight": torch.zeros((2, 2), dtype=torch.bfloat16),
        "talker.code_predictor.model.layers.4.mlp.gate_proj.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "talker.model.embed_tokens.weight": torch.arange(4, dtype=torch.bfloat16).reshape(2, 2),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def run_inspector(script: Path, checkpoint: Path, expected_dtype: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(script),
            str(checkpoint.parent),
            "--name-regex",
            r"talker\.code_predictor\.model\.layers\.4\.mlp\.(down|gate)_proj\.weight$",
            "--expect-all-dtype",
            "BF16",
            "--expect-matched-dtype",
            expected_dtype,
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test safetensors dtype inspection on a tiny checkpoint")
    parser.add_argument("--inspector-script", type=Path, default=Path(__file__).with_name("inspect_safetensors_dtypes.py"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="qwen3_tts_dtype_smoke_") as tmp:
            run_smoke(args.inspector_script, Path(tmp))
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_smoke(args.inspector_script, args.output_dir)


def run_smoke(inspector_script: Path, output_dir: Path) -> None:
    checkpoint = output_dir / "tiny_model.safetensors"
    make_checkpoint(checkpoint)

    ok = run_inspector(inspector_script, checkpoint, "BF16")
    if ok.returncode != 0:
        print(ok.stdout)
        raise SystemExit(ok.returncode)
    print("Synthetic safetensors dtype inspection passed.")

    bad = run_inspector(inspector_script, checkpoint, "F32")
    if bad.returncode == 0:
        print(bad.stdout)
        raise SystemExit("negative dtype expectation unexpectedly passed")
    print("Negative dtype expectation failed as expected.")


if __name__ == "__main__":
    main()
