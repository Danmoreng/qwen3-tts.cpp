#!/usr/bin/env python3
"""
Create a reusable voice clone prompt JSON asset using the upstream qwen-tts
Python package when reference-aware cloning is requested.

This is intentionally a helper bridge. qwen3-tts.cpp consumes the saved asset
for synthesis; prompt creation remains aligned with upstream until a native
Mimi-based tokenizer encoder exists in C++.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_hf_base_model(models_dir: Path) -> Path | None:
    candidates = []
    for child in models_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "config.json").exists() and (child / "speech_tokenizer" / "model.safetensors").exists():
            candidates.append(child)
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a reusable voice clone prompt asset")
    parser.add_argument("--model-dir", required=True, help="Directory containing HF model assets and GGUF files")
    parser.add_argument("--reference-audio", required=True, help="Reference WAV/audio input")
    parser.add_argument("--reference-text", required=True, help="Transcript matching the reference audio")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_path = Path(args.output).resolve()

    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except Exception as exc:
        print(
            "ERROR: reference-aware prompt creation requires the upstream `qwen-tts` Python package "
            "plus its torch dependencies.\n"
            f"Import failure: {exc}",
            file=sys.stderr,
        )
        return 2

    hf_model_dir = find_hf_base_model(model_dir)
    if hf_model_dir is None:
        print(
            "ERROR: could not find a Hugging Face base model directory under "
            f"{model_dir}. Expected a subdirectory with config.json and speech_tokenizer/model.safetensors.",
            file=sys.stderr,
        )
        return 3

    model = Qwen3TTSModel.from_pretrained(
        str(hf_model_dir),
        device_map="cpu",
        dtype=torch.float32,
    )
    model.model = model.model.eval()

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=str(Path(args.reference_audio).resolve()),
        ref_text=args.reference_text,
        x_vector_only_mode=False,
    )
    prompt_item = prompt_items[0]

    ref_code = prompt_item.ref_code
    if ref_code is None:
        print("ERROR: upstream prompt creation did not return reference codes", file=sys.stderr)
        return 4

    speaker_embedding = prompt_item.ref_spk_embedding.float().cpu().numpy().reshape(-1)
    reference_codes = ref_code.cpu().numpy()
    if reference_codes.ndim != 2:
        print(f"ERROR: unexpected ref_code shape: {reference_codes.shape}", file=sys.stderr)
        return 5

    asset = {
        "format_version": 1,
        "prompt_mode": "reference_aware",
        "model_kind": "base",
        "model_name": hf_model_dir.name,
        "speaker_embedding_dim": int(speaker_embedding.shape[0]),
        "speaker_embedding": speaker_embedding.tolist(),
        "reference_text": args.reference_text,
        "reference_codebooks": int(reference_codes.shape[1]),
        "reference_frames": int(reference_codes.shape[0]),
        "reference_codes": reference_codes.reshape(-1).astype("int32").tolist(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asset, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
