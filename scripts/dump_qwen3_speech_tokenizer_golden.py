#!/usr/bin/env python3
"""Dump Qwen3-TTS speech-tokenizer golden tensors for C++ encoder work.

The script uses the public Qwen3TTSTokenizer API and forward hooks to capture
inputs, selected module outputs, and final audio codes. These artifacts are
intended as behavioral test data for a clean-room C++ implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_HOOKS = (
    "encoder.encoder.layers.0",
    "encoder.encoder.layers.3",
    "encoder.encoder.layers.6",
    "encoder.encoder.layers.9",
    "encoder.encoder.layers.12",
    "encoder.encoder.layers.14",
    "encoder.encoder_transformer.layers.0",
    "encoder.encoder_transformer.layers.7",
    "encoder.downsample",
    "encoder.quantizer.semantic_residual_vector_quantizer.input_proj",
    "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj",
)


def _default_python_repo() -> Path:
    return Path(__file__).resolve().parents[2] / "Qwen3-TTS"


def _tensor_from_any(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "last_hidden_state") and isinstance(value.last_hidden_state, torch.Tensor):
        return value.last_hidden_state
    if hasattr(value, "audio_codes") and isinstance(value.audio_codes, torch.Tensor):
        return value.audio_codes
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _tensor_from_any(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _tensor_from_any(item)
            if tensor is not None:
                return tensor
    return None


def _save_array(out_dir: Path, name: str, tensor: torch.Tensor, summary: dict[str, Any]) -> None:
    safe_name = name.replace(".", "_").replace("/", "_")
    array = tensor.detach().cpu().float().numpy() if tensor.is_floating_point() else tensor.detach().cpu().numpy()
    rel_path = f"{safe_name}.npy"
    np.save(out_dir / rel_path, array)
    summary["tensors"][name] = {
        "file": rel_path,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="Path to the speech_tokenizer directory")
    parser.add_argument("--audio", required=True, help="Reference WAV path")
    parser.add_argument("--out-dir", required=True, help="Directory for .npy tensors and summary.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--python-repo", default=str(_default_python_repo()))
    parser.add_argument("--hook", action="append", default=[], help="Additional module name to hook")
    args = parser.parse_args()

    python_repo = Path(args.python_repo)
    if python_repo.exists():
        sys.path.insert(0, str(python_repo))

    import librosa
    from qwen_tts import Qwen3TTSTokenizer

    dtype = _resolve_dtype(args.dtype, str(args.device))
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer,
        device_map=args.device,
        dtype=dtype,
        attn_implementation="eager",
    )
    tokenizer.model.eval()

    sample_rate = int(getattr(tokenizer.feature_extractor, "sampling_rate", 24000))
    audio, loaded_sr = librosa.load(args.audio, sr=sample_rate, mono=True)
    features = tokenizer.feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    input_values = features["input_values"].to(device=args.device, dtype=dtype)
    model_input_values = input_values
    if model_input_values.ndim == 3 and model_input_values.shape[1] == 1:
        model_input_values = model_input_values[:, 0, :]
    padding_mask = features.get("padding_mask")
    if padding_mask is not None:
        padding_mask = padding_mask.to(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "audio": str(Path(args.audio).resolve()),
        "loaded_sample_rate": int(loaded_sr),
        "feature_sample_rate": sample_rate,
        "num_samples": int(audio.shape[0]),
        "device": str(args.device),
        "dtype": str(dtype).replace("torch.", ""),
        "hooks": {},
        "tensors": {},
    }

    _save_array(out_dir, "input_values", input_values, summary)
    _save_array(out_dir, "model_input_values", model_input_values, summary)
    if padding_mask is not None:
        _save_array(out_dir, "padding_mask", padding_mask, summary)

    modules = dict(tokenizer.model.named_modules())
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            tensor = _tensor_from_any(output)
            if tensor is not None and name not in captured:
                captured[name] = tensor.detach()
        return hook

    for name in list(DEFAULT_HOOKS) + list(args.hook):
        module = modules.get(name)
        if module is None:
            summary["hooks"][name] = {"found": False}
            continue
        summary["hooks"][name] = {"found": True, "class": module.__class__.__name__}
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.inference_mode():
        model_encoded = tokenizer.model.encode(input_values=model_input_values, padding_mask=padding_mask)
        api_encoded = tokenizer.encode(audio, sr=sample_rate)

    for handle in handles:
        handle.remove()

    for name, tensor in captured.items():
        _save_array(out_dir, f"hook.{name}", tensor, summary)

    model_codes = _tensor_from_any(model_encoded)
    if model_codes is not None:
        _save_array(out_dir, "model_encode_audio_codes", model_codes.to(torch.int32), summary)
    api_codes = api_encoded.audio_codes[0].detach().cpu().to(torch.int32)
    _save_array(out_dir, "api_audio_codes", api_codes, summary)

    summary["api_audio_codes_preview"] = api_codes[: min(8, api_codes.shape[0])].tolist()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
