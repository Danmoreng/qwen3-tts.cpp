#!/usr/bin/env python3
"""Small worker used by benchmark_frameworks.ps1 for Python Qwen3 TTS stacks."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np


def _import_torch():
    import torch

    return torch


def _write_wav(path: Path, wav, sample_rate: int) -> None:
    import soundfile as sf

    arr = wav[0] if isinstance(wav, (list, tuple)) else wav
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, sample_rate)


def _audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def _set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch = _import_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync_cuda() -> None:
    torch = _import_torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _torch_dtype(name: str):
    torch = _import_torch()
    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def _load_official(args: argparse.Namespace):
    if args.repo:
        sys.path.insert(0, str(Path(args.repo).resolve()))
    from qwen_tts import Qwen3TTSModel

    kwargs = {}
    if args.device_map:
        kwargs["device_map"] = args.device_map
    dtype = _torch_dtype(args.dtype)
    if dtype is not None:
        kwargs["dtype"] = dtype
    return Qwen3TTSModel.from_pretrained(args.model, **kwargs)


def _load_faster(args: argparse.Namespace):
    if args.repo:
        sys.path.insert(0, str(Path(args.repo).resolve()))
    from faster_qwen3_tts import FasterQwen3TTS

    dtype = _torch_dtype(args.dtype)
    kwargs = {}
    if args.device_map:
        kwargs["device"] = "cuda" if args.device_map.startswith("cuda") else args.device_map
    if dtype is not None:
        kwargs["dtype"] = dtype
    return FasterQwen3TTS.from_pretrained(args.model, **kwargs)


def _move_prompt(prompt, device: str | None = None):
    if isinstance(prompt, dict):
        return {key: _move_prompt(value, device) for key, value in prompt.items()}
    if isinstance(prompt, list):
        return [_move_prompt(value, device) for value in prompt]
    if hasattr(prompt, "detach"):
        tensor = prompt.detach()
        if device:
            tensor = tensor.to(device)
        else:
            tensor = tensor.cpu()
        return tensor
    return prompt


def _create_voice_prompt(model, args: argparse.Namespace):
    ref_text = "" if args.xvec_only else args.reference_text
    if args.backend == "faster":
        base = model.model
        prompt_items = base.create_voice_clone_prompt(
            ref_audio=args.reference_audio,
            ref_text=ref_text,
            x_vector_only_mode=args.xvec_only,
        )
        return base._prompt_items_to_voice_clone_prompt(prompt_items)

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=args.reference_audio,
        ref_text=ref_text,
        x_vector_only_mode=args.xvec_only,
    )
    return model._prompt_items_to_voice_clone_prompt(prompt_items)


def _load_voice_prompt(path: Path, device_hint: str):
    torch = _import_torch()
    target = device_hint if device_hint and device_hint.startswith("cuda") else None
    prompt = torch.load(str(path), map_location=target or "cpu", weights_only=False)
    return _move_prompt(prompt, target)


def _generate_voice_clone(model, args: argparse.Namespace):
    kwargs = {
        "text": args.text,
        "language": args.language,
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": not args.greedy,
        "repetition_penalty": args.repetition_penalty,
    }

    if args.prompt_artifact:
        kwargs["voice_clone_prompt"] = _load_voice_prompt(Path(args.prompt_artifact), args.device_map)
    else:
        kwargs["ref_audio"] = args.reference_audio
        kwargs["ref_text"] = args.reference_text

    if args.backend == "faster":
        if not args.prompt_artifact:
            kwargs["xvec_only"] = args.xvec_only
        kwargs["non_streaming_mode"] = args.non_streaming_mode
    else:
        if not args.prompt_artifact:
            kwargs["x_vector_only_mode"] = args.xvec_only
        kwargs["non_streaming_mode"] = args.non_streaming_mode

    if args.streaming:
        if args.backend != "faster":
            raise ValueError("--streaming is currently supported only with --backend faster")
        kwargs["chunk_size"] = args.chunk_size
        kwargs["parity_mode"] = args.parity_mode
        chunks = []
        sample_rate = 24000
        ttfa_s = None
        t0 = time.perf_counter()
        for chunk, sample_rate, timing in model.generate_voice_clone_streaming(**kwargs):
            _sync_cuda()
            if ttfa_s is None:
                ttfa_s = time.perf_counter() - t0
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        _sync_cuda()
        synth_s = time.perf_counter() - t0
        wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        return [wav], sample_rate, {
            "streaming": True,
            "chunk_size": args.chunk_size,
            "ttfa_seconds": ttfa_s,
            "synth_seconds": synth_s,
        }

    return model.generate_voice_clone(**kwargs)


def _warmup_faster_streaming(model, args: argparse.Namespace) -> float:
    if args.backend != "faster" or not args.streaming or args.warmup_tokens <= 0:
        return 0.0

    kwargs = {
        "text": args.text[:50] if len(args.text) > 50 else args.text,
        "language": args.language,
        "max_new_tokens": args.warmup_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": not args.greedy,
        "repetition_penalty": args.repetition_penalty,
        "non_streaming_mode": args.non_streaming_mode,
    }
    if args.prompt_artifact:
        kwargs["voice_clone_prompt"] = _load_voice_prompt(Path(args.prompt_artifact), args.device_map)
    else:
        kwargs["ref_audio"] = args.reference_audio
        kwargs["ref_text"] = args.reference_text
        kwargs["xvec_only"] = args.xvec_only

    _sync_cuda()
    t0 = time.perf_counter()
    model.generate_voice_clone(**kwargs)
    _sync_cuda()
    return time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "encode_reference", "synth_preencoded"], default="full")
    parser.add_argument("--backend", choices=["official", "faster"], required=True)
    parser.add_argument("--repo", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-artifact", default="")
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--parity-mode", action="store_true")
    parser.add_argument("--warmup-tokens", type=int, default=20)
    parser.add_argument("--non-streaming-mode", action="store_true")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    _set_seed(args.seed)
    t_load0 = time.perf_counter()
    if args.backend == "official":
        model = _load_official(args)
    else:
        model = _load_faster(args)
    _sync_cuda()
    load_s = time.perf_counter() - t_load0

    if args.mode == "encode_reference":
        t0 = time.perf_counter()
        prompt = _create_voice_prompt(model, args)
        _sync_cuda()
        encode_s = time.perf_counter() - t0
        artifact = Path(args.prompt_artifact)
        if not artifact:
            raise ValueError("--prompt-artifact is required for --mode encode_reference")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch = _import_torch()
        torch.save(_move_prompt(prompt), str(artifact))
        result = {
            "backend": args.backend,
            "artifact": str(artifact),
            "load_seconds": load_s,
            "encode_seconds": encode_s,
            "wall_seconds": load_s + encode_s,
        }
        print("BENCHMARK_JSON " + json.dumps(result, sort_keys=True))
        return 0

    warmup_s = _warmup_faster_streaming(model, args)

    t0 = time.perf_counter()
    generated = _generate_voice_clone(model, args)
    if isinstance(generated, tuple) and len(generated) == 3:
        wavs, sample_rate, extra_metrics = generated
        synth_s = float(extra_metrics.get("synth_seconds", 0.0))
    else:
        wavs, sample_rate = generated
        _sync_cuda()
        synth_s = time.perf_counter() - t0
        extra_metrics = {}

    out = Path(args.output)
    _write_wav(out, wavs, int(sample_rate))
    duration_s = _audio_duration(out)

    result = {
        "backend": args.backend,
        "output": str(out),
        "load_seconds": load_s,
        "synth_seconds": synth_s,
        "wall_seconds": load_s + synth_s,
        "audio_seconds": duration_s,
        "rtf_audio_per_wall": duration_s / (load_s + synth_s) if load_s + synth_s > 0 else 0.0,
        "rtf_audio_per_synth": duration_s / synth_s if synth_s > 0 else 0.0,
        "sample_rate": int(sample_rate),
        "warmup_seconds": warmup_s,
    }
    result.update(extra_metrics)
    print("BENCHMARK_JSON " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
