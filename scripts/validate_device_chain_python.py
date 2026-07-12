#!/usr/bin/env python3
"""Validate automatic C++ Code Predictor paths against legacy C++ and official Python.

The hard optimization gate is byte identity between the automatic and legacy
C++ paths. The official float32 Python run adds an independent model-parity
gate for token structure, early-token agreement, audio validity, and duration.
Full Python/C++ waveform or token identity is not required because recurrent
F16/Q8 inference can diverge after near-tied logits.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path | None) -> str | None:
    if not path:
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _hf_revision(path: Path) -> str | None:
    return path.name if path.parent.name == "snapshots" else None


def _relevant_environment() -> dict[str, str]:
    prefixes = ("QWEN3_TTS_", "GGML_CUDA_")
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes)
    }


def _slug(value: str) -> str:
    result = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in result.split("_") if part)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_code_json(path: Path, codes: np.ndarray) -> None:
    payload = {
        "frames": int(codes.shape[0]),
        "codebooks": int(codes.shape[1]),
        "codes": codes.astype(np.int64, copy=False).reshape(-1).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_code_json(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        result = np.asarray(payload, dtype=np.int64)
    else:
        frames = int(payload["frames"])
        codebooks = int(payload["codebooks"])
        result = np.asarray(payload["codes"], dtype=np.int64).reshape(
            frames, codebooks
        )
    if result.ndim != 2:
        raise ValueError(f"Expected 2-D codes in {path}, got {result.shape}")
    return result


def _compare_codes(python_codes: np.ndarray, cpp_codes: np.ndarray) -> dict[str, Any]:
    frames = min(python_codes.shape[0], cpp_codes.shape[0])
    codebooks = min(python_codes.shape[1], cpp_codes.shape[1])
    py = python_codes[:frames, :codebooks]
    cpp = cpp_codes[:frames, :codebooks]
    equal = py == cpp

    first_diff = None
    if equal.size and not bool(equal.all()):
        frame, codebook = np.argwhere(~equal)[0]
        first_diff = {
            "frame": int(frame),
            "codebook": int(codebook),
            "python": int(py[frame, codebook]),
            "cpp": int(cpp[frame, codebook]),
        }

    exact_prefix_frames = 0
    for frame in range(frames):
        if bool(equal[frame].all()):
            exact_prefix_frames += 1
        else:
            break

    return {
        "python_shape": list(python_codes.shape),
        "cpp_shape": list(cpp_codes.shape),
        "common_shape": [frames, codebooks],
        "match_ratio": float(equal.mean()) if equal.size else 0.0,
        "cb0_match_ratio": float(equal[:, 0].mean()) if frames else 0.0,
        "first_frame_match_ratio": float(equal[0].mean()) if frames else 0.0,
        "exact_prefix_frames": exact_prefix_frames,
        "python_fully_compared": frames == python_codes.shape[0]
        and codebooks == python_codes.shape[1],
        "cpp_extra_frames": max(0, int(cpp_codes.shape[0] - python_codes.shape[0])),
        "first_diff": first_diff,
    }


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.reshape(-1), int(sample_rate)


def _compare_audio(python_wav: Path, cpp_wav: Path) -> dict[str, Any]:
    py, py_rate = _load_audio(python_wav)
    cpp, cpp_rate = _load_audio(cpp_wav)
    common = min(py.size, cpp.size)
    py_rms = float(np.sqrt(np.mean(np.square(py)))) if py.size else 0.0
    cpp_rms = float(np.sqrt(np.mean(np.square(cpp)))) if cpp.size else 0.0
    duration_ratio = float(cpp.size / py.size) if py.size else 0.0

    correlation = 0.0
    if common > 1:
        py_common = py[:common]
        cpp_common = cpp[:common]
        if float(py_common.std()) > 1e-10 and float(cpp_common.std()) > 1e-10:
            correlation = float(np.corrcoef(py_common, cpp_common)[0, 1])

    return {
        "python_samples": int(py.size),
        "cpp_samples": int(cpp.size),
        "python_sample_rate": py_rate,
        "cpp_sample_rate": cpp_rate,
        "python_rms": py_rms,
        "cpp_rms": cpp_rms,
        "duration_ratio": duration_ratio,
        "correlation_informational": correlation,
    }


def _run_cpp(
    args: argparse.Namespace,
    model_name: str,
    speaker_path: Path,
    length: int,
    mode: str,
    output_base: Path,
    reference_codes_path: Path | None,
) -> dict[str, Any]:
    generated_path = output_base.with_name(output_base.name + "_generated.json")
    decoder_path = output_base.with_name(output_base.name + "_decoder.json")
    wav_path = output_base.with_suffix(".wav")
    env = os.environ.copy()
    if mode == "legacy":
        env["QWEN3_TTS_CODE_PRED_DEVICE_CHAIN"] = "0"
        env["QWEN3_TTS_CODE_PRED_SUPERGRAPH"] = "0"
    else:
        env.pop("QWEN3_TTS_CODE_PRED_DEVICE_CHAIN", None)
        env.pop("QWEN3_TTS_CODE_PRED_SUPERGRAPH", None)

    temperature = "1.0" if args.decode_mode == "topk1" else "0"
    top_k = "1" if args.decode_mode == "topk1" else "0"
    cmd = [
        str(args.cpp_cli),
        "--model",
        str(args.cpp_model_dir),
        "--model-name",
        model_name,
        "--codec-model",
        str(args.codec_model),
        "--text",
        args.text,
        "--speaker-embedding",
        str(speaker_path),
        "--language",
        args.language_cpp,
        "--temperature",
        temperature,
        "--top-k",
        top_k,
        "--top-p",
        "1.0",
        "--seed",
        str(args.seed),
        "--max-tokens",
        str(length),
        "--dump-generated-codes",
        str(generated_path),
        "--dump-decoder-codes",
        str(decoder_path),
        "--output",
        str(wav_path),
    ]
    if reference_codes_path is not None:
        cmd.extend(
            [
                "--reference-codes",
                str(reference_codes_path),
                "--reference-text-file",
                str(args.reference_text_file),
            ]
        )
    completed = subprocess.run(
        cmd,
        cwd=str(args.repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=args.cpp_timeout,
    )
    log_path = output_base.with_suffix(".log")
    log_path.write_text(
        "COMMAND: " + subprocess.list2cmdline(cmd) + "\n\n"
        + completed.stdout
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"C++ {mode} run failed; see {log_path}")

    combined_log = completed.stdout + completed.stderr
    backend = None
    hidden_size = None
    for line in combined_log.splitlines():
        if "TTSTransformer backend:" in line:
            backend = line.split("TTSTransformer backend:", 1)[1].strip()
        if "TTS transformer loaded: hidden_size=" in line:
            value = line.split("hidden_size=", 1)[1].split(",", 1)[0]
            hidden_size = int(value)
    return {
        "generated": generated_path,
        "decoder": decoder_path,
        "wav": wav_path,
        "log": log_path,
        "device_chain_active": "CodePred device chain: enabled" in combined_log,
        "supergraph_active": "CodePred supergraph: enabled" in combined_log,
        "transformer_backend": backend,
        "transformer_hidden_size": hidden_size,
    }


def _run_cpp_benchmark(
    args: argparse.Namespace,
    model_name: str,
    speaker_path: Path,
    length: int,
    mode: str,
    output_base: Path,
    reference_codes_path: Path | None,
) -> dict[str, Any]:
    env = os.environ.copy()
    if mode == "legacy":
        env["QWEN3_TTS_CODE_PRED_DEVICE_CHAIN"] = "0"
        env["QWEN3_TTS_CODE_PRED_SUPERGRAPH"] = "0"
    else:
        env.pop("QWEN3_TTS_CODE_PRED_DEVICE_CHAIN", None)
        env.pop("QWEN3_TTS_CODE_PRED_SUPERGRAPH", None)
    temperature = "1.0" if args.decode_mode == "topk1" else "0"
    top_k = "1" if args.decode_mode == "topk1" else "0"
    cmd = [
        str(args.cpp_cli),
        "--model",
        str(args.cpp_model_dir),
        "--model-name",
        model_name,
        "--codec-model",
        str(args.codec_model),
        "--text",
        args.text,
        "--speaker-embedding",
        str(speaker_path),
        "--language",
        args.language_cpp,
        "--temperature",
        temperature,
        "--top-k",
        top_k,
        "--top-p",
        "1.0",
        "--seed",
        str(args.seed),
        "--max-tokens",
        str(length),
        "--bench-warmup",
        str(args.benchmark_warmups),
        "--bench-server",
        str(args.benchmark_runs),
        "--output",
        str(output_base.with_suffix(".wav")),
    ]
    if reference_codes_path is not None:
        cmd.extend(
            [
                "--reference-codes",
                str(reference_codes_path),
                "--reference-text-file",
                str(args.reference_text_file),
            ]
        )
    completed = subprocess.run(
        cmd,
        cwd=str(args.repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=args.cpp_timeout,
    )
    log_path = output_base.with_suffix(".log")
    combined_log = completed.stdout + completed.stderr
    log_path.write_text(
        "COMMAND: " + subprocess.list2cmdline(cmd) + "\n\n" + combined_log,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"C++ {mode} benchmark failed; see {log_path}")

    rows = []
    for line in combined_log.splitlines():
        if line.startswith("BENCH_JSON "):
            payload = json.loads(line[len("BENCH_JSON ") :])
            if not payload.get("warmup", False):
                rows.append(payload)
    if len(rows) != args.benchmark_runs:
        raise RuntimeError(
            f"Expected {args.benchmark_runs} measured rows, got {len(rows)} in {log_path}"
        )
    return {
        "runs": len(rows),
        "generation_median_ms": float(
            statistics.median(float(row["generate_ms"]) for row in rows)
        ),
        "wall_median_ms": float(
            statistics.median(float(row["wall_ms"]) for row in rows)
        ),
        "device_chain_active": "CodePred device chain: enabled" in combined_log,
        "supergraph_active": "CodePred supergraph: enabled" in combined_log,
        "log": str(log_path),
    }


def _load_official_model(args: argparse.Namespace):
    if args.python_repo:
        sys.path.insert(0, str(args.python_repo.resolve()))
    from qwen_tts import Qwen3TTSModel

    dtype = torch.float32
    return Qwen3TTSModel.from_pretrained(
        str(args.hf_model),
        device_map=args.python_device,
        dtype=dtype,
        attn_implementation="eager",
    )


def _create_python_prompt(model, args: argparse.Namespace):
    use_icl = args.prompt_mode == "icl"
    if args.python_speaker_embedding:
        speaker = json.loads(args.python_speaker_embedding.read_text(encoding="utf-8"))
        speaker_tensor = torch.tensor(
            speaker, device=model.device, dtype=torch.float32
        ).reshape(-1)
        reference_codes = None
        if use_icl:
            if not args.python_reference_codes:
                raise ValueError(
                    "--python-reference-codes is required with an external ICL speaker"
                )
            reference_codes = _load_code_json(args.python_reference_codes)
            reference_tensor = torch.tensor(
                reference_codes, device=model.device, dtype=torch.long
            )
        else:
            reference_tensor = None
        prompt = {
            "ref_code": [reference_tensor],
            "ref_spk_embedding": [speaker_tensor],
            "x_vector_only_mode": [not use_icl],
            "icl_mode": [use_icl],
        }
        return prompt, speaker, reference_codes

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=str(args.reference_audio),
        ref_text=args.reference_text,
        x_vector_only_mode=not use_icl,
    )
    prompt = model._prompt_items_to_voice_clone_prompt(prompt_items)
    speaker = (
        prompt_items[0]
        .ref_spk_embedding.detach()
        .cpu()
        .to(torch.float32)
        .reshape(-1)
        .tolist()
    )
    reference_codes = None
    if use_icl:
        reference_codes = (
            prompt_items[0]
            .ref_code.detach()
            .cpu()
            .to(torch.int64)
            .numpy()
        )
    return prompt, speaker, reference_codes


def _run_python_case(
    model,
    prompt,
    args: argparse.Namespace,
    length: int,
    output_dir: Path,
) -> tuple[np.ndarray, Path]:
    _set_seed(args.seed)
    assistant_text = model._build_assistant_text(args.text)
    input_ids = model._tokenize_texts([assistant_text])
    ref_ids = None
    if args.prompt_mode == "icl":
        ref_ids = [model._tokenize_texts([model._build_ref_text(args.reference_text)])[0]]
    topk1 = args.decode_mode == "topk1"
    with torch.no_grad():
        code_list, _ = model.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=prompt,
            languages=[args.language_python],
            non_streaming_mode=False,
            max_new_tokens=length,
            do_sample=topk1,
            top_k=1 if topk1 else None,
            top_p=1.0 if topk1 else None,
            temperature=1.0 if topk1 else None,
            repetition_penalty=1.05,
            subtalker_dosample=False,
            subtalker_top_k=None,
            subtalker_top_p=None,
            subtalker_temperature=None,
        )
    codes = code_list[0].detach().cpu().to(torch.int64).numpy()
    decode_codes = code_list[0]
    reference_frame_count = 0
    if args.prompt_mode == "icl" and prompt["ref_code"][0] is not None:
        reference = prompt["ref_code"][0].to(decode_codes.device)
        reference_frame_count = int(reference.shape[0])
        decode_codes = torch.cat([reference, decode_codes], dim=0)
    wavs, sample_rate = model.model.speech_tokenizer.decode(
        [{"audio_codes": decode_codes}]
    )
    wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    if reference_frame_count:
        cut = int(reference_frame_count / max(int(decode_codes.shape[0]), 1) * wav.size)
        wav = wav[cut:]
    codes_path = output_dir / f"python_{length}f_codes.json"
    wav_path = output_dir / f"python_{length}f.wav"
    _write_code_json(codes_path, codes)
    sf.write(str(wav_path), wav, int(sample_rate))
    return codes, wav_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cpp-cli", type=Path, required=True)
    parser.add_argument("--cpp-model-dir", type=Path, required=True)
    parser.add_argument("--cpp-model", action="append", required=True)
    parser.add_argument("--codec-model", type=Path, required=True)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--python-repo", type=Path)
    parser.add_argument("--python-device", default="cuda:0")
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text-file", type=Path, required=True)
    parser.add_argument("--python-speaker-embedding", type=Path)
    parser.add_argument("--python-reference-codes", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", default="Hello. This is a deterministic parity check.")
    parser.add_argument("--length", type=int, action="append", required=True)
    parser.add_argument("--language-python", default="English")
    parser.add_argument("--language-cpp", default="en")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-mode", choices=("speaker-only", "icl"), default="speaker-only"
    )
    parser.add_argument("--decode-mode", choices=("greedy", "topk1"), default="greedy")
    parser.add_argument("--require-exact-python-codes", action="store_true")
    parser.add_argument("--cpp-timeout", type=int, default=300)
    parser.add_argument("--benchmark-warmups", type=int, default=0)
    parser.add_argument("--benchmark-runs", type=int, default=0)
    parser.add_argument("--min-code-match", type=float, default=0.05)
    parser.add_argument("--min-first-frame-match", type=float, default=0.50)
    parser.add_argument("--min-rms", type=float, default=0.001)
    parser.add_argument("--min-duration-ratio", type=float, default=0.50)
    parser.add_argument("--max-duration-ratio", type=float, default=1.50)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.cpp_cli = args.cpp_cli.resolve()
    args.cpp_model_dir = args.cpp_model_dir.resolve()
    args.codec_model = args.codec_model.resolve()
    args.hf_model = args.hf_model.resolve()
    args.reference_audio = args.reference_audio.resolve()
    args.reference_text_file = args.reference_text_file.resolve()
    if args.python_speaker_embedding:
        args.python_speaker_embedding = args.python_speaker_embedding.resolve()
    if args.python_reference_codes:
        args.python_reference_codes = args.python_reference_codes.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.python_repo:
        args.python_repo = args.python_repo.resolve()
    args.reference_text = args.reference_text_file.read_text(encoding="utf-8").strip()
    return args


def main() -> int:
    args = _parse_args()
    required = [
        args.cpp_cli,
        args.cpp_model_dir,
        args.codec_model,
        args.hf_model,
        args.reference_audio,
        args.reference_text_file,
    ]
    if args.python_speaker_embedding:
        required.append(args.python_speaker_embedding)
    if args.python_reference_codes:
        required.append(args.python_reference_codes)
    missing = [str(path) for path in required if not path.exists()]
    for model_name in args.cpp_model:
        model_path = args.cpp_model_dir / model_name
        if not model_path.exists():
            missing.append(str(model_path))
    if missing:
        raise FileNotFoundError("Missing validation inputs: " + ", ".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    print(f"Loading official Python model: {args.hf_model}")
    model = _load_official_model(args)
    model.model = model.model.eval()
    prompt, speaker_values, reference_codes = _create_python_prompt(model, args)
    speaker_path = args.output_dir / "python_speaker_embedding.json"
    speaker_path.write_text(json.dumps(speaker_values) + "\n", encoding="utf-8")

    reference_codes_path = None
    if reference_codes is not None:
        reference_codes_path = args.output_dir / "python_reference_codes.json"
        _write_code_json(reference_codes_path, reference_codes)

    python_cases: dict[int, tuple[np.ndarray, Path]] = {}
    for length in args.length:
        print(f"Python reference: max_tokens={length}")
        python_cases[length] = _run_python_case(
            model, prompt, args, length, args.output_dir
        )

    # The 1.7B F32 Python and C++ models do not fit concurrently on common GPUs.
    # Persist the independent reference first, then release all Python model state.
    del prompt
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    all_pass = True
    for length in args.length:
        python_codes, python_wav = python_cases[length]
        for model_name in args.cpp_model:
            model_slug = _slug(Path(model_name).stem)
            runs = {}
            for mode in ("legacy", "automatic"):
                base = args.output_dir / f"{model_slug}_{length}f_{mode}"
                print(f"C++ {mode}: model={model_name} max_tokens={length}")
                runs[mode] = _run_cpp(
                    args,
                    model_name,
                    speaker_path,
                    length,
                    mode,
                    base,
                    reference_codes_path,
                )

            auto = runs["automatic"]
            legacy = runs["legacy"]
            cpp_exact = {
                key: _sha256(auto[key]) == _sha256(legacy[key])
                for key in ("generated", "decoder", "wav")
            }
            cpp_codes = _load_code_json(auto["generated"])
            code_metrics = _compare_codes(python_codes, cpp_codes)
            audio_metrics = _compare_audio(python_wav, auto["wav"])
            expect_optimized = (
                args.decode_mode == "greedy"
                and auto["transformer_backend"] is not None
                and "CUDA" in auto["transformer_backend"]
            )
            optimized_active = auto["supergraph_active"] or auto["device_chain_active"]

            performance = None
            if args.benchmark_runs > 0:
                benchmark = {}
                for mode in ("legacy", "automatic"):
                    benchmark_base = args.output_dir / (
                        f"{model_slug}_{length}f_{mode}_benchmark"
                    )
                    print(
                        f"C++ {mode} benchmark: model={model_name} "
                        f"max_tokens={length}"
                    )
                    benchmark[mode] = _run_cpp_benchmark(
                        args,
                        model_name,
                        speaker_path,
                        length,
                        mode,
                        benchmark_base,
                        reference_codes_path,
                    )
                legacy_ms = benchmark["legacy"]["generation_median_ms"]
                automatic_ms = benchmark["automatic"]["generation_median_ms"]
                performance = {
                    "legacy": benchmark["legacy"],
                    "automatic": benchmark["automatic"],
                    "generation_gain_percent": (
                        100.0 * (legacy_ms - automatic_ms) / legacy_ms
                        if legacy_ms > 0.0
                        else 0.0
                    ),
                }

            gates = {
                "cpp_auto_legacy_exact": all(cpp_exact.values()),
                "automatic_dispatch": optimized_active == expect_optimized,
                "python_cpp_codebooks": code_metrics["common_shape"][1] == 16,
                "python_cpp_nonempty": code_metrics["common_shape"][0] > 0,
                "python_cpp_code_match": code_metrics["match_ratio"]
                >= args.min_code_match,
                "python_cpp_exact_common_trajectory": (
                    code_metrics["match_ratio"] == 1.0
                    and code_metrics["python_fully_compared"]
                    and code_metrics["cpp_shape"][1] == code_metrics["python_shape"][1]
                    and code_metrics["python_shape"][0]
                    <= code_metrics["cpp_shape"][0]
                    <= code_metrics["python_shape"][0] + 1
                )
                if args.require_exact_python_codes
                else True,
                "python_cpp_first_frame": code_metrics["first_frame_match_ratio"]
                >= args.min_first_frame_match,
                "python_cpp_sample_rate": audio_metrics["python_sample_rate"]
                == audio_metrics["cpp_sample_rate"],
                "python_audio_non_silent": audio_metrics["python_rms"] >= args.min_rms,
                "cpp_audio_non_silent": audio_metrics["cpp_rms"] >= args.min_rms,
                "duration_ratio": args.min_duration_ratio
                <= audio_metrics["duration_ratio"]
                <= args.max_duration_ratio,
            }
            passed = all(gates.values())
            all_pass &= passed
            row = {
                "model": model_name,
                "max_tokens": length,
                "cpp_exact": cpp_exact,
                "automatic_device_chain_active": auto["device_chain_active"],
                "automatic_supergraph_active": auto["supergraph_active"],
                "automatic_optimized_path_active": optimized_active,
                "automatic_transformer_backend": auto["transformer_backend"],
                "automatic_transformer_hidden_size": auto["transformer_hidden_size"],
                "expected_optimized_path_active": expect_optimized,
                "python_cpp_codes": code_metrics,
                "python_cpp_audio": audio_metrics,
                "cpp_performance": performance,
                "gates": gates,
                "passed": passed,
            }
            rows.append(row)
            print(
                f"  {'PASS' if passed else 'FAIL'} "
                f"code_match={100.0 * code_metrics['match_ratio']:.2f}% "
                f"first_frame={100.0 * code_metrics['first_frame_match_ratio']:.2f}% "
                f"prefix={code_metrics['exact_prefix_frames']} "
                f"duration_ratio={audio_metrics['duration_ratio']:.3f}"
            )

    report = {
        "schema_version": 3,
        "provenance": {
            "repo_revision": _git_revision(args.repo_root),
            "python_repo_revision": _git_revision(args.python_repo),
            "hf_snapshot_revision": _hf_revision(args.hf_model),
            "hf_config_sha256": _sha256(args.hf_model / "config.json")
            if (args.hf_model / "config.json").exists()
            else None,
            "environment": _relevant_environment(),
        },
        "official_python": {
            "model": str(args.hf_model),
            "device": args.python_device,
            "dtype": "float32",
        },
        "cpp": {
            "binary": str(args.cpp_cli),
            "binary_sha256": _sha256(args.cpp_cli),
            "models": [
                {
                    "name": model_name,
                    "sha256": _sha256(args.cpp_model_dir / model_name),
                }
                for model_name in args.cpp_model
            ],
            "codec_model": str(args.codec_model),
            "codec_sha256": _sha256(args.codec_model),
        },
        "text": args.text,
        "prompt_mode": args.prompt_mode,
        "decode_mode": args.decode_mode,
        "require_exact_python_codes": args.require_exact_python_codes,
        "exact_gate_semantics": (
            "Require every Python frame/codebook to match; C++ may contain the "
            "known max_new_tokens extra frame, which is reported but not compared."
        ),
        "lengths": args.length,
        "thresholds": {
            "min_code_match": args.min_code_match,
            "min_first_frame_match": args.min_first_frame_match,
            "min_rms": args.min_rms,
            "duration_ratio": [args.min_duration_ratio, args.max_duration_ratio],
        },
        "benchmark": {
            "warmups": args.benchmark_warmups,
            "runs": args.benchmark_runs,
            "performance_is_informational": True,
        },
        "rows": rows,
        "passed": all_pass,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    print("Overall: " + ("PASS" if all_pass else "FAIL"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
