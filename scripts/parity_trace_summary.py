#!/usr/bin/env python3
"""
Produce a compact JSON summary for two qwen3-tts parity traces.

The verbose debug trace report is useful while inspecting tensors by hand. This
script is meant for parity notes and lightweight local gates: it reports token
match counts, the first divergent frame/codebook, and logit evidence for that
step when both traces contain raw code-predictor logits.
"""

from __future__ import annotations

import argparse
import json
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


def load_entry(trace_dir: Path, entries: dict[str, dict[str, Any]], name: str) -> np.ndarray:
    meta = entries[name]
    arr = np.fromfile(trace_dir / name, dtype=dtype_to_np(meta["dtype"]))
    if arr.size != meta["count"]:
        raise ValueError(f"{name}: expected {meta['count']} values, got {arr.size}")
    if meta["shape"]:
        arr = arr.reshape(meta["shape"])
    return arr


def topk(logits: np.ndarray, n: int) -> list[dict[str, float | int]]:
    flat = logits.reshape(-1)
    n = max(1, min(n, flat.size))
    idx = np.argpartition(-flat, n - 1)[:n]
    idx = idx[np.argsort(-flat[idx])]
    return [{"token": int(i), "logit": float(flat[i])} for i in idx]


def token_rank(logits: np.ndarray, token: int) -> int:
    flat = logits.reshape(-1)
    target = flat[token]
    return int(1 + np.sum(flat > target))


def top1_margin(top: list[dict[str, float | int]]) -> float | None:
    if len(top) < 2:
        return None
    return float(top[0]["logit"] - top[1]["logit"])


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value_float = float(value)
    if not np.isfinite(value_float):
        return None
    return value_float


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


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    n = min(a.size, b.size)
    flat_a = a.reshape(-1)[:n]
    flat_b = b.reshape(-1)[:n]
    result: dict[str, Any] = {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "values_compared": int(n),
    }

    if np.issubdtype(a.dtype, np.integer) or np.issubdtype(b.dtype, np.integer):
        result["values_matching"] = int(np.sum(flat_a == flat_b))
        result["exact_match"] = bool(a.shape == b.shape and result["values_matching"] == n)
        return result

    finite = np.isfinite(flat_a) & np.isfinite(flat_b)
    if not np.any(finite):
        result.update({"cosine": float("nan"), "max_abs": float("nan"), "mean_abs": float("nan"), "rmse": float("nan")})
        return result

    diff = flat_a[finite].astype(np.float64) - flat_b[finite].astype(np.float64)
    result.update(
        {
            "cosine": cosine(flat_a, flat_b),
            "max_abs": float(np.max(np.abs(diff))),
            "mean_abs": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff * diff))),
        }
    )
    return result


def frame_ids(entries: dict[str, dict[str, Any]]) -> list[int]:
    frame_re = re.compile(r"^frame(\d+)_codec_tokens_cb0_15\.i32\.bin$")
    return sorted(
        int(m.group(1))
        for name in entries
        for m in [frame_re.match(name)]
        if m is not None
    )


def compare_tokens(trace_a: Path, entries_a: dict[str, Any], trace_b: Path, entries_b: dict[str, Any]) -> dict[str, Any]:
    frames = sorted(set(frame_ids(entries_a)) & set(frame_ids(entries_b)))
    total = 0
    matches = 0
    first_diff: dict[str, int] | None = None

    for frame in frames:
        name = f"frame{frame:03d}_codec_tokens_cb0_15.i32.bin"
        a = load_entry(trace_a, entries_a, name).reshape(-1)
        b = load_entry(trace_b, entries_b, name).reshape(-1)
        n = min(a.size, b.size)
        total += n
        matches += int(np.sum(a[:n] == b[:n]))
        if first_diff is None:
            for codebook in range(n):
                if int(a[codebook]) != int(b[codebook]):
                    first_diff = {
                        "frame": frame,
                        "codebook": codebook,
                        "token_a": int(a[codebook]),
                        "token_b": int(b[codebook]),
                    }
                    break

    return {
        "frames_compared": len(frames),
        "tokens_compared": total,
        "tokens_matching": matches,
        "match_percent": (100.0 * matches / total) if total else None,
        "first_diff": first_diff,
    }


def logit_tensor_name(frame: int, codebook: int) -> str:
    if codebook == 0:
        return f"frame{frame:03d}_cb0_logits_post_rules.f32.bin"
    return f"frame{frame:03d}_codepred_logits_step{codebook - 1:02d}.f32.bin"


def logit_evidence(
    trace_a: Path,
    entries_a: dict[str, Any],
    trace_b: Path,
    entries_b: dict[str, Any],
    first_diff: dict[str, int] | None,
    top_k: int,
) -> dict[str, Any] | None:
    if first_diff is None:
        return None
    codebook = first_diff["codebook"]
    step = 0 if codebook == 0 else codebook - 1
    name_a = logit_tensor_name(first_diff["frame"], codebook)
    name_b = name_a
    if name_a not in entries_a or name_b not in entries_b:
        return None

    a = load_entry(trace_a, entries_a, name_a)
    b = load_entry(trace_b, entries_b, name_b)
    finite = np.isfinite(a) & np.isfinite(b)
    max_abs = float(np.max(np.abs(a[finite] - b[finite]))) if np.any(finite) else float("nan")

    token_a = first_diff["token_a"]
    token_b = first_diff["token_b"]
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    top_a = topk(a, top_k)
    top_b = topk(b, top_k)
    return {
        "frame": first_diff["frame"],
        "codebook": codebook,
        "step": step,
        "tensor": name_a,
        "cosine": cosine(a, b),
        "max_abs": max_abs,
        "top_a": top_a,
        "top_b": top_b,
        "top1_margin_a": top1_margin(top_a),
        "top1_margin_b": top1_margin(top_b),
        "token_a_rank_in_b": token_rank(b, token_a),
        "token_b_rank_in_a": token_rank(a, token_b),
        "token_a_logit_a": float(flat_a[token_a]),
        "token_a_logit_b": float(flat_b[token_a]),
        "token_a_delta_b_minus_a": float(flat_b[token_a] - flat_a[token_a]),
        "token_b_logit_a": float(flat_a[token_b]),
        "token_b_logit_b": float(flat_b[token_b]),
        "token_b_delta_b_minus_a": float(flat_b[token_b] - flat_a[token_b]),
        "token_a_margin_over_token_b_in_a": float(flat_a[token_a] - flat_a[token_b]),
        "token_b_margin_over_token_a_in_b": float(flat_b[token_b] - flat_b[token_a]),
    }


def classify_first_diff(
    logits: dict[str, Any] | None,
    near_tie_margin: float,
    rank_threshold: int,
) -> dict[str, Any] | None:
    if logits is None:
        return None

    margin_a = finite_float(logits.get("top1_margin_a"))
    margin_b = finite_float(logits.get("top1_margin_b"))
    max_abs = finite_float(logits.get("max_abs"))
    token_a_rank_in_b = logits.get("token_a_rank_in_b")
    token_b_rank_in_a = logits.get("token_b_rank_in_a")
    token_a_rank_in_b = int(token_a_rank_in_b) if token_a_rank_in_b is not None else None
    token_b_rank_in_a = int(token_b_rank_in_a) if token_b_rank_in_a is not None else None

    margins = [margin for margin in (margin_a, margin_b) if margin is not None]
    min_margin = min(margins) if margins else None
    exact_tie_a = margin_a is not None and margin_a == 0.0
    exact_tie_b = margin_b is not None and margin_b == 0.0
    near_tie_a = margin_a is not None and margin_a <= near_tie_margin
    near_tie_b = margin_b is not None and margin_b <= near_tie_margin
    cross_rank_close = (
        token_a_rank_in_b is not None
        and token_b_rank_in_a is not None
        and token_a_rank_in_b <= rank_threshold
        and token_b_rank_in_a <= rank_threshold
    )

    if exact_tie_a or exact_tie_b:
        category = "exact_tie"
    elif cross_rank_close and (near_tie_a or near_tie_b):
        category = "near_tie_token_swap"
    elif near_tie_a or near_tie_b:
        category = "near_tie"
    elif cross_rank_close:
        category = "token_swap"
    else:
        category = "logit_drift"

    result: dict[str, Any] = {
        "category": category,
        "near_tie_margin": near_tie_margin,
        "rank_threshold": rank_threshold,
        "exact_tie_a": exact_tie_a,
        "exact_tie_b": exact_tie_b,
        "near_tie_a": near_tie_a,
        "near_tie_b": near_tie_b,
        "cross_rank_close": cross_rank_close,
        "min_top1_margin": min_margin,
    }
    if max_abs is not None:
        result["max_abs"] = max_abs
    if min_margin is not None and min_margin > 0.0 and max_abs is not None:
        result["max_abs_over_min_top1_margin"] = float(max_abs / min_margin)
    return result


def first_diff_step_trajectory(
    trace_a: Path,
    entries_a: dict[str, Any],
    trace_b: Path,
    entries_b: dict[str, Any],
    first_diff: dict[str, int] | None,
) -> list[dict[str, Any]] | None:
    if first_diff is None:
        return None

    codebook = first_diff["codebook"]
    frames = sorted(set(frame_ids(entries_a)) & set(frame_ids(entries_b)))
    rows: list[dict[str, Any]] = []

    for frame in frames:
        token_name = f"frame{frame:03d}_codec_tokens_cb0_15.i32.bin"
        logit_name = logit_tensor_name(frame, codebook)
        if (
            token_name not in entries_a
            or token_name not in entries_b
            or logit_name not in entries_a
            or logit_name not in entries_b
        ):
            continue

        tokens_a = load_entry(trace_a, entries_a, token_name).reshape(-1)
        tokens_b = load_entry(trace_b, entries_b, token_name).reshape(-1)
        if codebook >= tokens_a.size or codebook >= tokens_b.size:
            continue

        logits_a = load_entry(trace_a, entries_a, logit_name)
        logits_b = load_entry(trace_b, entries_b, logit_name)
        top_a = topk(logits_a, 2)
        top_b = topk(logits_b, 2)
        flat_a = logits_a.reshape(-1)
        flat_b = logits_b.reshape(-1)
        token_a = int(tokens_a[codebook])
        token_b = int(tokens_b[codebook])
        finite = np.isfinite(flat_a) & np.isfinite(flat_b)
        max_abs = float(np.max(np.abs(flat_a[finite] - flat_b[finite]))) if np.any(finite) else float("nan")

        rows.append(
            {
                "frame": frame,
                "codebook": codebook,
                "token_a": token_a,
                "token_b": token_b,
                "tokens_match": token_a == token_b,
                "logit_cosine": cosine(logits_a, logits_b),
                "logit_max_abs": max_abs,
                "top_token_a": int(top_a[0]["token"]),
                "top_token_b": int(top_b[0]["token"]),
                "top_tokens_match": int(top_a[0]["token"]) == int(top_b[0]["token"]),
                "top1_margin_a": top1_margin(top_a),
                "top1_margin_b": top1_margin(top_b),
                "token_a_rank_in_b": token_rank(logits_b, token_a),
                "token_b_rank_in_a": token_rank(logits_a, token_b),
                "token_a_logit_a": float(flat_a[token_a]),
                "token_a_logit_b": float(flat_b[token_a]),
                "token_b_logit_a": float(flat_a[token_b]),
                "token_b_logit_b": float(flat_b[token_b]),
            }
        )

    return rows


def row_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    value_float = float(value)
    if not np.isfinite(value_float):
        return None
    return value_float


def metric_extreme(rows: list[dict[str, Any]], key: str, mode: str) -> dict[str, Any] | None:
    values = [(row, row_metric(row, key)) for row in rows]
    values = [(row, value) for row, value in values if value is not None]
    if not values:
        return None
    if mode == "min":
        row, value = min(values, key=lambda item: item[1])
    elif mode == "max":
        row, value = max(values, key=lambda item: item[1])
    else:
        raise ValueError(f"unsupported extreme mode: {mode}")
    return {"frame": int(row["frame"]), "value": float(value)}


def first_matching_row(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any] | None:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


def summarize_first_diff_step_trajectory(
    rows: list[dict[str, Any]] | None,
    first_diff: dict[str, int] | None,
) -> dict[str, Any] | None:
    if rows is None or first_diff is None:
        return None

    first_diff_frame = first_diff["frame"]
    first_diff_row = first_matching_row(rows, "frame", first_diff_frame)
    prior_rows = [row for row in rows if int(row["frame"]) < first_diff_frame]
    first_token_disagreement = next((row for row in rows if not row["tokens_match"]), None)
    first_top_disagreement = next((row for row in rows if not row["top_tokens_match"]), None)

    summary: dict[str, Any] = {
        "frames_with_step_compared": len(rows),
        "prior_frames_compared": len(prior_rows),
        "prior_frames_with_matching_tokens": int(sum(1 for row in prior_rows if row["tokens_match"])),
        "prior_frames_with_matching_top_tokens": int(sum(1 for row in prior_rows if row["top_tokens_match"])),
        "first_token_disagreement_frame": int(first_token_disagreement["frame"]) if first_token_disagreement else None,
        "first_top_disagreement_frame": int(first_top_disagreement["frame"]) if first_top_disagreement else None,
        "min_prior_top1_margin_a": metric_extreme(prior_rows, "top1_margin_a", "min"),
        "min_prior_top1_margin_b": metric_extreme(prior_rows, "top1_margin_b", "min"),
        "max_prior_logit_max_abs": metric_extreme(prior_rows, "logit_max_abs", "max"),
        "min_all_top1_margin_a": metric_extreme(rows, "top1_margin_a", "min"),
        "min_all_top1_margin_b": metric_extreme(rows, "top1_margin_b", "min"),
        "max_all_logit_max_abs": metric_extreme(rows, "logit_max_abs", "max"),
    }

    if first_diff_row is not None:
        summary["first_diff_row"] = {
            "frame": int(first_diff_row["frame"]),
            "token_a": int(first_diff_row["token_a"]),
            "token_b": int(first_diff_row["token_b"]),
            "top1_margin_a": row_metric(first_diff_row, "top1_margin_a"),
            "top1_margin_b": row_metric(first_diff_row, "top1_margin_b"),
            "logit_max_abs": row_metric(first_diff_row, "logit_max_abs"),
            "logit_cosine": row_metric(first_diff_row, "logit_cosine"),
        }

        min_prior_margin_a = summary["min_prior_top1_margin_a"]
        min_prior_margin_b = summary["min_prior_top1_margin_b"]
        max_prior_abs = summary["max_prior_logit_max_abs"]
        margin_a = row_metric(first_diff_row, "top1_margin_a")
        margin_b = row_metric(first_diff_row, "top1_margin_b")
        max_abs = row_metric(first_diff_row, "logit_max_abs")
        if margin_a is not None and min_prior_margin_a is not None:
            summary["first_diff_margin_a_vs_min_prior"] = float(margin_a / min_prior_margin_a["value"])
        if margin_b is not None and min_prior_margin_b is not None:
            summary["first_diff_margin_b_vs_min_prior"] = float(margin_b / min_prior_margin_b["value"])
        if max_abs is not None and max_prior_abs is not None:
            summary["first_diff_logit_max_abs_vs_max_prior"] = float(max_abs / max_prior_abs["value"])

    return summary


def boundary_evidence(
    trace_a: Path,
    entries_a: dict[str, Any],
    trace_b: Path,
    entries_b: dict[str, Any],
    first_diff: dict[str, int] | None,
) -> dict[str, Any] | None:
    if first_diff is None:
        return None

    frame = first_diff["frame"]
    candidates = [
        f"frame{frame:03d}_talker_hidden.f32.bin",
        f"frame{frame:03d}_codepred_input_hidden.f32.bin",
        f"frame{frame:03d}_codepred_input_cb0_embd.f32.bin",
        f"frame{frame:03d}_codepred_prefill_input.f32.bin",
        f"frame{frame:03d}_codepred_prefill_projected.f32.bin",
        f"frame{frame:03d}_codepred_prefill_pos.i32.bin",
        f"frame{frame:03d}_codepred_prefill_mask.f32.bin",
    ]

    tensors: dict[str, Any] = {}
    for name in candidates:
        if name not in entries_a or name not in entries_b:
            continue
        a = load_entry(trace_a, entries_a, name)
        b = load_entry(trace_b, entries_b, name)
        tensors[name] = compare_arrays(a, b)

    return tensors


def codepred_layer_evidence(
    trace_a: Path,
    entries_a: dict[str, Any],
    trace_b: Path,
    entries_b: dict[str, Any],
    first_diff: dict[str, int] | None,
) -> dict[str, Any] | None:
    if first_diff is None or first_diff["codebook"] == 0:
        return None

    frame = first_diff["frame"]
    step = first_diff["codebook"] - 1
    if step == 0:
        prefix = f"frame{frame:03d}_codepred_prefill"
        candidates = [f"{prefix}_projected.f32.bin"]
    else:
        prefix = f"frame{frame:03d}_codepred_step{step:02d}"
        candidates = [f"{prefix}_projected.f32.bin"]

    for layer_idx in range(64):
        for suffix in ("attn_norm", "attn_out", "ffn_norm", "ffn_out", "hidden"):
            candidates.append(f"{prefix}_layer{layer_idx:02d}_{suffix}.f32.bin")
    candidates.append(f"{prefix}_final_hidden.f32.bin")

    tensors: dict[str, Any] = {}
    for name in candidates:
        if name not in entries_a or name not in entries_b:
            continue
        a = load_entry(trace_a, entries_a, name)
        b = load_entry(trace_b, entries_b, name)
        tensors[name] = compare_arrays(a, b)

    return {
        "frame": frame,
        "codebook": first_diff["codebook"],
        "step": step,
        "tensors": tensors,
    }


def check_expectations(result: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    tokens = result["tokens"]
    first_diff = tokens["first_diff"]
    logits = result["logits_at_first_diff"]

    def add_equal(label: str, actual: Any, expected: Any) -> None:
        if expected is not None and actual != expected:
            failures.append(f"{label}: expected {expected}, got {actual}")

    if args.expect_match_percent_at_least is not None:
        match_percent = tokens["match_percent"]
        if match_percent is None or match_percent < args.expect_match_percent_at_least:
            failures.append(
                f"match_percent: expected >= {args.expect_match_percent_at_least}, got {match_percent}"
            )

    if any(
        value is not None
        for value in (
            args.expect_first_diff_frame,
            args.expect_first_diff_codebook,
            args.expect_first_diff_token_a,
            args.expect_first_diff_token_b,
        )
    ):
        if first_diff is None:
            failures.append("first_diff: expected a mismatch, got exact token match")
        else:
            add_equal("first_diff.frame", first_diff["frame"], args.expect_first_diff_frame)
            add_equal("first_diff.codebook", first_diff["codebook"], args.expect_first_diff_codebook)
            add_equal("first_diff.token_a", first_diff["token_a"], args.expect_first_diff_token_a)
            add_equal("first_diff.token_b", first_diff["token_b"], args.expect_first_diff_token_b)

    if args.expect_first_diff_cosine_at_least is not None:
        if logits is None:
            failures.append("logits_at_first_diff: missing; cannot check cosine")
        elif logits["cosine"] < args.expect_first_diff_cosine_at_least:
            failures.append(
                f"first_diff cosine: expected >= {args.expect_first_diff_cosine_at_least}, got {logits['cosine']}"
            )

    if args.expect_first_diff_max_abs_at_most is not None:
        if logits is None:
            failures.append("logits_at_first_diff: missing; cannot check max_abs")
        elif logits["max_abs"] > args.expect_first_diff_max_abs_at_most:
            failures.append(
                f"first_diff max_abs: expected <= {args.expect_first_diff_max_abs_at_most}, got {logits['max_abs']}"
            )

    if args.expect_first_diff_category is not None:
        classification = result.get("first_diff_classification")
        if classification is None:
            failures.append("first_diff_classification: missing; cannot check category")
        else:
            add_equal(
                "first_diff_classification.category",
                classification.get("category"),
                args.expect_first_diff_category,
            )

    if args.expect_first_diff_max_abs_over_margin_at_least is not None:
        classification = result.get("first_diff_classification")
        if classification is None:
            failures.append("first_diff_classification: missing; cannot check max_abs_over_min_top1_margin")
        else:
            ratio = classification.get("max_abs_over_min_top1_margin")
            if ratio is None:
                failures.append("first_diff max_abs_over_min_top1_margin: missing")
            elif ratio < args.expect_first_diff_max_abs_over_margin_at_least:
                failures.append(
                    "first_diff max_abs_over_min_top1_margin: "
                    f"expected >= {args.expect_first_diff_max_abs_over_margin_at_least}, got {ratio}"
                )

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize token and logit parity between two traces")
    parser.add_argument("--trace-a", required=True, type=Path)
    parser.add_argument("--trace-b", required=True, type=Path)
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--near-tie-margin", type=float, default=0.02)
    parser.add_argument("--near-tie-rank-threshold", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expect-match-percent-at-least", type=float, default=None)
    parser.add_argument("--expect-first-diff-frame", type=int, default=None)
    parser.add_argument("--expect-first-diff-codebook", type=int, default=None)
    parser.add_argument("--expect-first-diff-token-a", type=int, default=None)
    parser.add_argument("--expect-first-diff-token-b", type=int, default=None)
    parser.add_argument("--expect-first-diff-cosine-at-least", type=float, default=None)
    parser.add_argument("--expect-first-diff-max-abs-at-most", type=float, default=None)
    parser.add_argument(
        "--expect-first-diff-category",
        choices=["exact_tie", "near_tie_token_swap", "near_tie", "token_swap", "logit_drift"],
        default=None,
    )
    parser.add_argument("--expect-first-diff-max-abs-over-margin-at-least", type=float, default=None)
    args = parser.parse_args()

    entries_a = load_manifest(args.trace_a)
    entries_b = load_manifest(args.trace_b)
    token_summary = compare_tokens(args.trace_a, entries_a, args.trace_b, entries_b)
    trajectory = first_diff_step_trajectory(
        args.trace_a,
        entries_a,
        args.trace_b,
        entries_b,
        token_summary["first_diff"],
    )
    logits = logit_evidence(
        args.trace_a,
        entries_a,
        args.trace_b,
        entries_b,
        token_summary["first_diff"],
        args.top_k,
    )
    result = {
        "trace_a": str(args.trace_a),
        "trace_b": str(args.trace_b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "tokens": token_summary,
        "logits_at_first_diff": logits,
        "first_diff_classification": classify_first_diff(
            logits,
            args.near_tie_margin,
            args.near_tie_rank_threshold,
        ),
        "boundary_tensors_at_first_diff": boundary_evidence(
            args.trace_a,
            entries_a,
            args.trace_b,
            entries_b,
            token_summary["first_diff"],
        ),
        "codepred_layer_tensors_at_first_diff": codepred_layer_evidence(
            args.trace_a,
            entries_a,
            args.trace_b,
            entries_b,
            token_summary["first_diff"],
        ),
        "first_diff_step_trajectory_summary": summarize_first_diff_step_trajectory(
            trajectory,
            token_summary["first_diff"],
        ),
        "first_diff_step_trajectory": trajectory,
    }

    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)

    failures = check_expectations(result, args)
    if failures:
        for failure in failures:
            print(f"EXPECTATION FAILED: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
