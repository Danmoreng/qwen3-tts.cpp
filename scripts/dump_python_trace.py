#!/usr/bin/env python3
"""
Dump Python (qwen_tts) generation traces in the same file layout as C++ debug dumps.

Current scope:
- CustomVoice path (1.7B) with `--speaker`
- Base path with an external `--speaker-embedding`
- Optional ICL reference text/codes path
- Non-streaming prompt layout
- Deterministic top-k=1 decoding
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def write_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def manifest_append(path: Path, name: str, dtype: str, count: int, shape: tuple[int, ...]) -> None:
    with path.open("a", encoding="utf-8") as f:
        shape_text = "x".join(str(x) for x in shape)
        f.write(f"{name}\t{dtype}\t{count}\t{shape_text}\n")


def save_tensor(trace_dir: Path, manifest: Path, name: str, tensor: torch.Tensor, dtype: str) -> None:
    if dtype == "f32":
        arr = tensor.detach().float().cpu().numpy()
        arr = arr.astype(np.float32, copy=False)
    elif dtype == "i32":
        arr = tensor.detach().cpu().numpy()
        arr = arr.astype(np.int32, copy=False)
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
    write_bin(trace_dir / name, arr.reshape(-1))
    manifest_append(manifest, name, dtype, int(arr.size), tuple(int(x) for x in arr.shape))


def save_codepred_hidden_states(
    trace_dir: Path,
    manifest: Path,
    prefix: str,
    hidden_states: tuple[torch.Tensor, ...] | None,
    layer_records: list[tuple[int, str, torch.Tensor]] | None = None,
) -> None:
    if layer_records:
        for layer_idx, suffix, layer_hidden in layer_records:
            save_tensor(
                trace_dir,
                manifest,
                f"{prefix}_layer{layer_idx:02d}_{suffix}.f32.bin",
                layer_hidden.squeeze(0),
                "f32",
            )
    elif hidden_states:
        for layer_idx, layer_hidden in enumerate(hidden_states[1:-1]):
            save_tensor(
                trace_dir,
                manifest,
                f"{prefix}_layer{layer_idx:02d}_hidden.f32.bin",
                layer_hidden.squeeze(0),
                "f32",
            )

    if not hidden_states:
        return
    save_tensor(
        trace_dir,
        manifest,
        f"{prefix}_final_hidden.f32.bin",
        hidden_states[-1].squeeze(0),
        "f32",
    )


def save_talker_layer_outputs(
    trace_dir: Path,
    manifest: Path,
    prefix: str,
    layer_records: list[tuple[int, str, torch.Tensor]] | None,
) -> None:
    if not layer_records:
        return
    for layer_idx, suffix, layer_hidden in layer_records:
        if suffix == "hidden":
            continue
        save_tensor(
            trace_dir,
            manifest,
            f"{prefix}_layer{layer_idx:02d}_{suffix}.f32.bin",
            layer_hidden[:, -1, :].reshape(-1),
            "f32",
        )


def capture_decoder_layer_outputs(layers) -> tuple[list[tuple[int, str, torch.Tensor]], list[torch.utils.hooks.RemovableHandle]]:
    captures: list[tuple[int, str, torch.Tensor]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx, layer in enumerate(layers):
        def make_hook(suffix: str, layer_idx: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                captures.append((layer_idx, suffix, hidden.detach()))

            return hook

        handles.append(layer.input_layernorm.register_forward_hook(make_hook("attn_norm", layer_idx)))
        handles.append(layer.self_attn.register_forward_hook(make_hook("attn_out", layer_idx)))
        handles.append(layer.post_attention_layernorm.register_forward_hook(make_hook("ffn_norm", layer_idx)))
        handles.append(layer.mlp.register_forward_hook(make_hook("ffn_out", layer_idx)))

        def layer_hook(_module, _inputs, output, layer_idx=layer_idx):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captures.append((layer_idx, "hidden", hidden.detach()))

        handles.append(layer.register_forward_hook(layer_hook))

    return captures, handles


def capture_codepred_layer_outputs(cp) -> tuple[list[tuple[int, str, torch.Tensor]], list[torch.utils.hooks.RemovableHandle]]:
    return capture_decoder_layer_outputs(cp.model.layers)


def remove_hooks(handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


def load_int_matrix_json(path: Path) -> list[list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        codebooks = int(payload.get("codebooks", 16))
        flat = payload.get("codes")
        if not isinstance(flat, list):
            raise ValueError(f"missing codes array in: {path}")
        frames = int(payload.get("frames", 0))
        if frames <= 0:
            if len(flat) % codebooks != 0:
                raise ValueError(f"flat code array is not divisible by {codebooks}: {path}")
            frames = len(flat) // codebooks
        return [flat[i * codebooks : (i + 1) * codebooks] for i in range(frames)]
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        return payload
    if isinstance(payload, list):
        if len(payload) % 16 != 0:
            raise ValueError(f"flat code array is not divisible by 16: {path}")
        return [payload[i : i + 16] for i in range(0, len(payload), 16)]
    raise ValueError(f"unsupported reference code JSON shape: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump qwen_tts Python traces")
    ap.add_argument("--model", required=True, help="HF model path/id (e.g., models/Qwen3-TTS-12Hz-1.7B-CustomVoice)")
    ap.add_argument("--speaker", default=None, help="Speaker name for CustomVoice models")
    ap.add_argument("--speaker-embedding", type=Path, default=None, help="External speaker embedding JSON")
    ap.add_argument("--reference-text", default=None, help="Reference transcript for ICL mode")
    ap.add_argument("--reference-text-file", type=Path, default=None, help="Reference transcript file for ICL mode")
    ap.add_argument("--reference-codes", type=Path, default=None, help="Reference speech codes JSON for ICL mode")
    ap.add_argument("--text", default="Hello.", help="Input text")
    ap.add_argument("--language", default="English", help="Language")
    ap.add_argument("--trace-dir", required=True, help="Output trace directory")
    ap.add_argument("--max-new-tokens", type=int, default=64, help="Max generated frames")
    ap.add_argument("--max-frames", type=int, default=2, help="How many frames to dump")
    ap.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"], help="Torch dtype")
    ap.add_argument("--do-sample", action="store_true", help="Use sampling path; top-k=1 remains deterministic")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel

    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    manifest = trace_dir / "manifest.tsv"
    manifest.write_text("name\tdtype\tcount\tshape\n", encoding="utf-8")

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa" if args.device == "cuda" else "eager",
    )
    m = model.model
    talker = m.talker

    if (args.speaker is None) == (args.speaker_embedding is None):
        raise ValueError("pass exactly one of --speaker or --speaker-embedding")
    if args.reference_text is not None and args.reference_text_file is not None:
        raise ValueError("pass at most one of --reference-text or --reference-text-file")
    if (args.reference_text is not None or args.reference_text_file is not None) != (args.reference_codes is not None):
        raise ValueError("ICL tracing requires both reference text and reference codes")

    # Build input ids exactly like inference wrapper.
    input_id = model._tokenize_texts([model._build_assistant_text(args.text)])[0]  # [1, T]
    save_tensor(trace_dir, manifest, "input_text_tokens.i32.bin", input_id.squeeze(0), "i32")
    reference_text = None
    reference_id = None
    reference_codes = None
    if args.reference_codes is not None:
        reference_text = args.reference_text
        if args.reference_text_file is not None:
            reference_text = args.reference_text_file.read_text(encoding="utf-8").strip()
        if not reference_text:
            raise ValueError("reference text is empty")
        reference_id = model._tokenize_texts([model._build_ref_text(reference_text)])[0]
        reference_codes_values = load_int_matrix_json(args.reference_codes)
        reference_codes = torch.tensor(reference_codes_values, device=talker.device, dtype=torch.long)
        if reference_codes.ndim != 2:
            raise ValueError(f"reference codes must be 2-D, got shape={tuple(reference_codes.shape)}")
        if reference_codes.shape[1] != m.config.talker_config.num_code_groups:
            raise ValueError(
                f"reference codes have {reference_codes.shape[1]} codebooks, "
                f"expected {m.config.talker_config.num_code_groups}"
            )
        save_tensor(trace_dir, manifest, "reference_text_tokens.i32.bin", reference_id.squeeze(0), "i32")
        save_tensor(trace_dir, manifest, "reference_codes.i32.bin", reference_codes, "i32")

    if args.speaker is not None:
        if m.tts_model_type != "custom_voice":
            raise ValueError(f"--speaker requires custom_voice model, got: {m.tts_model_type}")
        if args.speaker.lower() not in m.config.talker_config.spk_id:
            raise ValueError(f"unknown speaker '{args.speaker}'")
        spk_id = m.config.talker_config.spk_id[args.speaker.lower()]
        speaker_embed = talker.get_input_embeddings()(
            torch.tensor(spk_id, device=talker.device, dtype=input_id.dtype)
        ).view(1, 1, -1)
    else:
        speaker_values = json.loads(args.speaker_embedding.read_text(encoding="utf-8"))
        speaker_embed = torch.tensor(speaker_values, device=talker.device, dtype=torch_dtype).view(1, 1, -1)
    save_tensor(trace_dir, manifest, "speaker_embd.f32.bin", speaker_embed.squeeze(0).squeeze(0), "f32")

    language = args.language
    if language.lower() == "auto":
        language_id = None
    else:
        language_id = m.config.talker_config.codec_language_id[language.lower()]

    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor(
                [[m.config.tts_bos_token_id, m.config.tts_eos_token_id, m.config.tts_pad_token_id]],
                device=talker.device,
                dtype=input_id.dtype,
            )
        )
    ).chunk(3, dim=1)

    if language_id is None:
        codec_prefill = [[
            m.config.talker_config.codec_nothink_id,
            m.config.talker_config.codec_think_bos_id,
            m.config.talker_config.codec_think_eos_id,
        ]]
    else:
        codec_prefill = [[
            m.config.talker_config.codec_think_id,
            m.config.talker_config.codec_think_bos_id,
            language_id,
            m.config.talker_config.codec_think_eos_id,
        ]]

    codec_embd_0 = talker.get_input_embeddings()(
        torch.tensor(codec_prefill, device=talker.device, dtype=input_id.dtype)
    )
    codec_embd_1 = talker.get_input_embeddings()(
        torch.tensor(
            [[m.config.talker_config.codec_pad_id, m.config.talker_config.codec_bos_id]],
            device=talker.device,
            dtype=input_id.dtype,
        )
    )
    codec_input_embedding = torch.cat([codec_embd_0, speaker_embed, codec_embd_1], dim=1)

    role_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    pre_codec = torch.cat(
        (
            tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
            tts_bos_embed,
        ),
        dim=1,
    ) + codec_input_embedding[:, :-1]
    codec_pad_embed, codec_bos_embed = codec_embd_1[:, :1], codec_embd_1[:, 1:]

    if reference_codes is not None:
        assert reference_id is not None
        text_ids = torch.cat((reference_id[:, 3:-2], input_id[:, 3:-5]), dim=-1)
        text_embed = talker.text_projection(talker.get_text_embeddings()(text_ids))
        text_embed = torch.cat((text_embed, tts_eos_embed), dim=1)
        text_with_codec_pad = text_embed + talker.get_input_embeddings()(
            torch.tensor(
                [[m.config.talker_config.codec_pad_id] * text_embed.shape[1]],
                device=talker.device,
                dtype=input_id.dtype,
            )
        )

        codec_rows = []
        ref_code_batched = reference_codes.unsqueeze(0)
        for cb in range(m.config.talker_config.num_code_groups):
            if cb == 0:
                codec_rows.append(talker.get_input_embeddings()(ref_code_batched[:, :, cb]))
            else:
                codec_rows.append(talker.code_predictor.get_input_embeddings()[cb - 1](ref_code_batched[:, :, cb]))
        ref_codec_embed = torch.stack(codec_rows, dim=0).sum(0)
        ref_codec_embed = torch.cat((codec_bos_embed, ref_codec_embed), dim=1)
        codec_with_tts_pad = ref_codec_embed + tts_pad_embed
        icl_input_embed = torch.cat((text_with_codec_pad, codec_with_tts_pad), dim=1)
        talker_input_embed = torch.cat((role_embed, pre_codec, icl_input_embed), dim=1)
    else:
        text_body_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:-5]))
        text_body_with_codec = text_body_embed + codec_pad_embed
        eos_with_codec = tts_eos_embed + codec_pad_embed
        bos_with_codec = tts_pad_embed + codec_bos_embed
        talker_input_embed = torch.cat(
            (
                role_embed,
                pre_codec,
                text_body_with_codec,
                eos_with_codec,
                bos_with_codec,
            ),
            dim=1,
        )
    trailing_text_hidden = tts_pad_embed

    save_tensor(trace_dir, manifest, "prefill_embd.f32.bin", talker_input_embed.squeeze(0), "f32")

    # Single-item batch: no left padding needed.
    talker_input_embeds = talker_input_embed
    talker_attention_mask = torch.ones(
        (1, talker_input_embeds.shape[1]), device=talker.device, dtype=torch.long
    )
    trailing_text_hiddens = trailing_text_hidden

    suppress_tokens = [
        i
        for i in range(m.config.talker_config.vocab_size - 1024, m.config.talker_config.vocab_size)
        if i not in (m.config.talker_config.codec_eos_token_id,)
    ]

    talker_layer_outputs, talker_layer_hooks = capture_decoder_layer_outputs(talker.model.layers)
    try:
        talker_result = talker.generate(
            inputs_embeds=talker_input_embeds,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=2,
            do_sample=args.do_sample,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            subtalker_dosample=args.do_sample,
            subtalker_top_k=1,
            subtalker_top_p=1.0,
            subtalker_temperature=1.0,
            eos_token_id=m.config.talker_config.codec_eos_token_id,
            repetition_penalty=1.05,
            suppress_tokens=suppress_tokens,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    finally:
        remove_hooks(talker_layer_hooks)

    talker_codes = torch.stack(
        [hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1
    )  # [1, steps, 16]
    talker_hidden_states = torch.cat(
        [hid[0][-1][:, -1:] for hid in talker_result.hidden_states], dim=1
    )[:, :-1]  # [1, steps, H]
    prefill_hidden_layers = talker_result.hidden_states[0][0]
    for layer_idx, layer_hidden in enumerate(prefill_hidden_layers[1:-1]):
        save_tensor(
            trace_dir,
            manifest,
            f"talker_prefill_layer{layer_idx:02d}_hidden.f32.bin",
            layer_hidden[:, -1, :].reshape(-1),
            "f32",
        )
    save_tensor(
        trace_dir,
        manifest,
        "talker_prefill_final_hidden.f32.bin",
        prefill_hidden_layers[-1][:, -1, :].reshape(-1),
        "f32",
    )
    talker_records_per_forward = len(talker.model.layers) * 5
    save_talker_layer_outputs(
        trace_dir,
        manifest,
        "talker_prefill",
        talker_layer_outputs[:talker_records_per_forward],
    )

    steps = min(args.max_frames, int(talker_codes.shape[1]))
    cp = talker.code_predictor
    n_cp = m.config.talker_config.num_code_groups - 1

    for frame in range(steps):
        cb0_logits = talker_result.scores[frame][0].detach()
        cb0_token = talker_codes[0, frame, 0].detach()
        frame_codes = talker_codes[0, frame, :].detach()
        hidden = talker_hidden_states[0, frame, :].detach()

        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_cb0_logits_post_rules.f32.bin",
            cb0_logits,
            "f32",
        )
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_cb0_token.i32.bin",
            cb0_token.view(1),
            "i32",
        )
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_codec_tokens_cb0_15.i32.bin",
            frame_codes,
            "i32",
        )
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_talker_hidden.f32.bin",
            hidden,
            "f32",
        )
        frame_hidden_layers = talker_result.hidden_states[frame][0]
        for layer_idx, layer_hidden in enumerate(frame_hidden_layers[1:-1]):
            save_tensor(
                trace_dir,
                manifest,
                f"frame{frame:03d}_talker_layer{layer_idx:02d}_hidden.f32.bin",
                layer_hidden[:, -1, :].reshape(-1),
                "f32",
            )
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_talker_final_hidden.f32.bin",
            frame_hidden_layers[-1][:, -1, :].reshape(-1),
            "f32",
        )
        layer_start = frame * talker_records_per_forward
        layer_end = layer_start + talker_records_per_forward
        save_talker_layer_outputs(
            trace_dir,
            manifest,
            f"frame{frame:03d}_talker",
            talker_layer_outputs[layer_start:layer_end],
        )

        cb0_embd = talker.get_input_embeddings()(cb0_token.view(1, 1))
        cp_input_hidden = hidden.view(1, 1, -1)
        cp_inputs = torch.cat((cp_input_hidden, cb0_embd), dim=1)
        cp_inputs_projected = cp.small_to_mtp_projection(cp_inputs)

        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_codepred_input_hidden.f32.bin",
            cp_input_hidden.view(-1),
            "f32",
        )
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_codepred_input_cb0_embd.f32.bin",
            cb0_embd.view(-1),
            "f32",
        )
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_codepred_prefill_input.f32.bin",
            cp_inputs.squeeze(0),
            "f32",
        )
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_codepred_prefill_concat.f32.bin",
            cp_inputs.squeeze(0),
            "f32",
        )
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_codepred_prefill_projected.f32.bin",
            cp_inputs_projected.squeeze(0),
            "f32",
        )
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_codepred_prefill_pos.i32.bin",
            torch.tensor([0, 1], device=talker.device, dtype=torch.int32),
            "i32",
        )
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_codepred_prefill_mask.f32.bin",
            torch.tensor([[0.0, float("-inf")], [0.0, 0.0]], device=talker.device),
            "f32",
        )

        cp_forward_layer_outputs, cp_forward_hooks = capture_codepred_layer_outputs(cp)
        try:
            with torch.no_grad():
                cp_forward = cp(
                    inputs_embeds=cp_inputs,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            remove_hooks(cp_forward_hooks)
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_codepred_logits_step00.f32.bin",
            cp_forward.logits[:, -1, :].reshape(-1),
            "f32",
        )
        save_codepred_hidden_states(
            trace_dir,
            manifest,
            f"frame{frame:03d}_codepred_prefill",
            cp_forward.hidden_states,
            cp_forward_layer_outputs,
        )

        cp_generate_layer_outputs, cp_generate_hooks = capture_codepred_layer_outputs(cp)
        try:
            cp_result = cp.generate(
                inputs_embeds=cp_inputs,
                max_new_tokens=n_cp,
                do_sample=args.do_sample,
                top_k=1,
                top_p=1.0,
                temperature=1.0,
                output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )
        finally:
            remove_hooks(cp_generate_hooks)
        cp_tokens = cp_result.sequences[0].detach()
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_codepred_tokens_cb1_15.i32.bin",
            cp_tokens,
            "i32",
        )
        for step in range(1, min(n_cp, len(cp_tokens))):
            prev_token = cp_tokens[step - 1].view(1, 1)
            step_embd = cp.get_input_embeddings()[step - 1](prev_token)
            step_projected = cp.small_to_mtp_projection(step_embd)
            save_tensor(
                trace_dir,
                manifest,
                f"frame{frame:03d}_codepred_step{step:02d}_projected.f32.bin",
                step_projected.squeeze(0),
                "f32",
            )

        generated_hidden_states = getattr(cp_result, "hidden_states", None)
        if generated_hidden_states is not None:
            n_layers = len(cp.model.layers)
            records_per_forward = n_layers * 5
            for step in range(1, min(n_cp, len(generated_hidden_states))):
                layer_start = step * records_per_forward
                layer_end = layer_start + records_per_forward
                layer_outputs = cp_generate_layer_outputs[layer_start:layer_end]
                save_codepred_hidden_states(
                    trace_dir,
                    manifest,
                    f"frame{frame:03d}_codepred_step{step:02d}",
                    generated_hidden_states[step],
                    layer_outputs,
                )
        for step in range(min(n_cp, len(cp_result.scores))):
            step_logits = cp_result.scores[step][0].detach()
            save_tensor(
                trace_dir,
                manifest,
                f"frame{frame:03d}_codepred_logits_step{step:02d}_post_warp.f32.bin",
                step_logits,
                "f32",
            )
        raw_logits = getattr(cp_result, "logits", None)
        if raw_logits is not None:
            for step in range(min(n_cp, len(raw_logits))):
                step_logits = raw_logits[step][0].detach()
                save_tensor(
                    trace_dir,
                    manifest,
                    f"frame{frame:03d}_codepred_logits_step{step:02d}.f32.bin",
                    step_logits,
                    "f32",
                )

    info = trace_dir / "trace_info.txt"
    info.write_text(
        "\n".join(
            [
                f"python_model={args.model}",
                f"speaker={args.speaker or ''}",
                f"speaker_embedding={args.speaker_embedding or ''}",
                f"reference_text={reference_text or ''}",
                f"reference_codes={args.reference_codes or ''}",
                f"text={args.text}",
                f"language={args.language}",
                f"max_new_tokens={args.max_new_tokens}",
                f"max_frames={args.max_frames}",
                f"device={args.device}",
                f"dtype={args.dtype}",
                f"do_sample={args.do_sample}",
                f"hidden_size={m.config.talker_config.hidden_size}",
                f"codec_vocab_size={m.config.talker_config.vocab_size}",
                f"code_pred_vocab_size={m.config.talker_config.code_predictor_config.vocab_size}",
                f"n_codebooks={m.config.talker_config.num_code_groups}",
                f"n_tokens={input_id.shape[-1]}",
                f"prefill_len={talker_input_embed.shape[1]}",
                f"trailing_len={trailing_text_hidden.shape[1]}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Trace written to: {trace_dir}")


if __name__ == "__main__":
    main()
