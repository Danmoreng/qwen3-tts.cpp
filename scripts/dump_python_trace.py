#!/usr/bin/env python3
"""
Dump Python (qwen_tts) generation traces in the same file layout as C++ debug dumps.

Current scope:
- CustomVoice path (1.7B) with `--speaker`
- Non-streaming disabled (matches C++ behavior)
- Deterministic decoding (`temperature=0`, `top-k=1`, no sampling)
"""

from __future__ import annotations

import argparse
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
    arr = tensor.detach().cpu().numpy()
    if dtype == "f32":
        arr = arr.astype(np.float32, copy=False)
    elif dtype == "i32":
        arr = arr.astype(np.int32, copy=False)
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
    write_bin(trace_dir / name, arr.reshape(-1))
    manifest_append(manifest, name, dtype, int(arr.size), tuple(int(x) for x in arr.shape))


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump qwen_tts Python traces for CustomVoice")
    ap.add_argument("--model", required=True, help="HF model path/id (e.g., models/Qwen3-TTS-12Hz-1.7B-CustomVoice)")
    ap.add_argument("--speaker", required=True, help="Speaker name")
    ap.add_argument("--text", default="Hello.", help="Input text")
    ap.add_argument("--language", default="English", help="Language")
    ap.add_argument("--trace-dir", required=True, help="Output trace directory")
    ap.add_argument("--max-new-tokens", type=int, default=64, help="Max generated frames")
    ap.add_argument("--max-frames", type=int, default=2, help="How many frames to dump")
    ap.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"], help="Torch dtype")
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
        attn_implementation="sdpa",
    )
    m = model.model
    talker = m.talker

    if m.tts_model_type != "custom_voice":
        raise ValueError(f"expected custom_voice model, got: {m.tts_model_type}")
    if args.speaker.lower() not in m.config.talker_config.spk_id:
        raise ValueError(f"unknown speaker '{args.speaker}'")

    # Build input ids exactly like inference wrapper.
    input_id = model._tokenize_texts([model._build_assistant_text(args.text)])[0]  # [1, T]
    save_tensor(trace_dir, manifest, "input_text_tokens.i32.bin", input_id.squeeze(0), "i32")

    spk_id = m.config.talker_config.spk_id[args.speaker.lower()]
    speaker_embed = talker.get_input_embeddings()(
        torch.tensor(spk_id, device=talker.device, dtype=input_id.dtype)
    ).view(1, 1, -1)
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
    talker_input_embed = torch.cat((role_embed, pre_codec), dim=1)
    talker_input_embed = torch.cat(
        [
            talker_input_embed,
            talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input_embedding[:, -1:],
        ],
        dim=1,
    )
    trailing_text_hidden = torch.cat(
        (
            talker.text_projection(talker.get_text_embeddings()(input_id[:, 4:-5])),
            tts_eos_embed,
        ),
        dim=1,
    )

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

    talker_result = talker.generate(
        inputs_embeds=talker_input_embeds,
        attention_mask=talker_attention_mask,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=2,
        do_sample=False,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        subtalker_dosample=False,
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

    talker_codes = torch.stack(
        [hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1
    )  # [1, steps, 16]
    talker_hidden_states = torch.cat(
        [hid[0][-1][:, -1:] for hid in talker_result.hidden_states], dim=1
    )[:, :-1]  # [1, steps, H]

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

        cb0_embd = talker.get_input_embeddings()(cb0_token.view(1, 1))
        cp_input_hidden = hidden.view(1, 1, -1)
        cp_inputs = torch.cat((cp_input_hidden, cb0_embd), dim=1)

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

        cp_result = cp.generate(
            inputs_embeds=cp_inputs,
            max_new_tokens=n_cp,
            do_sample=False,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        cp_tokens = cp_result.sequences[0].detach()
        save_tensor(
            trace_dir,
            manifest,
            f"frame{frame:03d}_codepred_tokens_cb1_15.i32.bin",
            cp_tokens,
            "i32",
        )
        for step in range(min(n_cp, len(cp_result.scores))):
            step_logits = cp_result.scores[step][0].detach()
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
                f"speaker={args.speaker}",
                f"text={args.text}",
                f"language={args.language}",
                f"max_new_tokens={args.max_new_tokens}",
                f"max_frames={args.max_frames}",
                f"device={args.device}",
                f"dtype={args.dtype}",
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
