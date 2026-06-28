#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)

def run_cmd(cmd):
    print(f"[run] {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=True)

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    
    # 1. Download 1.7B Base Model
    base_17b_dir = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base"
    if not base_17b_dir.exists():
        print(f"Downloading 1.7B Base model to {base_17b_dir}...")
        run_cmd([
            VENV_PYTHON, "-c",
            f"from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='{base_17b_dir.as_posix()}', resume_download=True)"
        ])
    else:
        print(f"1.7B Base model already exists at {base_17b_dir}")

    # 2. Download 1.7B CustomVoice model
    custom_voice_dir = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    if not custom_voice_dir.exists():
        print(f"Downloading 1.7B CustomVoice model to {custom_voice_dir}...")
        run_cmd([
            VENV_PYTHON, "-c",
            f"from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', local_dir='{custom_voice_dir.as_posix()}', resume_download=True)"
        ])
    else:
        print(f"1.7B CustomVoice model already exists at {custom_voice_dir}")

    # 3. Download Tokenizer if missing
    tokenizer_dir = MODELS_DIR / "Qwen3-TTS-Tokenizer-12Hz"
    if not tokenizer_dir.exists():
        print(f"Downloading Tokenizer to {tokenizer_dir}...")
        run_cmd([
            VENV_PYTHON, "-c",
            f"from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-TTS-Tokenizer-12Hz', local_dir='{tokenizer_dir.as_posix()}', resume_download=True)"
        ])

    # 4. Convert 1.7B Base to GGUF
    base_gguf = MODELS_DIR / "qwen-talker-1.7b-base-Q8_0.gguf"
    print(f"Converting 1.7B Base to GGUF: {base_gguf}...")
    run_cmd([
        VENV_PYTHON, (REPO_ROOT / "scripts" / "convert_tts_to_gguf.py").as_posix(),
        "--input", base_17b_dir.as_posix(),
        "--output", base_gguf.as_posix(),
        "--type", "q8_0"
    ])

    # 5. Convert 1.7B CustomVoice to GGUF
    custom_gguf = MODELS_DIR / "qwen-talker-1.7b-customvoice-Q8_0.gguf"
    print(f"Converting 1.7B CustomVoice to GGUF: {custom_gguf}...")
    run_cmd([
        VENV_PYTHON, (REPO_ROOT / "scripts" / "convert_tts_to_gguf.py").as_posix(),
        "--input", custom_voice_dir.as_posix(),
        "--output", custom_gguf.as_posix(),
        "--type", "q8_0"
    ])

    # 6. Convert Tokenizer to GGUF if missing
    tokenizer_gguf = MODELS_DIR / "qwen-tokenizer-12hz-Q8_0.gguf"
    if not tokenizer_gguf.exists():
        print(f"Converting Tokenizer to GGUF: {tokenizer_gguf}...")
        run_cmd([
            VENV_PYTHON, (REPO_ROOT / "scripts" / "convert_tokenizer_to_gguf.py").as_posix(),
            "--input", tokenizer_dir.as_posix(),
            "--output", tokenizer_gguf.as_posix(),
            "--type", "q8_0"
        ])

    print("\nSetup complete!")
    print(f"Base Model: {base_gguf}")
    if (MODELS_DIR / "qwen-talker-1.7b-customvoice-Q8_0.gguf").exists():
        print(f"CustomVoice Model: {MODELS_DIR / 'qwen-talker-1.7b-customvoice-Q8_0.gguf'}")
    print(f"Tokenizer: {tokenizer_gguf}")

if __name__ == "__main__":
    main()
