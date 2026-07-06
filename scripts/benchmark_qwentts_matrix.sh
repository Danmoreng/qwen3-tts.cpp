#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

RUNS=8
WARMUP=1
MAX_TOKENS=4096
THREADS=4
TEXT="The strange thing about knowledge is that the more you gather, the more you realize how much remains unknown. Perhaps that is the whole point of the journey."
REFERENCE_AUDIO=""
REFERENCE_TEXT=""
LANGUAGE="en"
SEED=42
TEMPERATURE=0.9
TOP_K=50
TOP_P=1.0
REPETITION_PENALTY=1.05
OUT_DIR=""
QWEN_CPP_EXE=""
QWEN_CPP_MODELS=""
QWEN_CPP_MODEL_NAME="qwen-talker-1.7b-base-Q8_0.gguf"
QWEN_CPP_CODEC_NAME="qwen-tokenizer-12hz-Q8_0.gguf"
SERVEUR_REPO=""
SERVEUR_EXE=""
SERVEUR_CODEC_EXE=""
SERVEUR_SERVER_EXE=""
SERVEUR_TALKER=""
SERVEUR_CODEC=""
SKIP_COLD=0
SKIP_SERVER=0
VALIDATE_ONLY=0

usage() {
    cat <<'USAGE'
Usage: scripts/benchmark_qwentts_matrix.sh [options]

Runs the same qwen3-tts.cpp vs qwentts.cpp benchmark matrix as the Windows
PowerShell script, but with native Linux bash/curl tooling.

Options:
  --runs N                     Measured runs per scope (default: 8)
  --warmup N                   Warmup server/resident requests (default: 1)
  --max-tokens N               Max generated audio tokens (default: 4096)
  --threads N                  CPU helper threads (default: 4)
  --text TEXT                  Synthesis prompt
  --reference-audio PATH       Reference WAV
  --reference-text TEXT        Reference transcript
  --reference-text-file PATH   Read reference transcript from file
  --language LANG              en, zh, de, fr, es, it, ja, ko, pt, ru, auto
  --seed N                     Sampling seed (default: 42)
  --temperature X              Sampling temperature (default: 0.9)
  --top-k N                    Top-k (default: 50)
  --top-p X                    Top-p (default: 1.0)
  --repetition-penalty X       Repetition penalty (default: 1.05)
  --out-dir PATH               Output directory
  --workspace-root PATH        Parent workspace containing both repos
  --qwen-cpp-exe PATH          qwen3-tts-cli executable
  --qwen-cpp-models PATH       qwen3 model directory
  --qwen-cpp-model-name NAME   Talker GGUF name
  --qwen-cpp-codec-name NAME   Codec GGUF name
  --serveur-repo PATH          qwentts.cpp checkout
  --serveur-exe PATH           qwentts qwen-tts executable
  --serveur-codec-exe PATH     qwentts qwen-codec executable
  --serveur-server-exe PATH    qwentts tts-server executable
  --serveur-talker PATH        qwentts talker GGUF
  --serveur-codec PATH         qwentts codec GGUF
  --skip-cold                  Skip cold CLI scopes
  --skip-server                Skip qwentts HTTP server scopes
  --validate-only              Resolve inputs and exit
  -h, --help                   Show this help

Outputs:
  benchmark_matrix_results.csv/json
  benchmark_matrix_summary.csv/json
  logs/ and artifacts/
USAGE
}

die() {
    echo "error: $*" >&2
    exit 1
}

first_existing() {
    local path
    for path in "$@"; do
        if [[ -n "$path" && -e "$path" ]]; then
            realpath "$path"
            return 0
        fi
    done
    return 1
}

require_path() {
    local path="$1"
    local desc="$2"
    [[ -n "$path" && -e "$path" ]] || die "$desc not found: $path"
    realpath "$path"
}

serveur_language() {
    case "$1" in
        en) echo "English" ;;
        zh) echo "Chinese" ;;
        de) echo "German" ;;
        fr) echo "French" ;;
        es) echo "Spanish" ;;
        it) echo "Italian" ;;
        ja) echo "Japanese" ;;
        ko) echo "Korean" ;;
        pt) echo "Portuguese" ;;
        ru) echo "Russian" ;;
        auto) echo "auto" ;;
        *) echo "$1" ;;
    esac
}

quote_cmd() {
    local arg
    printf '%q' "$1"
}

format_command() {
    local exe="$1"
    shift
    local out
    out="$(quote_cmd "$exe")"
    local arg
    for arg in "$@"; do
        out+=" $(quote_cmd "$arg")"
    done
    printf '%s' "$out"
}

now_ms() {
    python3 - <<'PY'
import time
print(int(time.monotonic() * 1000))
PY
}

read_file_or_empty() {
    local path="$1"
    if [[ -f "$path" ]]; then
        cat "$path"
    fi
}

run_command() {
    local name="$1"
    local exe="$2"
    local cwd="$3"
    local log_path="$4"
    local stdin_text="$5"
    shift 5
    local -a args=("$@")

    mkdir -p "$(dirname "$log_path")"
    local stdout_path stderr_path
    stdout_path="$(mktemp)"
    stderr_path="$(mktemp)"
    local t0 t1 rc
    t0="$(now_ms)"
    set +e
    if [[ -n "$stdin_text" ]]; then
        (cd "$cwd" && printf '%s' "$stdin_text" | "$exe" "${args[@]}") >"$stdout_path" 2>"$stderr_path"
    else
        (cd "$cwd" && "$exe" "${args[@]}") >"$stdout_path" 2>"$stderr_path"
    fi
    rc=$?
    set -e
    t1="$(now_ms)"

    CMD_EXIT="$rc"
    CMD_WALL_SEC="$(python3 - <<PY
print(($t1 - $t0) / 1000.0)
PY
)"
    CMD_STDOUT="$(cat "$stdout_path")"
    CMD_STDERR="$(cat "$stderr_path")"
    CMD_LOG_TEXT="${CMD_STDOUT}${CMD_STDERR}"
    CMD_COMMAND="$(format_command "$exe" "${args[@]}")"

    {
        echo "Implementation: $name"
        echo "Command: $CMD_COMMAND"
        echo "ExitCode: $CMD_EXIT"
        echo "WallSeconds: $CMD_WALL_SEC"
        echo
        echo "[stdout]"
        cat "$stdout_path"
        echo
        echo "[stderr]"
        cat "$stderr_path"
    } >"$log_path"

    rm -f "$stdout_path" "$stderr_path"
}

append_row() {
    local engine="$1"
    local scope="$2"
    local run="$3"
    local out_wav="$4"
    local wall_seconds="$5"
    local exit_code="$6"
    local log_path="$7"
    local command="$8"
    local ttfa_ms="${9:-}"
    local bench_json="${10:-}"

    python3 - "$ROWS_JSONL" "$engine" "$scope" "$run" "$out_wav" "$wall_seconds" "$exit_code" "$log_path" "$command" "$ttfa_ms" "$bench_json" <<'PY'
import json, math, re, struct, sys, wave
from pathlib import Path

rows_path, engine, scope, run, out_wav, wall_seconds, exit_code, log_path, command, ttfa_ms, bench_json = sys.argv[1:12]
run = int(run)
wall_seconds = float(wall_seconds)
exit_code = int(exit_code)
bench = json.loads(bench_json) if bench_json else None

def wav_stats(path):
    if not path:
        return False, "N/A", 0.0, 0, 0.0, 0.0, 0
    p = Path(path)
    if not p.exists():
        return False, "file not found", 0.0, 0, 0.0, 0.0, 0
    try:
        with wave.open(str(p), "rb") as w:
            frames = w.getnframes()
            sr = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(frames)
        duration = frames / sr if sr else 0.0
        nonzero = 0
        peak = 0.0
        ss = 0.0
        if width == 2:
            count = len(raw) // 2
            vals = struct.unpack("<" + "h" * count, raw)
            for v in vals:
                x = v / 32768.0
                if v:
                    nonzero += 1
                ax = abs(x)
                peak = max(peak, ax)
                ss += x * x
            rms = math.sqrt(ss / count) if count else 0.0
        else:
            nonzero = len(raw)
            rms = 0.0
        return True, "OK" if nonzero >= 16 and peak >= 1e-4 else "LOW_AUDIO", duration, frames * channels, peak, rms, nonzero
    except Exception as exc:
        return False, str(exc), 0.0, 0, 0.0, 0.0, 0

valid, status, audio_sec, samples, peak, rms, nonzero = wav_stats(out_wav)
if not valid and bench:
    audio_sec = float(bench.get("audio_sec", 0.0))
    status = "RAW_PCM" if audio_sec > 0 else status

rtf = wall_seconds / audio_sec if audio_sec > 0 else None
xrt = audio_sec / wall_seconds if wall_seconds > 0 else None
internal_total = generate_ms = decode_ms = parsed_ttfa = None

if bench:
    internal_total = bench.get("internal_total_ms")
    generate_ms = bench.get("generate_ms")
    decode_ms = bench.get("decode_ms")
    if bench.get("ttfa_ms", -1) >= 0:
        parsed_ttfa = bench.get("ttfa_ms")
else:
    text = Path(log_path).read_text(errors="replace") if log_path and Path(log_path).exists() else ""
    def metric(pattern):
        m = re.search(pattern, text, re.M)
        return float(m.group(1)) if m else None
    if engine == "qwen3-tts.cpp":
        internal_total = metric(r"^\s*Total:\s+([0-9]+(?:\.[0-9]+)?)\s+ms")
        generate_ms = metric(r"^\s*Generate:\s+([0-9]+(?:\.[0-9]+)?)\s+ms")
        if generate_ms is None:
            generate_ms = metric(r"^\s*Code\+streaming:\s+([0-9]+(?:\.[0-9]+)?)\s+ms")
        decode_ms = metric(r"^\s*Decode:\s+([0-9]+(?:\.[0-9]+)?)\s+ms")
        if decode_ms is None:
            decode_ms = metric(r"^\s*Streaming decode:\s*([0-9]+(?:\.[0-9]+)?)\s+ms")
    elif engine == "qwentts.cpp":
        internal_total = metric(r"\[Perf\]\s+Total\s+([0-9]+(?:\.[0-9]+)?)\s+ms")
        talker = metric(r"\[Perf\]\s+TalkerDecode\s+([0-9]+(?:\.[0-9]+)?)\s+ms")
        codepred = metric(r"\[Perf\]\s+CodePredictor\s+([0-9]+(?:\.[0-9]+)?)\s+ms")
        generate_ms = talker + codepred if talker is not None and codepred is not None else talker
        decode_ms = metric(r"\[Perf\]\s+CodecDecode\s+([0-9]+(?:\.[0-9]+)?)\s+ms")
        parsed_ttfa = metric(r"\[Perf\]\s+TTFA\s+([0-9]+(?:\.[0-9]+)?)\s+ms")

if ttfa_ms:
    parsed_ttfa = float(ttfa_ms)

row = {
    "Engine": engine,
    "Scope": scope,
    "Run": run,
    "ExitCode": exit_code,
    "AudioStatus": status,
    "WallSeconds": round(wall_seconds, 3),
    "AudioSeconds": round(audio_sec, 3),
    "RTF_WallPerAudio": round(rtf, 4) if rtf is not None else None,
    "XRealtime_AudioPerWall": round(xrt, 3) if xrt is not None else None,
    "MsPerAudioSecond": round(rtf * 1000.0, 1) if rtf is not None else None,
    "TTFA_Ms": round(parsed_ttfa, 1) if parsed_ttfa is not None else None,
    "InternalTotalMs": round(float(internal_total), 1) if internal_total is not None else None,
    "GenerateMs": round(float(generate_ms), 1) if generate_ms is not None else None,
    "DecodeMs": round(float(decode_ms), 1) if decode_ms is not None else None,
    "StreamChunks": int(bench.get("stream_chunks", 0)) if bench else None,
    "Output": out_wav,
    "LogPath": log_path,
    "Command": command,
}
with open(rows_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

write_outputs() {
    python3 - "$ROWS_JSONL" "$RESULTS_CSV" "$RESULTS_JSON" "$SUMMARY_CSV" "$SUMMARY_JSON" <<'PY'
import csv, json, sys
from collections import defaultdict

rows_jsonl, results_csv, results_json, summary_csv, summary_json = sys.argv[1:6]
rows = []
with open(rows_jsonl, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

fields = [
    "Engine", "Scope", "Run", "ExitCode", "AudioStatus", "WallSeconds",
    "AudioSeconds", "RTF_WallPerAudio", "XRealtime_AudioPerWall",
    "MsPerAudioSecond", "TTFA_Ms", "InternalTotalMs", "GenerateMs",
    "DecodeMs", "StreamChunks", "Output", "LogPath", "Command",
]
with open(results_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k) for k in fields})
with open(results_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

groups = defaultdict(list)
for row in rows:
    if row.get("ExitCode") == 0 and row.get("AudioSeconds"):
        groups[(row["Engine"], row["Scope"])].append(row)

summary = []
for (engine, scope), items in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    wall = sum(float(x["WallSeconds"]) for x in items)
    audio = sum(float(x["AudioSeconds"]) for x in items)
    ttfas = [float(x["TTFA_Ms"]) for x in items if x.get("TTFA_Ms") is not None]
    summary.append({
        "Engine": engine,
        "Scope": scope,
        "Runs": len(items),
        "WallTotal": round(wall, 3),
        "AudioTotal": round(audio, 3),
        "RTF": round(wall / audio, 4) if audio > 0 else None,
        "XRealtime": round(audio / wall, 3) if wall > 0 else None,
        "AvgTTFA": round(sum(ttfas) / len(ttfas), 1) if ttfas else None,
    })

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    fields = ["Engine", "Scope", "Runs", "WallTotal", "AudioTotal", "RTF", "XRealtime", "AvgTTFA"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(summary)
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

widths = {k: len(k) for k in ["Engine", "Scope", "Runs", "WallTotal", "AudioTotal", "RTF", "XRealtime", "AvgTTFA"]}
for row in summary:
    for k in widths:
        widths[k] = max(widths[k], len("" if row.get(k) is None else str(row[k])))
print("Summary (standard RTF is wall/audio, lower is better):")
header = "  ".join(k.ljust(widths[k]) for k in widths)
print(header)
print("  ".join("-" * widths[k] for k in widths))
for row in summary:
    print("  ".join(("" if row.get(k) is None else str(row[k])).ljust(widths[k]) for k in widths))
PY
}

convert_pcm16_to_wav() {
    local pcm="$1"
    local wav="$2"
    python3 - "$pcm" "$wav" <<'PY'
import sys, wave
pcm, wav = sys.argv[1:3]
data = open(pcm, "rb").read()
with wave.open(wav, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(24000)
    w.writeframes(data)
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --text) TEXT="$2"; shift 2 ;;
        --reference-audio) REFERENCE_AUDIO="$2"; shift 2 ;;
        --reference-text) REFERENCE_TEXT="$2"; shift 2 ;;
        --reference-text-file) REFERENCE_TEXT="$(cat "$2")"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --repetition-penalty) REPETITION_PENALTY="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --workspace-root) WORKSPACE_ROOT="$(realpath "$2")"; shift 2 ;;
        --qwen-cpp-exe) QWEN_CPP_EXE="$2"; shift 2 ;;
        --qwen-cpp-models) QWEN_CPP_MODELS="$2"; shift 2 ;;
        --qwen-cpp-model-name) QWEN_CPP_MODEL_NAME="$2"; shift 2 ;;
        --qwen-cpp-codec-name) QWEN_CPP_CODEC_NAME="$2"; shift 2 ;;
        --serveur-repo) SERVEUR_REPO="$2"; shift 2 ;;
        --serveur-exe) SERVEUR_EXE="$2"; shift 2 ;;
        --serveur-codec-exe) SERVEUR_CODEC_EXE="$2"; shift 2 ;;
        --serveur-server-exe) SERVEUR_SERVER_EXE="$2"; shift 2 ;;
        --serveur-talker) SERVEUR_TALKER="$2"; shift 2 ;;
        --serveur-codec) SERVEUR_CODEC="$2"; shift 2 ;;
        --skip-cold) SKIP_COLD=1; shift ;;
        --skip-server) SKIP_SERVER=1; shift ;;
        --validate-only) VALIDATE_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

command -v python3 >/dev/null || die "python3 is required"
command -v curl >/dev/null || die "curl is required"
command -v base64 >/dev/null || die "base64 is required"
command -v realpath >/dev/null || die "realpath is required"

if [[ -z "$SERVEUR_REPO" ]]; then
    SERVEUR_REPO="$(first_existing "$WORKSPACE_ROOT/qwentts.cpp" "$WORKSPACE_ROOT/qwentts.cpp-serveurperso" || true)"
fi
[[ -n "$SERVEUR_REPO" ]] || die "qwentts.cpp repo not found; pass --serveur-repo"
SERVEUR_REPO="$(realpath "$SERVEUR_REPO")"

if [[ -z "$QWEN_CPP_EXE" ]]; then
    QWEN_CPP_EXE="$(first_existing \
        "$PROJECT_ROOT/build/qwen3-tts-cli" \
        "$PROJECT_ROOT/build/bin/qwen3-tts-cli" \
        "$PROJECT_ROOT/build-timing/qwen3-tts-cli" || true)"
fi
if [[ -z "$QWEN_CPP_MODELS" ]]; then
    QWEN_CPP_MODELS="$(first_existing "$HOME/.qwen-tts-studio/models" "$PROJECT_ROOT/models" || true)"
fi
if [[ -z "$SERVEUR_EXE" ]]; then
    SERVEUR_EXE="$(first_existing \
        "$SERVEUR_REPO/build/qwen-tts" \
        "$SERVEUR_REPO/build/bin/qwen-tts" \
        "$SERVEUR_REPO/build-cuda/qwen-tts" || true)"
fi
if [[ -z "$SERVEUR_CODEC_EXE" ]]; then
    SERVEUR_CODEC_EXE="$(first_existing \
        "$SERVEUR_REPO/build/qwen-codec" \
        "$SERVEUR_REPO/build/bin/qwen-codec" \
        "$SERVEUR_REPO/build-cuda/qwen-codec" || true)"
fi
if [[ -z "$SERVEUR_SERVER_EXE" ]]; then
    SERVEUR_SERVER_EXE="$(first_existing \
        "$SERVEUR_REPO/build/tts-server" \
        "$SERVEUR_REPO/build/bin/tts-server" \
        "$SERVEUR_REPO/build-cuda/tts-server" || true)"
fi

QWEN_CPP_EXE="$(require_path "$QWEN_CPP_EXE" "qwen3-tts.cpp CLI")"
QWEN_CPP_MODELS="$(require_path "$QWEN_CPP_MODELS" "qwen3-tts.cpp model directory")"
QWEN_CPP_TALKER="$(require_path "$QWEN_CPP_MODELS/$QWEN_CPP_MODEL_NAME" "qwen3-tts.cpp talker GGUF")"
QWEN_CPP_CODEC="$(require_path "$QWEN_CPP_MODELS/$QWEN_CPP_CODEC_NAME" "qwen3-tts.cpp codec GGUF")"
SERVEUR_EXE="$(require_path "$SERVEUR_EXE" "qwentts.cpp qwen-tts")"
SERVEUR_CODEC_EXE="$(require_path "$SERVEUR_CODEC_EXE" "qwentts.cpp qwen-codec")"
if [[ "$SKIP_SERVER" -eq 0 ]]; then
    SERVEUR_SERVER_EXE="$(require_path "$SERVEUR_SERVER_EXE" "qwentts.cpp tts-server")"
fi
if [[ -z "$SERVEUR_TALKER" ]]; then
    SERVEUR_TALKER="$QWEN_CPP_TALKER"
fi
if [[ -z "$SERVEUR_CODEC" ]]; then
    SERVEUR_CODEC="$QWEN_CPP_CODEC"
fi
SERVEUR_TALKER="$(require_path "$SERVEUR_TALKER" "qwentts.cpp talker GGUF")"
SERVEUR_CODEC="$(require_path "$SERVEUR_CODEC" "qwentts.cpp codec GGUF")"

if [[ -z "$REFERENCE_AUDIO" ]]; then
    REFERENCE_AUDIO="$(first_existing \
        "$SERVEUR_REPO/examples/freeman.wav" \
        "$WORKSPACE_ROOT/ref_audio_pcm.wav" \
        "$PROJECT_ROOT/examples/readme_clone_input.wav" || true)"
fi
REFERENCE_AUDIO="$(require_path "$REFERENCE_AUDIO" "reference audio")"
if [[ -z "$REFERENCE_TEXT" ]]; then
    if [[ -f "${REFERENCE_AUDIO%.*}.txt" ]]; then
        REFERENCE_TEXT="$(cat "${REFERENCE_AUDIO%.*}.txt")"
    elif [[ -f "$SERVEUR_REPO/examples/freeman.txt" ]]; then
        REFERENCE_TEXT="$(cat "$SERVEUR_REPO/examples/freeman.txt")"
    elif [[ -f "$PROJECT_ROOT/reference_text.txt" ]]; then
        REFERENCE_TEXT="$(cat "$PROJECT_ROOT/reference_text.txt")"
    fi
fi
[[ -n "$REFERENCE_TEXT" ]] || die "reference text is required; pass --reference-text or --reference-text-file"

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$WORKSPACE_ROOT/benchmark_output/qwentts_matrix/$(date +%Y%m%d-%H%M%S)"
elif [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$PROJECT_ROOT/$OUT_DIR"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(realpath "$OUT_DIR")"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"
ROWS_JSONL="$OUT_DIR/benchmark_matrix_results.jsonl"
RESULTS_CSV="$OUT_DIR/benchmark_matrix_results.csv"
RESULTS_JSON="$OUT_DIR/benchmark_matrix_results.json"
SUMMARY_CSV="$OUT_DIR/benchmark_matrix_summary.csv"
SUMMARY_JSON="$OUT_DIR/benchmark_matrix_summary.json"
: >"$ROWS_JSONL"

echo "Benchmark matrix preflight"
echo "  OutDir:        $OUT_DIR"
echo "  Runs/Warmup:   $RUNS / $WARMUP"
echo "  qwen3 CLI:     $QWEN_CPP_EXE"
echo "  qwentts CLI:   $SERVEUR_EXE"
echo "  qwentts server:${SERVEUR_SERVER_EXE:-<skipped>}"
echo "  Models:        $QWEN_CPP_MODELS"
echo "  Reference:     $REFERENCE_AUDIO"
echo

if [[ "$VALIDATE_ONLY" -eq 1 ]]; then
    echo "ValidateOnly completed. No benchmark commands were run."
    exit 0
fi

REFERENCE_TEXT_FILE="$ARTIFACT_DIR/reference.txt"
printf '%s' "$REFERENCE_TEXT" >"$REFERENCE_TEXT_FILE"
QWEN_ICL_PROMPT="$ARTIFACT_DIR/qwen3_icl_prompt.json"
SERVEUR_REF_COPY="$ARTIFACT_DIR/$(basename "$REFERENCE_AUDIO")"
cp -f "$REFERENCE_AUDIO" "$SERVEUR_REF_COPY"
SERVEUR_SPK="${SERVEUR_REF_COPY%.*}.spk"
SERVEUR_RVQ="${SERVEUR_REF_COPY%.*}.rvq"

echo "[artifacts] qwen3 ICL prompt"
run_command "qwen3-tts.cpp" "$QWEN_CPP_EXE" "$PROJECT_ROOT" "$LOG_DIR/qwen3_extract_icl.log" "" \
    -m "$QWEN_CPP_MODELS" --model-name "$QWEN_CPP_MODEL_NAME" --codec-model "$QWEN_CPP_CODEC" \
    -r "$REFERENCE_AUDIO" --reference-text-file "$REFERENCE_TEXT_FILE" \
    --extract-icl-prompt "$QWEN_ICL_PROMPT" -j "$THREADS"
[[ "$CMD_EXIT" -eq 0 && -f "$QWEN_ICL_PROMPT" ]] || die "qwen3 ICL prompt extraction failed. See $LOG_DIR/qwen3_extract_icl.log"

echo "[artifacts] qwentts voice latents"
run_command "qwentts.cpp" "$SERVEUR_CODEC_EXE" "$SERVEUR_REPO" "$LOG_DIR/qwentts_codec_extract.log" "" \
    --model "$SERVEUR_CODEC" --talker "$SERVEUR_TALKER" -i "$SERVEUR_REF_COPY"
[[ "$CMD_EXIT" -eq 0 && -f "$SERVEUR_SPK" && -f "$SERVEUR_RVQ" ]] || die "qwentts voice latent extraction failed. See $LOG_DIR/qwentts_codec_extract.log"

QWEN_BASE_ARGS=(-m "$QWEN_CPP_MODELS" --model-name "$QWEN_CPP_MODEL_NAME" --codec-model "$QWEN_CPP_CODEC" -t "$TEXT" --max-tokens "$MAX_TOKENS" --seed "$SEED" --temperature "$TEMPERATURE" --top-k "$TOP_K" --top-p "$TOP_P" --repetition-penalty "$REPETITION_PENALTY" -l "$LANGUAGE" -j "$THREADS")
SERVEUR_BASE_ARGS=(--model "$SERVEUR_TALKER" --codec "$SERVEUR_CODEC" --lang "$(serveur_language "$LANGUAGE")" --max-new "$MAX_TOKENS" --seed "$SEED" --temp "$TEMPERATURE" --top-k "$TOP_K" --top-p "$TOP_P" --rep-pen "$REPETITION_PENALTY")

if [[ "$SKIP_COLD" -eq 0 ]]; then
    for ((run = 1; run <= RUNS; run++)); do
        echo "[$run/$RUNS] cold_e2e_ref_wav"
        out="$OUT_DIR/qwen3_cold_ref_run${run}.wav"
        log="$LOG_DIR/qwen3_cold_ref_run${run}.log"
        run_command "qwen3-tts.cpp" "$QWEN_CPP_EXE" "$PROJECT_ROOT" "$log" "" \
            "${QWEN_BASE_ARGS[@]}" -o "$out" -r "$REFERENCE_AUDIO" --reference-text-file "$REFERENCE_TEXT_FILE"
        append_row "qwen3-tts.cpp" "cold_e2e_ref_wav" "$run" "$out" "$CMD_WALL_SEC" "$CMD_EXIT" "$log" "$CMD_COMMAND"

        out="$OUT_DIR/qwentts_cold_ref_run${run}.wav"
        log="$LOG_DIR/qwentts_cold_ref_run${run}.log"
        run_command "qwentts.cpp" "$SERVEUR_EXE" "$SERVEUR_REPO" "$log" "$TEXT" \
            "${SERVEUR_BASE_ARGS[@]}" -o "$out" --ref-wav "$REFERENCE_AUDIO" --ref-text "$REFERENCE_TEXT_FILE"
        append_row "qwentts.cpp" "cold_e2e_ref_wav" "$run" "$out" "$CMD_WALL_SEC" "$CMD_EXIT" "$log" "$CMD_COMMAND"

        echo "[$run/$RUNS] cold_e2e_preencoded"
        out="$OUT_DIR/qwen3_cold_preencoded_run${run}.wav"
        log="$LOG_DIR/qwen3_cold_preencoded_run${run}.log"
        run_command "qwen3-tts.cpp" "$QWEN_CPP_EXE" "$PROJECT_ROOT" "$log" "" \
            "${QWEN_BASE_ARGS[@]}" -o "$out" --icl-prompt "$QWEN_ICL_PROMPT"
        append_row "qwen3-tts.cpp" "cold_e2e_preencoded" "$run" "$out" "$CMD_WALL_SEC" "$CMD_EXIT" "$log" "$CMD_COMMAND"

        out="$OUT_DIR/qwentts_cold_preencoded_run${run}.wav"
        log="$LOG_DIR/qwentts_cold_preencoded_run${run}.log"
        run_command "qwentts.cpp" "$SERVEUR_EXE" "$SERVEUR_REPO" "$log" "$TEXT" \
            "${SERVEUR_BASE_ARGS[@]}" -o "$out" --ref-spk "$SERVEUR_SPK" --ref-rvq "$SERVEUR_RVQ" --ref-text "$REFERENCE_TEXT_FILE"
        append_row "qwentts.cpp" "cold_e2e_preencoded" "$run" "$out" "$CMD_WALL_SEC" "$CMD_EXIT" "$log" "$CMD_COMMAND"
    done
fi

echo "[resident] qwen3 buffered"
out="$OUT_DIR/qwen3_resident.wav"
log="$LOG_DIR/qwen3_resident.log"
run_command "qwen3-tts.cpp" "$QWEN_CPP_EXE" "$PROJECT_ROOT" "$log" "" \
    "${QWEN_BASE_ARGS[@]}" -o "$out" --icl-prompt "$QWEN_ICL_PROMPT" --bench-server "$RUNS" --bench-warmup "$WARMUP"
while IFS= read -r line; do
    [[ "$line" == BENCH_JSON\ * ]] || continue
    json="${line#BENCH_JSON }"
    warmup="$(python3 -c 'import json,sys; print(str(json.loads(sys.argv[1]).get("warmup")).lower())' "$json")"
    [[ "$warmup" == "false" ]] || continue
    iter="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["iteration"])' "$json")"
    row_out="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["output"])' "$json")"
    wall_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["wall_ms"])' "$json")"
    exit_code="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["exit_code"])' "$json")"
    append_row "qwen3-tts.cpp" "resident_preencoded" "$iter" "$row_out" "$(python3 - <<PY
print($wall_ms / 1000.0)
PY
)" "$exit_code" "$log" "$CMD_COMMAND" "" "$json"
done <<<"$CMD_STDOUT"

echo "[resident] qwen3 streaming"
out="$OUT_DIR/qwen3_resident_stream.wav"
log="$LOG_DIR/qwen3_resident_stream.log"
run_command "qwen3-tts.cpp" "$QWEN_CPP_EXE" "$PROJECT_ROOT" "$log" "" \
    "${QWEN_BASE_ARGS[@]}" -o "$out" --icl-prompt "$QWEN_ICL_PROMPT" --stream --bench-server "$RUNS" --bench-warmup "$WARMUP"
while IFS= read -r line; do
    [[ "$line" == BENCH_JSON\ * ]] || continue
    json="${line#BENCH_JSON }"
    warmup="$(python3 -c 'import json,sys; print(str(json.loads(sys.argv[1]).get("warmup")).lower())' "$json")"
    [[ "$warmup" == "false" ]] || continue
    iter="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["iteration"])' "$json")"
    row_out="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["output"])' "$json")"
    wall_ms="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["wall_ms"])' "$json")"
    exit_code="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["exit_code"])' "$json")"
    append_row "qwen3-tts.cpp" "resident_streaming_preencoded" "$iter" "$row_out" "$(python3 - <<PY
print($wall_ms / 1000.0)
PY
)" "$exit_code" "$log" "$CMD_COMMAND" "" "$json"
done <<<"$CMD_STDOUT"

SERVER_PID=""
cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup_server EXIT

if [[ "$SKIP_SERVER" -eq 0 ]]; then
    port="$(python3 - <<'PY'
import random
print(random.randint(18080, 25000))
PY
)"
    echo "[server] starting qwentts on port $port"
    "$SERVEUR_SERVER_EXE" --model "$SERVEUR_TALKER" --codec "$SERVEUR_CODEC" --host 127.0.0.1 --port "$port" --lang "$(serveur_language "$LANGUAGE")" >"$LOG_DIR/qwentts_server_stdout.log" 2>"$LOG_DIR/qwentts_server_stderr.log" &
    SERVER_PID="$!"
    ready=0
    for _ in $(seq 1 120); do
        if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 0.25
    done
    [[ "$ready" -eq 1 ]] || die "qwentts server did not become ready. See $LOG_DIR/qwentts_server_stderr.log"

    spk_b64="$(base64 -w0 "$SERVEUR_SPK")"
    rvq_b64="$(base64 -w0 "$SERVEUR_RVQ")"
    voice_json="$(python3 - "$REFERENCE_TEXT" "$spk_b64" "$rvq_b64" <<'PY'
import json, sys
print(json.dumps({"name": "bench_voice", "ref_text": sys.argv[1], "spk_b64": sys.argv[2], "rvq_b64": sys.argv[3]}))
PY
)"
    curl -fsS -X POST "http://127.0.0.1:$port/v1/voices" -H "Content-Type: application/json" -d "$voice_json" >/dev/null

    for ((run = 1; run <= WARMUP + RUNS; run++)); do
        warm=0
        iter="$run"
        if (( run <= WARMUP )); then
            warm=1
        else
            iter=$((run - WARMUP))
        fi
        payload="$(python3 - "$TEXT" "$SEED" "$MAX_TOKENS" "$TEMPERATURE" "$TOP_K" "$TOP_P" "$REPETITION_PENALTY" <<'PY'
import json, sys
print(json.dumps({
    "input": sys.argv[1], "voice": "bench_voice", "response_format": "wav",
    "seed": int(sys.argv[2]), "max_new_tokens": int(sys.argv[3]),
    "temperature": float(sys.argv[4]), "top_k": int(sys.argv[5]),
    "top_p": float(sys.argv[6]), "repetition_penalty": float(sys.argv[7]),
}))
PY
)"
        out="$OUT_DIR/qwentts_server_buffered_run${iter}.wav"
        time_total="$(curl -fsS -w '%{time_total}' -o "$out" -X POST "http://127.0.0.1:$port/v1/audio/speech" -H "Content-Type: application/json" -d "$payload")"
        if [[ "$warm" -eq 0 ]]; then
            append_row "qwentts.cpp" "http_server_preencoded" "$iter" "$out" "$time_total" "0" "$LOG_DIR/qwentts_server_stderr.log" "POST /v1/audio/speech response_format=wav"
        fi
    done

    for ((run = 1; run <= WARMUP + RUNS; run++)); do
        warm=0
        iter="$run"
        if (( run <= WARMUP )); then
            warm=1
        else
            iter=$((run - WARMUP))
        fi
        payload="$(python3 - "$TEXT" "$SEED" "$MAX_TOKENS" "$TEMPERATURE" "$TOP_K" "$TOP_P" "$REPETITION_PENALTY" <<'PY'
import json, sys
print(json.dumps({
    "input": sys.argv[1], "voice": "bench_voice", "response_format": "pcm",
    "seed": int(sys.argv[2]), "max_new_tokens": int(sys.argv[3]),
    "temperature": float(sys.argv[4]), "top_k": int(sys.argv[5]),
    "top_p": float(sys.argv[6]), "repetition_penalty": float(sys.argv[7]),
}))
PY
)"
        pcm="$OUT_DIR/qwentts_server_stream_run${iter}.pcm"
        wav="$OUT_DIR/qwentts_server_stream_run${iter}.wav"
        read -r start_transfer time_total < <(curl -fsS -w '%{time_starttransfer} %{time_total}' -o "$pcm" -X POST "http://127.0.0.1:$port/v1/audio/speech" -H "Content-Type: application/json" -d "$payload")
        convert_pcm16_to_wav "$pcm" "$wav"
        if [[ "$warm" -eq 0 ]]; then
            ttfa_ms="$(python3 - <<PY
print(round(float("$start_transfer") * 1000.0, 1))
PY
)"
            append_row "qwentts.cpp" "http_server_streaming_preencoded" "$iter" "$wav" "$time_total" "0" "$LOG_DIR/qwentts_server_stderr.log" "POST /v1/audio/speech response_format=pcm" "$ttfa_ms"
        fi
    done
fi

write_outputs
echo "Results: $RESULTS_CSV"
echo "Summary: $SUMMARY_CSV"
