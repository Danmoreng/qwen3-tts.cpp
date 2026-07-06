#include "qwen3_tts.h"
#include "gguf_loader.h"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#endif

#ifdef _WIN32
static std::string wide_to_utf8(const std::wstring & wide) {
    if (wide.empty()) {
        return {};
    }

    int size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), (int) wide.size(), nullptr, 0, nullptr, nullptr);
    if (size <= 0) {
        return {};
    }

    std::string utf8(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), (int) wide.size(), utf8.data(), size, nullptr, nullptr);
    return utf8;
}

static std::vector<std::string> get_utf8_argv() {
    int argc_w = 0;
    LPWSTR * argv_w = CommandLineToArgvW(GetCommandLineW(), &argc_w);
    std::vector<std::string> args;
    if (!argv_w) {
        return args;
    }

    args.reserve((size_t) argc_w);
    for (int i = 0; i < argc_w; ++i) {
        args.push_back(wide_to_utf8(argv_w[i]));
    }

    LocalFree(argv_w);
    return args;
}
#endif

static bool read_text_file(const std::string & path, std::string & text, std::string & error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "failed to open file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    text = ss.str();
    return true;
}

static void normalize_text_newlines(std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\r') {
            if (i + 1 < text.size() && text[i + 1] == '\n') {
                continue;
            }
            out.push_back('\n');
        } else {
            out.push_back(text[i]);
        }
    }
    text.swap(out);
}

static void trim_text_outer_whitespace(std::string & text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace((unsigned char) text[begin])) {
        ++begin;
    }

    size_t end = text.size();
    while (end > begin && std::isspace((unsigned char) text[end - 1])) {
        --end;
    }

    if (begin > 0 || end < text.size()) {
        text = text.substr(begin, end - begin);
    }
}

static std::string output_file_for_repeat(const std::string & path, int iteration, int repeat_count) {
    if (repeat_count <= 1) {
        return path;
    }

    const size_t slash = path.find_last_of("/\\");
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        dot = path.size();
    }

    std::ostringstream out;
    out << path.substr(0, dot) << "." << (iteration + 1) << path.substr(dot);
    return out.str();
}

static int64_t monotonic_time_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        clock::now().time_since_epoch()).count();
}

static std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) ch);
                    out += buf;
                } else {
                    out.push_back((char) ch);
                }
                break;
        }
    }
    return out;
}

static void print_bench_json(const char * engine,
                             const char * scope,
                             int iteration,
                             bool warmup,
                             int exit_code,
                             const qwen3_tts::tts_result & result,
                             int64_t wall_ms,
                             int64_t ttfa_ms,
                             const std::string & output_path) {
    const double audio_sec = result.sample_rate > 0
        ? (double) result.audio.size() / (double) result.sample_rate : 0.0;
    const double wall_sec = (double) wall_ms / 1000.0;
    const double rtf = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
    const double x_realtime = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;

    fprintf(stdout,
            "BENCH_JSON {"
            "\"engine\":\"%s\","
            "\"scope\":\"%s\","
            "\"iteration\":%d,"
            "\"warmup\":%s,"
            "\"exit_code\":%d,"
            "\"success\":%s,"
            "\"wall_ms\":%lld,"
            "\"audio_sec\":%.6f,"
            "\"rtf\":%.6f,"
            "\"x_realtime\":%.6f,"
            "\"ttfa_ms\":%lld,"
            "\"load_ms\":%lld,"
            "\"tokenize_ms\":%lld,"
            "\"encode_ms\":%lld,"
            "\"generate_ms\":%lld,"
            "\"decode_ms\":%lld,"
            "\"internal_total_ms\":%lld,"
            "\"decode_frames\":%d,"
            "\"decode_samples\":%lld,"
            "\"stream_chunks\":%d,"
            "\"stream_input_frames\":%d,"
            "\"stream_emitted_frames\":%d,"
            "\"stream_context_frames\":%d,"
            "\"output\":\"%s\""
            "}\n",
            engine,
            scope,
            iteration,
            warmup ? "true" : "false",
            exit_code,
            result.success ? "true" : "false",
            (long long) wall_ms,
            audio_sec,
            rtf,
            x_realtime,
            (long long) ttfa_ms,
            (long long) result.t_load_ms,
            (long long) result.t_tokenize_ms,
            (long long) result.t_encode_ms,
            (long long) result.t_generate_ms,
            (long long) result.t_decode_ms,
            (long long) result.t_total_ms,
            result.decode_frames,
            (long long) result.decode_samples,
            result.streaming_decode_chunks,
            result.streaming_decode_input_frames,
            result.streaming_decode_emitted_frames,
            result.streaming_decode_context_frames,
            json_escape(output_path).c_str());
    fflush(stdout);
}

static void print_result_timing(const qwen3_tts::tts_result & result) {
    fprintf(stderr, "\nTiming:\n");
    fprintf(stderr, "  Load:      %6lld ms\n", (long long) result.t_load_ms);
    fprintf(stderr, "  Tokenize:  %6lld ms\n", (long long) result.t_tokenize_ms);
    fprintf(stderr, "  Encode:    %6lld ms\n", (long long) result.t_encode_ms);
    if (result.t_reference_speaker_load_ms != 0 ||
        result.t_reference_speaker_encode_ms != 0 ||
        result.t_reference_speech_load_ms != 0 ||
        result.t_reference_speech_encode_ms != 0) {
        fprintf(stderr, "    speaker load:  %6lld ms\n",
                (long long) result.t_reference_speaker_load_ms);
        fprintf(stderr, "    speaker enc:   %6lld ms\n",
                (long long) result.t_reference_speaker_encode_ms);
        fprintf(stderr, "    speech load:   %6lld ms\n",
                (long long) result.t_reference_speech_load_ms);
        fprintf(stderr, "    speech enc:    %6lld ms\n",
                (long long) result.t_reference_speech_encode_ms);
        if (result.t_reference_speech_project_ms != 0 ||
            result.t_reference_speech_quantize_ms != 0) {
            fprintf(stderr, "      project:     %6lld ms\n",
                    (long long) result.t_reference_speech_project_ms);
            fprintf(stderr, "        build:     %6lld ms\n",
                    (long long) result.t_reference_speech_graph_build_ms);
            fprintf(stderr, "        alloc:     %6lld ms\n",
                    (long long) result.t_reference_speech_graph_alloc_ms);
            fprintf(stderr, "        upload:    %6lld ms\n",
                    (long long) result.t_reference_speech_input_upload_ms);
            fprintf(stderr, "        mask prep: %6lld ms\n",
                    (long long) result.t_reference_speech_mask_prepare_ms);
            fprintf(stderr, "        compute:   %6lld ms\n",
                    (long long) result.t_reference_speech_graph_compute_ms);
            fprintf(stderr, "        readback:  %6lld ms\n",
                    (long long) result.t_reference_speech_output_read_ms);
            fprintf(stderr, "      quantize:    %6lld ms\n",
                    (long long) result.t_reference_speech_quantize_ms);
            fprintf(stderr, "        semantic:  %6lld ms\n",
                    (long long) result.t_reference_speech_quantize_semantic_ms);
            fprintf(stderr, "        acoustic:  %6lld ms\n",
                    (long long) result.t_reference_speech_quantize_acoustic_ms);
        }
    }
    fprintf(stderr, "  Generate:  %6lld ms\n", (long long) result.t_generate_ms);
    fprintf(stderr, "  Decode:    %6lld ms\n", (long long) result.t_decode_ms);
    fprintf(stderr, "    graph build:   %6lld ms %s\n",
            (long long) result.t_decode_graph_build_ms,
            result.decode_graph_rebuilt ? "(rebuilt)" : "(cached)");
    fprintf(stderr, "    graph alloc:   %6lld ms\n", (long long) result.t_decode_graph_alloc_ms);
    fprintf(stderr, "    input upload:  %6lld ms\n", (long long) result.t_decode_input_upload_ms);
    fprintf(stderr, "    graph compute: %6lld ms\n", (long long) result.t_decode_graph_compute_ms);
    fprintf(stderr, "    output read:   %6lld ms\n", (long long) result.t_decode_output_read_ms);
    fprintf(stderr, "    frames/samples:%6d / %lld\n",
            result.decode_frames, (long long) result.decode_samples);
    if (result.streaming_decode_chunks > 0) {
        fprintf(stderr, "    stream chunks: %6d\n", result.streaming_decode_chunks);
        fprintf(stderr, "    stream frames: input=%d emitted=%d context=%d\n",
                result.streaming_decode_input_frames,
                result.streaming_decode_emitted_frames,
                result.streaming_decode_context_frames);
        fprintf(stderr,
                "    stream detail: rebuilds=%d build=%lld ms alloc=%lld ms upload=%lld ms compute=%lld ms read=%lld ms\n",
                result.streaming_decode_graph_rebuilds,
                (long long) result.streaming_decode_graph_build_ms,
                (long long) result.streaming_decode_graph_alloc_ms,
                (long long) result.streaming_decode_input_upload_ms,
                (long long) result.streaming_decode_graph_compute_ms,
                (long long) result.streaming_decode_output_read_ms);
    }
    fprintf(stderr, "  Total:     %6lld ms\n", (long long) result.t_total_ms);
}

static bool load_speech_codes_file(const std::string & path,
                                   qwen3_tts::speech_codes & codes,
                                   std::string & error) {
    std::string content;
    if (!read_text_file(path, content, error)) {
        return false;
    }
    auto parse_int_field = [&](const char * key, int32_t & out) {
        const std::string needle = std::string("\"") + key + "\"";
        const size_t key_pos = content.find(needle);
        if (key_pos == std::string::npos) {
            return;
        }
        const size_t colon = content.find(':', key_pos);
        if (colon == std::string::npos) {
            return;
        }
        std::stringstream ss(content.substr(colon + 1));
        int32_t value = 0;
        if (ss >> value) {
            out = value;
        }
    };
    parse_int_field("frames", codes.n_frames);
    parse_int_field("codebooks", codes.n_codebooks);

    const size_t codes_key = content.find("\"codes\"");
    if (codes_key != std::string::npos) {
        const size_t begin = content.find('[', codes_key);
        if (begin != std::string::npos) {
            int depth = 0;
            for (size_t i = begin; i < content.size(); ++i) {
                if (content[i] == '[') {
                    ++depth;
                } else if (content[i] == ']') {
                    --depth;
                    if (depth == 0) {
                        content = content.substr(begin + 1, i - begin - 1);
                        break;
                    }
                }
            }
        }
    }

    for (char & ch : content) {
        const unsigned char c = (unsigned char) ch;
        if (!std::isdigit(c) && ch != '-') {
            ch = ' ';
        }
    }

    std::stringstream ss(content);
    long long value = 0;
    codes.codes.clear();
    while (ss >> value) {
        if (value < std::numeric_limits<int32_t>::min() ||
            value > std::numeric_limits<int32_t>::max()) {
            error = "speech code out of int32 range in: " + path;
            return false;
        }
        codes.codes.push_back((int32_t) value);
    }
    if (codes.codes.empty()) {
        error = "no integer speech codes found in: " + path;
        return false;
    }
    if (codes.n_codebooks > 0 && codes.n_frames == 0) {
        codes.n_frames = (int32_t) (codes.codes.size() / (size_t) codes.n_codebooks);
    }
    return true;
}

static bool load_int32_list_file(const std::string & path,
                                 std::vector<int32_t> & values,
                                 std::string & error) {
    std::string content;
    if (!read_text_file(path, content, error)) {
        return false;
    }
    for (char & ch : content) {
        const unsigned char c = (unsigned char) ch;
        if (!std::isdigit(c) && ch != '-') {
            ch = ' ';
        }
    }

    std::stringstream ss(content);
    long long value = 0;
    values.clear();
    while (ss >> value) {
        if (value < std::numeric_limits<int32_t>::min() ||
            value > std::numeric_limits<int32_t>::max()) {
            error = "integer out of int32 range in: " + path;
            return false;
        }
        values.push_back((int32_t) value);
    }
    if (values.empty()) {
        error = "no integer values found in: " + path;
        return false;
    }
    return true;
}

void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir> -t <text>\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <dir>      Model directory (required)\n");
    fprintf(stderr, "  --codec-model <file>   Codec/tokenizer GGUF override\n");
    fprintf(stderr, "  --tokenizer-model <file> Alias for --codec-model\n");
    fprintf(stderr, "  -t, --text <text>      Text to synthesize (required)\n");
    fprintf(stderr, "  -o, --output <file>    Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -r, --reference <file> Reference audio for voice cloning\n");
    fprintf(stderr, "  --reference-text <text> Reference transcript for ICL voice cloning\n");
    fprintf(stderr, "  --reference-text-file <file> Read ICL reference transcript from file\n");
    fprintf(stderr, "  --reference-token-ids <file> Reference prompt token IDs for ICL voice cloning\n");
    fprintf(stderr, "  --reference-codes <file> Reference speech codes as integer text/JSON array\n");
    fprintf(stderr, "  --icl-prompt <file> Use precomputed full ICL voice prompt (.json)\n");
    fprintf(stderr, "  --dump-generated-codes <file> Save generated speech codes for debugging\n");
    fprintf(stderr, "  --dump-decoder-codes <file> Save decoder-input speech codes for debugging\n");
    fprintf(stderr, "  --speaker <name>       Named speaker (CustomVoice models)\n");
    fprintf(stderr, "  --speaker-embedding <file> Use precomputed speaker embedding (.json/.bin)\n");
    fprintf(stderr, "  --dump-speaker-embedding <file> Save extracted embedding from --reference\n");
    fprintf(stderr, "  --extract-speaker-embedding <file> Extract embedding from --reference and exit\n");
    fprintf(stderr, "  --extract-icl-prompt <file> Extract reusable ICL prompt from --reference and reference text\n");
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>          Top-p sampling (default: 1.0)\n");
    fprintf(stderr, "  --seed <n>             RNG seed for sampling (default: -1=random)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum audio tokens (default: 4096)\n");
    fprintf(stderr, "  --repeat <n>           Run synthesis n times in one process (default: 1)\n");
    fprintf(stderr, "  --bench-server <n>     Resident benchmark loop after loading models/voice artifacts\n");
    fprintf(stderr, "  --bench-warmup <n>     Warmup requests for --bench-server (default: 0)\n");
    fprintf(stderr, "  --stream               Use streaming synthesis API and collect chunks for WAV output\n");
    fprintf(stderr, "  --stream-chunk-sec <s> Streaming codec chunk duration (default: 1.0)\n");
    fprintf(stderr, "  --stream-left-context-sec <s> Streaming decoder left context (default: 2.0)\n");
    fprintf(stderr, "  --vocoder-left-context-sec <s> ICL vocoder reference context (default: 2.0)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  -l, --language <lang>  Language: auto,en,ru,zh,ja,ko,de,fr,es (default: auto)\n");
    fprintf(stderr, "  --instruction <instr>  Style/voice instruction\n");
    fprintf(stderr, "  --instruct <text>      Voice steering instructions (e.g. \"whispering\")\n");
    fprintf(stderr, "  -j, --threads <n>      CPU thread count (default: physical cores)\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello, world!\" -o hello.wav\n", program);
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -r reference.wav -o cloned.wav\n", program);
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" --speaker-embedding speaker.json -o cloned.wav\n", program);
}

int main(int argc, char ** argv) {
    std::vector<std::string> args;
#ifdef _WIN32
    args = get_utf8_argv();
    if (args.empty()) {
        args.reserve((size_t) argc);
        for (int i = 0; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }
    }
#else
    args.reserve((size_t) argc);
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
#endif

    std::string model_dir;
    std::string model_name;
    std::string tokenizer_model_path;
    std::string text;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string reference_text_file;
    std::string reference_token_ids_file;
    std::string reference_codes_file;
    std::string icl_prompt_file;
    std::string speaker_embedding_file;
    std::string dump_speaker_embedding_file;
    std::string extract_speaker_embedding_file;
    std::string extract_icl_prompt_file;
    int repeat_count = 1;
    int bench_server_count = 0;
    int bench_warmup_count = 0;
    bool use_streaming = false;
    bool threads_set = false;
    
    qwen3_tts::tts_params params;
    qwen3_tts::tts_streaming_params stream_params;
    params.print_progress = true;
    
    // Parse arguments
    for (int i = 1; i < (int) args.size(); i++) {
        std::string arg = args[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(args[0].c_str());
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing model directory\n");
                return 1;
            }
            model_dir = args[i];
        } else if (arg == "--model-name") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing model name\n");
                return 1;
            }
            model_name = args[i];
        } else if (arg == "--codec-model" || arg == "--tokenizer-model") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing codec/tokenizer model path\n");
                return 1;
            }
            tokenizer_model_path = args[i];
        } else if (arg == "-t" || arg == "--text") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing text\n");
                return 1;
            }
            text = args[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing output file\n");
                return 1;
            }
            output_file = args[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing reference audio\n");
                return 1;
            }
            reference_audio = args[i];
        } else if (arg == "--reference-text") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing reference text\n");
                return 1;
            }
            params.reference_text = args[i];
        } else if (arg == "--reference-text-file") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing reference text file\n");
                return 1;
            }
            reference_text_file = args[i];
        } else if (arg == "--reference-token-ids") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing reference token IDs file\n");
                return 1;
            }
            reference_token_ids_file = args[i];
        } else if (arg == "--reference-codes") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing reference codes file\n");
                return 1;
            }
            reference_codes_file = args[i];
        } else if (arg == "--icl-prompt") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing ICL prompt file\n");
                return 1;
            }
            icl_prompt_file = args[i];
        } else if (arg == "--dump-generated-codes") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing generated codes output file\n");
                return 1;
            }
            params.dump_generated_codes_path = args[i];
        } else if (arg == "--dump-decoder-codes") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing decoder codes output file\n");
                return 1;
            }
            params.dump_decoder_codes_path = args[i];
        } else if (arg == "--speaker") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing speaker name\n");
                return 1;
            }
            params.speaker = args[i];
        } else if (arg == "--speaker-embedding") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing speaker embedding file\n");
                return 1;
            }
            speaker_embedding_file = args[i];
        } else if (arg == "--dump-speaker-embedding") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing dump speaker embedding file\n");
                return 1;
            }
            dump_speaker_embedding_file = args[i];
        } else if (arg == "--extract-speaker-embedding") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing speaker embedding output file\n");
                return 1;
            }
            extract_speaker_embedding_file = args[i];
        } else if (arg == "--extract-icl-prompt") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing ICL prompt output file\n");
                return 1;
            }
            extract_icl_prompt_file = args[i];
        } else if (arg == "--temperature") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing temperature value\n");
                return 1;
            }
            params.temperature = std::stof(args[i]);
        } else if (arg == "--top-k") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing top-k value\n");
                return 1;
            }
            params.top_k = std::stoi(args[i]);
        } else if (arg == "--top-p") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing top-p value\n");
                return 1;
            }
            params.top_p = std::stof(args[i]);
        } else if (arg == "--seed") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing seed value\n");
                return 1;
            }
            params.seed = std::stoll(args[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing max-tokens value\n");
                return 1;
            }
            params.max_audio_tokens = std::stoi(args[i]);
        } else if (arg == "--repeat") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing repeat value\n");
                return 1;
            }
            repeat_count = std::stoi(args[i]);
        } else if (arg == "--bench-server") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing bench-server value\n");
                return 1;
            }
            bench_server_count = std::stoi(args[i]);
        } else if (arg == "--bench-warmup") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing bench-warmup value\n");
                return 1;
            }
            bench_warmup_count = std::stoi(args[i]);
        } else if (arg == "--stream") {
            use_streaming = true;
        } else if (arg == "--stream-chunk-sec") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing stream chunk duration\n");
                return 1;
            }
            stream_params.chunk_sec = std::stof(args[i]);
        } else if (arg == "--stream-left-context-sec") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing stream left context duration\n");
                return 1;
            }
            stream_params.left_context_sec = std::stof(args[i]);
        } else if (arg == "--vocoder-left-context-sec") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing vocoder left context duration\n");
                return 1;
            }
            params.vocoder_left_context_sec = std::stof(args[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing repetition-penalty value\n");
                return 1;
            }
            params.repetition_penalty = std::stof(args[i]);
        } else if (arg == "-l" || arg == "--language") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing language value\n");
                return 1;
            }
            std::string lang = args[i];
            if (lang == "auto")                          params.language_id = -1;
            else if (lang == "en" || lang == "english")  params.language_id = 2050;
            else if (lang == "ru" || lang == "russian")  params.language_id = 2069;
            else if (lang == "zh" || lang == "chinese")  params.language_id = 2055;
            else if (lang == "ja" || lang == "japanese")  params.language_id = 2058;
            else if (lang == "ko" || lang == "korean")   params.language_id = 2064;
            else if (lang == "de" || lang == "german")   params.language_id = 2053;
            else if (lang == "fr" || lang == "french")   params.language_id = 2061;
            else if (lang == "es" || lang == "spanish")  params.language_id = 2054;
            else if (lang == "it" || lang == "italian")  params.language_id = 2070;
            else if (lang == "pt" || lang == "portuguese") params.language_id = 2071;
            else {
                fprintf(stderr, "Error: unknown language '%s'. Supported: auto,en,ru,zh,ja,ko,de,fr,es,it,pt\n", lang.c_str());
                return 1;
            }
        } else if (arg == "--instruction" || arg == "--instruct") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing instruction value\n");
                return 1;
            }
            params.instruction = args[i];
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing threads value\n");
                return 1;
            }
            params.n_threads = std::stoi(args[i]);
            threads_set = true;
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(args[0].c_str());
            return 1;
        }
    }

    if (!qwen3_tts::set_cpu_thread_count(threads_set ? params.n_threads : qwen3_tts::default_cpu_thread_count(),
                                         threads_set)) {
        fprintf(stderr, "Error: failed to set CPU thread count\n");
        return 1;
    }
    params.n_threads = qwen3_tts::get_cpu_thread_count();
    
    // Validate required arguments
    if (model_dir.empty()) {
        fprintf(stderr, "Error: model directory is required\n");
        print_usage(args[0].c_str());
        return 1;
    }
    
    if (text.empty() && extract_speaker_embedding_file.empty() && extract_icl_prompt_file.empty()) {
        fprintf(stderr, "Error: text is required\n");
        print_usage(args[0].c_str());
        return 1;
    }
    if (repeat_count < 1) {
        fprintf(stderr, "Error: --repeat must be >= 1\n");
        return 1;
    }
    if (bench_server_count < 0 || bench_warmup_count < 0) {
        fprintf(stderr, "Error: --bench-server and --bench-warmup must be >= 0\n");
        return 1;
    }
    if (bench_server_count > 0 && repeat_count != 1) {
        fprintf(stderr, "Error: --bench-server and --repeat cannot be combined\n");
        return 1;
    }

    if (!reference_audio.empty() && !speaker_embedding_file.empty()) {
        fprintf(stderr, "Error: --reference and --speaker-embedding are mutually exclusive\n");
        return 1;
    }
    if (!icl_prompt_file.empty() &&
        (!reference_audio.empty() || !speaker_embedding_file.empty() || !params.speaker.empty())) {
        fprintf(stderr, "Error: --icl-prompt is mutually exclusive with --reference, --speaker-embedding, and --speaker\n");
        return 1;
    }
    if (!speaker_embedding_file.empty() && !params.speaker.empty()) {
        fprintf(stderr, "Error: --speaker and --speaker-embedding are mutually exclusive\n");
        return 1;
    }
    if (!reference_audio.empty() && !params.speaker.empty()) {
        fprintf(stderr, "Error: --reference and --speaker are mutually exclusive\n");
        return 1;
    }
    if (!dump_speaker_embedding_file.empty() && reference_audio.empty()) {
        fprintf(stderr, "Error: --dump-speaker-embedding requires --reference\n");
        return 1;
    }
    if (!extract_speaker_embedding_file.empty() && reference_audio.empty()) {
        fprintf(stderr, "Error: --extract-speaker-embedding requires --reference\n");
        return 1;
    }
    if (!extract_icl_prompt_file.empty() && reference_audio.empty()) {
        fprintf(stderr, "Error: --extract-icl-prompt requires --reference\n");
        return 1;
    }
    if (!reference_text_file.empty()) {
        std::string error;
        if (!read_text_file(reference_text_file, params.reference_text, error)) {
            fprintf(stderr, "Error: %s\n", error.c_str());
            return 1;
        }
        normalize_text_newlines(params.reference_text);
        trim_text_outer_whitespace(params.reference_text);
    }
    if (!reference_token_ids_file.empty()) {
        std::string error;
        if (!load_int32_list_file(reference_token_ids_file, params.reference_token_ids, error)) {
            fprintf(stderr, "Error: %s\n", error.c_str());
            return 1;
        }
    }
    if (!reference_codes_file.empty()) {
        qwen3_tts::speech_codes codes;
        std::string error;
        if (!load_speech_codes_file(reference_codes_file, codes, error)) {
            fprintf(stderr, "Error: %s\n", error.c_str());
            return 1;
        }
        params.reference_codes = std::move(codes);
    }
    if (!icl_prompt_file.empty() &&
        (!params.reference_text.empty() ||
         !params.reference_token_ids.empty() ||
         params.reference_codes.has_value())) {
        fprintf(stderr, "Error: --icl-prompt is mutually exclusive with --reference-text, --reference-token-ids, and --reference-codes\n");
        return 1;
    }
    const bool has_reference_prompt =
        !params.reference_text.empty() || !params.reference_token_ids.empty();
    const bool auto_reference_codes = has_reference_prompt &&
        !params.reference_codes.has_value() && !reference_audio.empty();
    if (has_reference_prompt &&
        !params.reference_codes.has_value() &&
        reference_audio.empty()) {
        fprintf(stderr, "Error: reference text/token IDs require --reference or --reference-codes\n");
        return 1;
    }
    if (params.reference_codes.has_value()) {
        if (!has_reference_prompt) {
            fprintf(stderr, "Error: --reference-codes requires --reference-text, --reference-text-file, or --reference-token-ids\n");
            return 1;
        }
        if (reference_audio.empty() && speaker_embedding_file.empty() && params.speaker.empty()) {
            fprintf(stderr, "Error: --reference-codes requires --reference, --speaker-embedding, or --speaker\n");
            return 1;
        }
    }
    if (use_streaming) {
        stream_params.generation = params;
        stream_params.collect_audio = true;
    }
    if (bench_server_count > 0) {
        params.print_progress = false;
        params.print_timing = false;
        stream_params.generation.print_progress = false;
        stream_params.generation.print_timing = false;
    }
    
    // Initialize TTS
    qwen3_tts::Qwen3TTS tts;

    if (!extract_speaker_embedding_file.empty()) {
        fprintf(stderr, "Loading speaker encoder from: %s\n", model_dir.c_str());
        if (!tts.load_speaker_encoder_only(model_dir, model_name)) {
            fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
            return 1;
        }

        std::vector<float> speaker_embedding;
        int64_t encode_ms = 0;
        if (!tts.extract_speaker_embedding(reference_audio, speaker_embedding, &encode_ms)) {
            fprintf(stderr, "Error: failed to extract speaker embedding: %s\n", tts.get_error().c_str());
            return 1;
        }
        if (!qwen3_tts::save_speaker_embedding_file(extract_speaker_embedding_file, speaker_embedding)) {
            fprintf(stderr, "Error: failed to save speaker embedding: %s\n", extract_speaker_embedding_file.c_str());
            return 1;
        }
        fprintf(stderr, "Speaker embedding saved to: %s\n", extract_speaker_embedding_file.c_str());
        fprintf(stderr, "Speaker encode: %lld ms (%zu floats)\n",
                (long long) encode_ms, speaker_embedding.size());
        return 0;
    }

    if (!extract_icl_prompt_file.empty()) {
        fprintf(stderr, "Loading ICL prompt encoders from: %s\n", model_dir.c_str());
        if (!tts.load_icl_prompt_encoder_only(model_dir, model_name, tokenizer_model_path)) {
            fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
            return 1;
        }

        qwen3_tts::icl_prompt prompt;
        int64_t encode_ms = 0;
        if (!tts.extract_icl_prompt(reference_audio, params.reference_text, prompt, &encode_ms)) {
            fprintf(stderr, "Error: failed to extract ICL prompt: %s\n", tts.get_error().c_str());
            return 1;
        }
        if (!qwen3_tts::save_icl_prompt_file(extract_icl_prompt_file, prompt)) {
            fprintf(stderr, "Error: failed to save ICL prompt: %s\n", extract_icl_prompt_file.c_str());
            return 1;
        }
        fprintf(stderr, "ICL prompt saved to: %s\n", extract_icl_prompt_file.c_str());
        fprintf(stderr, "ICL prompt encode: %lld ms (speaker=%zu floats, ref=%d frames x %d codebooks)\n",
                (long long) encode_ms,
                prompt.speaker_embedding.size(),
                prompt.reference_codes.n_frames,
                prompt.reference_codes.n_codebooks);
        return 0;
    }

    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    if (!tts.load_models(model_dir, model_name, tokenizer_model_path)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }
    
    // Set progress callback
    if (bench_server_count == 0) {
        tts.set_progress_callback([](int tokens, int max_tokens) {
            fprintf(stderr, "\rGenerating: %d/%d tokens", tokens, max_tokens);
        });
    }

    std::vector<float> speaker_embedding_from_file;
    qwen3_tts::icl_prompt icl_prompt_from_file;
    if (!icl_prompt_file.empty()) {
        if (!qwen3_tts::load_icl_prompt_file(icl_prompt_file, icl_prompt_from_file)) {
            fprintf(stderr, "Error: failed to load ICL prompt: %s\n", icl_prompt_file.c_str());
            return 1;
        }
        if (icl_prompt_from_file.speaker_embedding.size() != 1024 &&
            icl_prompt_from_file.speaker_embedding.size() != 2048) {
            fprintf(stderr,
                    "Warning: ICL prompt speaker embedding has %zu dimensions; expected 1024 (0.6B) or 2048 (1.7B)\n",
                    icl_prompt_from_file.speaker_embedding.size());
        }
    }
    if (!speaker_embedding_file.empty()) {
        if (!qwen3_tts::load_speaker_embedding_file(speaker_embedding_file, speaker_embedding_from_file)) {
            fprintf(stderr, "Error: failed to load speaker embedding: %s\n", speaker_embedding_file.c_str());
            return 1;
        }
        if (speaker_embedding_from_file.size() != 1024 && speaker_embedding_from_file.size() != 2048) {
            fprintf(stderr,
                    "Warning: speaker embedding has %zu dimensions; expected 1024 (0.6B) or 2048 (1.7B)\n",
                    speaker_embedding_from_file.size());
        }
    }

    auto run_synthesis_once = [&](int request_index,
                                  bool quiet,
                                  int64_t * ttfa_ms_out) -> qwen3_tts::tts_result {
        if (ttfa_ms_out) {
            *ttfa_ms_out = -1;
        }

        const int64_t request_start_ms = monotonic_time_ms();
        qwen3_tts::tts_audio_chunk_callback_t stream_callback =
            [&, request_start_ms](const qwen3_tts::tts_audio_chunk & chunk) {
                if (ttfa_ms_out && *ttfa_ms_out < 0) {
                    *ttfa_ms_out = monotonic_time_ms() - request_start_ms;
                }
                if (!quiet) {
                    fprintf(stderr,
                            "\rStreaming chunk: samples %lld-%lld (%d) frames %d-%d text %d-%d @ %d Hz",
                            (long long) chunk.start_sample,
                            (long long) chunk.end_sample,
                            chunk.n_samples,
                            chunk.start_frame,
                            chunk.end_frame,
                            chunk.start_text_byte,
                            chunk.end_text_byte,
                            chunk.sample_rate);
                }
                return true;
            };

        qwen3_tts::tts_result result;
        if (!icl_prompt_file.empty()) {
            qwen3_tts::tts_params icl_params = params;
            icl_params.reference_text = icl_prompt_from_file.reference_text;
            icl_params.reference_token_ids = icl_prompt_from_file.reference_token_ids;
            icl_params.reference_codes = icl_prompt_from_file.reference_codes;
            qwen3_tts::tts_streaming_params icl_stream_params = stream_params;
            icl_stream_params.generation = icl_params;
            if (!quiet) {
                fprintf(stderr, "Synthesizing with provided ICL prompt: \"%s\"\n", text.c_str());
                fprintf(stderr, "ICL prompt: %s (speaker=%zu floats, ref=%d frames x %d codebooks)\n",
                        icl_prompt_file.c_str(),
                        icl_prompt_from_file.speaker_embedding.size(),
                        icl_prompt_from_file.reference_codes.n_frames,
                        icl_prompt_from_file.reference_codes.n_codebooks);
            }
            result = use_streaming
                ? tts.synthesize_with_speaker_embedding_streaming(text, icl_prompt_from_file.speaker_embedding,
                                                                  stream_callback, icl_stream_params)
                : tts.synthesize_with_speaker_embedding(text, icl_prompt_from_file.speaker_embedding, icl_params);
        } else if (!speaker_embedding_file.empty()) {
            if (!quiet) {
                fprintf(stderr, "Synthesizing with provided speaker embedding: \"%s\"\n", text.c_str());
                fprintf(stderr, "Speaker embedding: %s (%zu floats)\n",
                        speaker_embedding_file.c_str(), speaker_embedding_from_file.size());
            }
            result = use_streaming
                ? tts.synthesize_with_speaker_embedding_streaming(text, speaker_embedding_from_file,
                                                                  stream_callback, stream_params)
                : tts.synthesize_with_speaker_embedding(text, speaker_embedding_from_file, params);
        } else if (reference_audio.empty()) {
            if (!quiet) {
                fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
            }
            result = use_streaming
                ? tts.synthesize_streaming(text, stream_callback, stream_params)
                : tts.synthesize(text, params);
        } else {
            std::vector<float> speaker_embedding;
            int64_t encode_ms = 0;
            if (!quiet) {
                fprintf(stderr, "Synthesizing with voice cloning: \"%s\"\n", text.c_str());
                fprintf(stderr, "Reference audio: %s\n", reference_audio.c_str());
            }
            if (auto_reference_codes) {
                if (!dump_speaker_embedding_file.empty() && request_index == 0) {
                    if (!tts.extract_speaker_embedding(reference_audio, speaker_embedding, &encode_ms)) {
                        result.error_msg = "failed to extract speaker embedding: " + tts.get_error();
                        return result;
                    }
                    if (!qwen3_tts::save_speaker_embedding_file(dump_speaker_embedding_file, speaker_embedding)) {
                        result.error_msg = "failed to save speaker embedding: " + dump_speaker_embedding_file;
                        return result;
                    }
                    if (!quiet) {
                        fprintf(stderr, "Speaker embedding saved to: %s\n", dump_speaker_embedding_file.c_str());
                        fprintf(stderr, "Speaker embedding will be extracted again for ICL synthesis.\n");
                    }
                }
                result = use_streaming
                    ? tts.synthesize_with_voice_streaming(text, reference_audio, stream_callback, stream_params)
                    : tts.synthesize_with_voice(text, reference_audio, params);
            } else {
                if (!tts.extract_speaker_embedding(reference_audio, speaker_embedding, &encode_ms)) {
                    result.error_msg = "failed to extract speaker embedding: " + tts.get_error();
                    return result;
                }
                if (!quiet && params.print_timing) {
                    fprintf(stderr, "  Speaker embedding extracted in %lld ms (%zu floats)\n",
                            (long long) encode_ms, speaker_embedding.size());
                }
                if (!dump_speaker_embedding_file.empty() && request_index == 0) {
                    if (!qwen3_tts::save_speaker_embedding_file(dump_speaker_embedding_file, speaker_embedding)) {
                        result.error_msg = "failed to save speaker embedding: " + dump_speaker_embedding_file;
                        return result;
                    }
                    if (!quiet) {
                        fprintf(stderr, "Speaker embedding saved to: %s\n", dump_speaker_embedding_file.c_str());
                    }
                }
                result = use_streaming
                    ? tts.synthesize_with_speaker_embedding_streaming(text, speaker_embedding,
                                                                      stream_callback, stream_params)
                    : tts.synthesize_with_speaker_embedding(text, speaker_embedding, params);
                if (result.success) {
                    result.t_encode_ms = encode_ms;
                    result.t_total_ms += encode_ms;
                }
            }
        }

        return result;
    };

    if (bench_server_count > 0) {
        const int total_requests = bench_warmup_count + bench_server_count;
        for (int request = 0; request < total_requests; ++request) {
            const bool warmup = request < bench_warmup_count;
            int64_t ttfa_ms = -1;
            const int64_t t0 = monotonic_time_ms();
            qwen3_tts::tts_result result = run_synthesis_once(request, true, &ttfa_ms);
            const int64_t wall_ms = monotonic_time_ms() - t0;
            const int bench_iteration = warmup ? request + 1 : request - bench_warmup_count + 1;
            if (!result.success) {
                fprintf(stderr, "\nError: %s\n", result.error_msg.c_str());
                print_bench_json("qwen3-tts.cpp", use_streaming ? "resident_streaming" : "resident",
                                 bench_iteration, warmup, 1, result, wall_ms, ttfa_ms, "");
                return 1;
            }

            std::string request_output_file;
            if (!warmup) {
                request_output_file = output_file_for_repeat(output_file,
                                                             request - bench_warmup_count,
                                                             bench_server_count);
                if (!qwen3_tts::save_audio_file(request_output_file, result.audio, result.sample_rate)) {
                    fprintf(stderr, "Error: failed to save output file: %s\n", request_output_file.c_str());
                    return 1;
                }
            }
            print_bench_json("qwen3-tts.cpp", use_streaming ? "resident_streaming" : "resident",
                             bench_iteration, warmup, 0, result, wall_ms, ttfa_ms,
                             request_output_file);
        }
        return 0;
    }

    for (int repeat = 0; repeat < repeat_count; ++repeat) {
        if (repeat_count > 1) {
            fprintf(stderr, "\nRepeat %d/%d\n", repeat + 1, repeat_count);
        }

        qwen3_tts::tts_result result = run_synthesis_once(repeat, false, nullptr);

        if (!result.success) {
            fprintf(stderr, "\nError: %s\n", result.error_msg.c_str());
            return 1;
        }

        fprintf(stderr, "\n");

        const std::string repeat_output_file = output_file_for_repeat(output_file, repeat, repeat_count);
        if (!qwen3_tts::save_audio_file(repeat_output_file, result.audio, result.sample_rate)) {
            fprintf(stderr, "Error: failed to save output file: %s\n", repeat_output_file.c_str());
            return 1;
        }

        fprintf(stderr, "Output saved to: %s\n", repeat_output_file.c_str());
        fprintf(stderr, "Audio duration: %.2f seconds\n",
                (float) result.audio.size() / result.sample_rate);

        if (params.print_timing) {
            print_result_timing(result);
        }
    }
    
    return 0;
}
