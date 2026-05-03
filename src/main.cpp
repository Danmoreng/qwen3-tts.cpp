#include "qwen3_tts.h"

#include <cstdio>
#include <cstring>
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

void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir> -t <text>\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <dir>      Model directory (required)\n");
    fprintf(stderr, "  -t, --text <text>      Text to synthesize (required)\n");
    fprintf(stderr, "  -o, --output <file>    Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -r, --reference <file> Reference audio for voice cloning\n");
    fprintf(stderr, "  --speaker <name>       Named speaker (CustomVoice models)\n");
    fprintf(stderr, "  --speaker-embedding <file> Use precomputed speaker embedding (.json/.bin)\n");
    fprintf(stderr, "  --dump-speaker-embedding <file> Save extracted embedding from --reference\n");
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>          Top-p sampling (default: 1.0)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum audio tokens (default: 4096)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  -l, --language <lang>  Language: en,ru,zh,ja,ko,de,fr,es (default: en)\n");
    fprintf(stderr, "  --instruction <instr>  Style/voice instruction\n");
    fprintf(stderr, "  --instruct <text>      Voice steering instructions (e.g. \"whispering\")\n");
    fprintf(stderr, "  --daemon               Persistent mode (read TEXT|OUTPUT|... from stdin)\n");
    fprintf(stderr, "  -j, --threads <n>      Number of threads (default: 4)\n");
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
    std::string text;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string speaker_embedding_file;
    std::string dump_speaker_embedding_file;
    
    qwen3_tts::tts_params params;
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
        } else if (arg == "--max-tokens") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing max-tokens value\n");
                return 1;
            }
            params.max_audio_tokens = std::stoi(args[i]);
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
            if (lang == "en" || lang == "english")       params.language_id = 2050;
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
                fprintf(stderr, "Error: unknown language '%s'. Supported: en,ru,zh,ja,ko,de,fr,es,it,pt\n", lang.c_str());
                return 1;
            }
        } else if (arg == "--instruction" || arg == "--instruct") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing instruction value\n");
                return 1;
            }
            params.instruction = args[i];
        } else if (arg == "--info") {
            params.print_timing = false;
        } else if (arg == "--list-speakers") {
            params.print_timing = false;
        } else if (arg == "--daemon") {
            // recognized but handled later
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= (int) args.size()) {
                fprintf(stderr, "Error: missing threads value\n");
                return 1;
            }
            params.n_threads = std::stoi(args[i]);
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(args[0].c_str());
            return 1;
        }
    }
    
    // Validate required arguments
    if (model_dir.empty()) {
        fprintf(stderr, "Error: model directory is required\n");
        print_usage(args[0].c_str());
        return 1;
    }
    
    bool info_mode = false;
    bool list_mode = false;
    bool daemon_mode = false;
    for (const auto & arg : args) {
        if (arg == "--info") info_mode = true;
        if (arg == "--list-speakers") list_mode = true;
        if (arg == "--daemon") daemon_mode = true;
    }

    if (text.empty() && !info_mode && !list_mode && !daemon_mode) {
        fprintf(stderr, "Error: text is required\n");
        print_usage(args[0].c_str());
        return 1;
    }

    if (!reference_audio.empty() && !speaker_embedding_file.empty()) {
        fprintf(stderr, "Error: --reference and --speaker-embedding are mutually exclusive\n");
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
    
    // Initialize TTS
    qwen3_tts::Qwen3TTS tts;
    
    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    if (!tts.load_models(model_dir, model_name)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }

    if (info_mode) {
        auto caps = tts.get_model_capabilities();
        printf("{\n");
        printf("  \"model_type\": \"%s\",\n", caps.model_type.c_str());
        printf("  \"supports_voice_clone\": %s,\n", caps.supports_voice_clone ? "true" : "false");
        printf("  \"supports_named_speakers\": %s,\n", caps.supports_named_speakers ? "true" : "false");
        printf("  \"supports_instruction\": %s,\n", caps.supports_instruction ? "true" : "false");
        printf("  \"speaker_count\": %d\n", caps.speaker_count);
        printf("}\n");
        return 0;
    }

    if (list_mode) {
        auto speakers = tts.get_available_speakers();
        printf("[\n");
        for (size_t i = 0; i < speakers.size(); ++i) {
            printf("  \"%s\"%s\n", speakers[i].c_str(), (i == speakers.size() - 1) ? "" : ",");
        }
        printf("]\n");
        return 0;
    }

    if (daemon_mode) {
        fprintf(stderr, "Daemon mode active. Waiting for input on stdin...\n");
        printf("READY\n");
        fflush(stdout);

        char line[8192];
        while (fgets(line, sizeof(line), stdin)) {
            // Format: TEXT|OUTPUT|SPEAKER|REF|INSTRUCT
            std::string l(line);
            if (l.empty() || l == "\n") continue;
            if (l.back() == '\n') l.pop_back();

            std::vector<std::string> parts;
            size_t start = 0, end;
            while ((end = l.find('|', start)) != std::string::npos) {
                parts.push_back(l.substr(start, end - start));
                start = end + 1;
            }
            parts.push_back(l.substr(start));

            if (parts.size() < 2) {
                fprintf(stderr, "Daemon error: invalid input format. Expected: TEXT|OUTPUT|...\n");
                continue;
            }

            std::string d_text = parts[0];
            std::string d_out = parts[1];
            qwen3_tts::tts_params d_params = params; // start with defaults
            std::string d_ref;
            std::string d_embed;

            if (parts.size() > 2 && !parts[2].empty()) d_params.speaker = parts[2];
            if (parts.size() > 3 && !parts[3].empty()) d_ref = parts[3];
            if (parts.size() > 4 && !parts[4].empty()) d_params.instruction = parts[4];
            if (parts.size() > 5 && !parts[5].empty()) d_embed = parts[5];

            qwen3_tts::tts_result d_res;
            if (!d_ref.empty()) {
                std::vector<float> emb;
                if (tts.extract_speaker_embedding(d_ref, emb, nullptr)) {
                    if (!d_embed.empty()) {
                        qwen3_tts::save_speaker_embedding_file(d_embed, emb);
                    }
                    d_res = tts.synthesize_with_speaker_embedding(d_text, emb, d_params);
                } else {
                    d_res.success = false;
                    d_res.error_msg = "Failed to extract embedding from: " + d_ref;
                }
            } else if (!d_embed.empty()) {
                std::vector<float> emb;
                if (qwen3_tts::load_speaker_embedding_file(d_embed, emb)) {
                    d_res = tts.synthesize_with_speaker_embedding(d_text, emb, d_params);
                } else {
                    d_res.success = false;
                    d_res.error_msg = "Failed to load embedding: " + d_embed;
                }
            } else {
                d_res = tts.synthesize(d_text, d_params);
            }

            if (d_res.success) {
                if (qwen3_tts::save_audio_file(d_out, d_res.audio, d_res.sample_rate)) {
                    printf("DONE|%s\n", d_out.c_str());
                } else {
                    printf("ERROR|Failed to save WAV\n");
                }
            } else {
                printf("ERROR|%s\n", d_res.error_msg.c_str());
            }
            fflush(stdout);
        }
        return 0;
    }
    
    // Set progress callback
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\rGenerating: %d/%d tokens", tokens, max_tokens);
    });
    
    // Generate speech (original non-daemon logic follows)
    qwen3_tts::tts_result result;
    
    if (!speaker_embedding_file.empty()) {
        std::vector<float> speaker_embedding;
        if (!qwen3_tts::load_speaker_embedding_file(speaker_embedding_file, speaker_embedding)) {
            fprintf(stderr, "Error: failed to load speaker embedding: %s\n", speaker_embedding_file.c_str());
            return 1;
        }
        result = tts.synthesize_with_speaker_embedding(text, speaker_embedding, params);
    } else if (reference_audio.empty()) {
        result = tts.synthesize(text, params);
    } else {
        std::vector<float> speaker_embedding;
        if (!tts.extract_speaker_embedding(reference_audio, speaker_embedding, nullptr)) {
            fprintf(stderr, "\nError: failed to extract speaker embedding\n");
            return 1;
        }
        if (!dump_speaker_embedding_file.empty()) {
            qwen3_tts::save_speaker_embedding_file(dump_speaker_embedding_file, speaker_embedding);
        }
        result = tts.synthesize_with_speaker_embedding(text, speaker_embedding, params);
    }
    
    if (!result.success) {
        fprintf(stderr, "\nError: %s\n", result.error_msg.c_str());
        return 1;
    }
    
    if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
        fprintf(stderr, "Error: failed to save output file\n");
        return 1;
    }
    
    fprintf(stderr, "\nOutput saved to: %s\n", output_file.c_str());
    return 0;
}
