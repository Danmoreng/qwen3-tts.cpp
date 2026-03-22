#include "qwen3_tts.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir> -t <text>\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <dir>      Model directory (required)\n");
    fprintf(stderr, "  -t, --text <text>      Text to synthesize (required)\n");
    fprintf(stderr, "  -o, --output <file>    Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -r, --reference <file> Reference audio for voice cloning\n");
    fprintf(stderr, "  --reference-text <text> Matching transcript for --reference\n");
    fprintf(stderr, "  --reference-text-file <file> Load matching transcript from a file\n");
    fprintf(stderr, "  --speaker <name>       Named speaker (CustomVoice models)\n");
    fprintf(stderr, "  --speaker-embedding <file> Use precomputed speaker embedding (.json/.bin)\n");
    fprintf(stderr, "  --dump-speaker-embedding <file> Save extracted embedding from --reference\n");
    fprintf(stderr, "  --voice-clone-prompt <file> Reuse a saved voice clone prompt asset\n");
    fprintf(stderr, "  --dump-voice-clone-prompt <file> Save a prompt asset created from --reference\n");
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>          Top-p sampling (default: 1.0)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum audio tokens (default: 4096)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  -l, --language <lang>  Language: en,ru,zh,ja,ko,de,fr,es (default: en)\n");
    fprintf(stderr, "  --instruction <instr>  Style/voice instruction\n");
    fprintf(stderr, "  --instruct <text>      Voice steering instructions (e.g. \"whispering\")\n");
    fprintf(stderr, "  -j, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello, world!\" -o hello.wav\n", program);
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -r reference.wav -o cloned.wav\n", program);
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" --speaker-embedding speaker.json -o cloned.wav\n", program);
}

int main(int argc, char ** argv) {
    std::string model_dir;
    std::string model_name;
    std::string text;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string reference_text;
    std::string reference_text_file;
    std::string speaker_embedding_file;
    std::string dump_speaker_embedding_file;
    std::string voice_clone_prompt_file;
    std::string dump_voice_clone_prompt_file;
    
    qwen3_tts::tts_params params;
    params.print_progress = true;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing model directory\n");
                return 1;
            }
            model_dir = argv[i];
        } else if (arg == "--model-name") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing model name\n");
                return 1;
            }
            model_name = argv[i];
        } else if (arg == "-t" || arg == "--text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing text\n");
                return 1;
            }
            text = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing output file\n");
                return 1;
            }
            output_file = argv[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing reference audio\n");
                return 1;
            }
            reference_audio = argv[i];
        } else if (arg == "--reference-text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing reference text\n");
                return 1;
            }
            reference_text = argv[i];
        } else if (arg == "--reference-text-file") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing reference text file\n");
                return 1;
            }
            reference_text_file = argv[i];
        } else if (arg == "--speaker") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing speaker name\n");
                return 1;
            }
            params.speaker = argv[i];
        } else if (arg == "--speaker-embedding") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing speaker embedding file\n");
                return 1;
            }
            speaker_embedding_file = argv[i];
        } else if (arg == "--dump-speaker-embedding") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing dump speaker embedding file\n");
                return 1;
            }
            dump_speaker_embedding_file = argv[i];
        } else if (arg == "--voice-clone-prompt") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing voice clone prompt file\n");
                return 1;
            }
            voice_clone_prompt_file = argv[i];
        } else if (arg == "--dump-voice-clone-prompt") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing output voice clone prompt file\n");
                return 1;
            }
            dump_voice_clone_prompt_file = argv[i];
        } else if (arg == "--temperature") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing temperature value\n");
                return 1;
            }
            params.temperature = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing top-k value\n");
                return 1;
            }
            params.top_k = std::stoi(argv[i]);
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing top-p value\n");
                return 1;
            }
            params.top_p = std::stof(argv[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing max-tokens value\n");
                return 1;
            }
            params.max_audio_tokens = std::stoi(argv[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing repetition-penalty value\n");
                return 1;
            }
            params.repetition_penalty = std::stof(argv[i]);
        } else if (arg == "-l" || arg == "--language") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing language value\n");
                return 1;
            }
            std::string lang = argv[i];
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
            if (++i >= argc) {
                fprintf(stderr, "Error: missing instruction value\n");
                return 1;
            }
            params.instruction = argv[i];
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing threads value\n");
                return 1;
            }
            params.n_threads = std::stoi(argv[i]);
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (model_dir.empty()) {
        fprintf(stderr, "Error: model directory is required\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (text.empty()) {
        fprintf(stderr, "Error: text is required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!reference_audio.empty() && !speaker_embedding_file.empty()) {
        fprintf(stderr, "Error: --reference and --speaker-embedding are mutually exclusive\n");
        return 1;
    }
    if (!voice_clone_prompt_file.empty() &&
        (!reference_audio.empty() || !speaker_embedding_file.empty() || !params.speaker.empty())) {
        fprintf(stderr, "Error: --voice-clone-prompt is mutually exclusive with --reference, --speaker-embedding, and --speaker\n");
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
    if (!reference_text.empty() && !reference_text_file.empty()) {
        fprintf(stderr, "Error: --reference-text and --reference-text-file are mutually exclusive\n");
        return 1;
    }
    if (!dump_voice_clone_prompt_file.empty() && reference_audio.empty()) {
        fprintf(stderr, "Error: --dump-voice-clone-prompt requires --reference\n");
        return 1;
    }
    if ((!reference_text.empty() || !reference_text_file.empty()) && reference_audio.empty()) {
        fprintf(stderr, "Error: reference text requires --reference\n");
        return 1;
    }

    if (!reference_text_file.empty()) {
        FILE * f = fopen(reference_text_file.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "Error: failed to open reference text file: %s\n", reference_text_file.c_str());
            return 1;
        }
        std::string file_text;
        char buf[4096];
        size_t nread = 0;
        while ((nread = fread(buf, 1, sizeof(buf), f)) > 0) {
            file_text.append(buf, buf + nread);
        }
        fclose(f);
        reference_text = file_text;
    }
    
    // Initialize TTS
    qwen3_tts::Qwen3TTS tts;
    
    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    if (!tts.load_models(model_dir, model_name)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }
    
    // Set progress callback
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\rGenerating: %d/%d tokens", tokens, max_tokens);
    });
    
    // Generate speech
    qwen3_tts::tts_result result;
    
    if (!voice_clone_prompt_file.empty()) {
        qwen3_tts::voice_clone_prompt_asset asset;
        std::string load_error;
        if (!qwen3_tts::load_voice_clone_prompt_file(voice_clone_prompt_file, asset, &load_error)) {
            fprintf(stderr, "Error: failed to load voice clone prompt: %s\n", load_error.c_str());
            return 1;
        }
        fprintf(stderr, "Synthesizing with voice clone prompt: \"%s\"\n", text.c_str());
        fprintf(stderr, "Voice clone prompt: %s\n", voice_clone_prompt_file.c_str());
        result = tts.synthesize_with_voice_clone_prompt(text, asset, params);
    } else if (!speaker_embedding_file.empty()) {
        std::vector<float> speaker_embedding;
        if (!qwen3_tts::load_speaker_embedding_file(speaker_embedding_file, speaker_embedding)) {
            fprintf(stderr, "Error: failed to load speaker embedding: %s\n", speaker_embedding_file.c_str());
            return 1;
        }
        if (speaker_embedding.size() != 1024 && speaker_embedding.size() != 2048) {
            fprintf(stderr,
                    "Warning: speaker embedding has %zu dimensions; expected 1024 (0.6B) or 2048 (1.7B)\n",
                    speaker_embedding.size());
        }
        fprintf(stderr, "Synthesizing with provided speaker embedding: \"%s\"\n", text.c_str());
        fprintf(stderr, "Speaker embedding: %s (%zu floats)\n",
                speaker_embedding_file.c_str(), speaker_embedding.size());
        result = tts.synthesize_with_speaker_embedding(text, speaker_embedding, params);
    } else if (reference_audio.empty()) {
        fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
        result = tts.synthesize(text, params);
    } else {
        fprintf(stderr, "Synthesizing with voice cloning: \"%s\"\n", text.c_str());
        fprintf(stderr, "Reference audio: %s\n", reference_audio.c_str());
        if (!reference_text.empty()) {
            fprintf(stderr, "Reference text: %zu characters\n", reference_text.size());
            qwen3_tts::voice_clone_prompt_asset asset;
            if (!tts.create_voice_clone_prompt(reference_audio, reference_text, asset)) {
                fprintf(stderr, "\nError: failed to create voice clone prompt: %s\n", tts.get_error().c_str());
                return 1;
            }
            if (!dump_voice_clone_prompt_file.empty()) {
                std::string save_error;
                if (!qwen3_tts::save_voice_clone_prompt_file(dump_voice_clone_prompt_file, asset, &save_error)) {
                    fprintf(stderr, "\nError: failed to save voice clone prompt: %s\n", save_error.c_str());
                    return 1;
                }
                fprintf(stderr, "Voice clone prompt saved to: %s\n", dump_voice_clone_prompt_file.c_str());
            }
            result = tts.synthesize_with_voice_clone_prompt(text, asset, params);
        } else {
            std::vector<float> speaker_embedding;
            int64_t encode_ms = 0;
            if (!tts.extract_speaker_embedding(reference_audio, speaker_embedding, &encode_ms)) {
                fprintf(stderr, "\nError: failed to extract speaker embedding: %s\n", tts.get_error().c_str());
                return 1;
            }
            if (params.print_timing) {
                fprintf(stderr, "  Speaker embedding extracted in %lld ms (%zu floats)\n",
                        (long long) encode_ms, speaker_embedding.size());
            }
            if (!dump_speaker_embedding_file.empty()) {
                if (!qwen3_tts::save_speaker_embedding_file(dump_speaker_embedding_file, speaker_embedding)) {
                    fprintf(stderr, "\nError: failed to save speaker embedding: %s\n",
                            dump_speaker_embedding_file.c_str());
                    return 1;
                }
                fprintf(stderr, "Speaker embedding saved to: %s\n", dump_speaker_embedding_file.c_str());
            }
            if (!dump_voice_clone_prompt_file.empty()) {
                qwen3_tts::voice_clone_prompt_asset asset;
                asset.format_version = 1;
                asset.prompt_mode = qwen3_tts::voice_clone_prompt_mode::audio_only;
                asset.model_kind = tts.get_model_capabilities().model_type;
                asset.model_name = model_name.empty() ? "active-model" : model_name;
                asset.speaker_embedding_dim = (int32_t) speaker_embedding.size();
                asset.speaker_embedding = speaker_embedding;
                std::string save_error;
                if (!qwen3_tts::save_voice_clone_prompt_file(dump_voice_clone_prompt_file, asset, &save_error)) {
                    fprintf(stderr, "\nError: failed to save audio-only voice clone prompt: %s\n", save_error.c_str());
                    return 1;
                }
                fprintf(stderr, "Audio-only voice clone prompt saved to: %s\n", dump_voice_clone_prompt_file.c_str());
            }
            result = tts.synthesize_with_speaker_embedding(text, speaker_embedding, params);
            if (result.success) {
                result.t_encode_ms = encode_ms;
            }
        }
    }
    
    if (!result.success) {
        fprintf(stderr, "\nError: %s\n", result.error_msg.c_str());
        return 1;
    }
    
    fprintf(stderr, "\n");
    
    // Save output
    if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
        fprintf(stderr, "Error: failed to save output file: %s\n", output_file.c_str());
        return 1;
    }
    
    fprintf(stderr, "Output saved to: %s\n", output_file.c_str());
    fprintf(stderr, "Audio duration: %.2f seconds\n", 
            (float)result.audio.size() / result.sample_rate);
    
    // Print timing
    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Load:      %6lld ms\n", (long long)result.t_load_ms);
        fprintf(stderr, "  Tokenize:  %6lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Encode:    %6lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Generate:  %6lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Decode:    %6lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:     %6lld ms\n", (long long)result.t_total_ms);
    }
    
    return 0;
}
