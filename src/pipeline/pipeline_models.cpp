#include "qwen3_tts.h"
#include "gguf_loader.h"
#include "pipeline/pipeline_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <filesystem>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace qwen3_tts {
namespace fs = std::filesystem;
using pipeline_internal::configure_ggml_logging_once;
using pipeline_internal::get_time_ms;
using pipeline_internal::log_memory_usage;

bool Qwen3TTS::load_models(const std::string & model_dir, const std::string & model_name) {
    configure_ggml_logging_once();

    int64_t t_start = get_time_ms();
    log_memory_usage("load/start");

    transformer_.unload_model();
    audio_encoder_.unload_model();
    speech_encoder_.unload_model();
    audio_decoder_.unload_model();
    voice_prompt_cache_ = {};
    encoder_loaded_ = false;
    transformer_loaded_ = false;
    speech_encoder_loaded_ = false;
    decoder_loaded_ = false;

    std::string tts_model_path;
    std::string tokenizer_model_path;
    std::vector<fs::path> tokenizer_candidates;

    if (fs::exists(model_dir) && fs::is_directory(model_dir)) {
        for (const auto & entry : fs::directory_iterator(model_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string filename = entry.path().filename().string();
            std::string filename_lower = filename;
            std::transform(filename_lower.begin(), filename_lower.end(), filename_lower.begin(), ::tolower);
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".gguf") {
                if (filename_lower.find("tokenizer") != std::string::npos) {
                    tokenizer_candidates.push_back(entry.path());
                } else if (filename_lower.find("qwen3-tts") != std::string::npos ||
                           filename_lower.find("qwen-talker") != std::string::npos ||
                           filename_lower.find("full") != std::string::npos) {
                    if (!model_name.empty()) {
                        if (filename.find(model_name) != std::string::npos) {
                            tts_model_path = entry.path().string();
                        }
                    } else if (tts_model_path.empty() || filename.find("0.6b") != std::string::npos) {
                        tts_model_path = entry.path().string();
                    }
                }
            }
        }
    }

    if (tts_model_path.empty()) {
        if (!model_name.empty()) {
            tts_model_path = model_dir + "/" + model_name;
        } else {
            tts_model_path = model_dir + "/qwen3-tts-0.6b-f16.gguf";
        }
    }
    const auto choose_tokenizer = [&](const bool prefer_qwen_talker) -> std::string {
        const std::vector<std::string> preferred_exact = prefer_qwen_talker
            ? std::vector<std::string>{
                "qwen-tokenizer-12hz-q8_0.gguf",
                "qwen-tokenizer-12hz-bf16.gguf",
                "qwen-tokenizer-12hz-f32.gguf",
                "qwen3-tts-tokenizer-f16.gguf",
            }
            : std::vector<std::string>{
                "qwen3-tts-tokenizer-f16.gguf",
                "qwen-tokenizer-12hz-q8_0.gguf",
                "qwen-tokenizer-12hz-bf16.gguf",
                "qwen-tokenizer-12hz-f32.gguf",
            };
        for (const auto & preferred : preferred_exact) {
            for (const auto & path : tokenizer_candidates) {
                std::string filename = path.filename().string();
                std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);
                if (filename == preferred) {
                    return path.string();
                }
            }
        }
        for (const auto & path : tokenizer_candidates) {
            std::string filename = path.filename().string();
            std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);
            if (prefer_qwen_talker && filename.find("qwen-tokenizer-12hz") != std::string::npos) {
                return path.string();
            }
            if (!prefer_qwen_talker && filename.find("qwen3-tts-tokenizer") != std::string::npos) {
                return path.string();
            }
        }
        if (!tokenizer_candidates.empty()) {
            return tokenizer_candidates.front().string();
        }
        return {};
    };

    if (tokenizer_model_path.empty()) {
        std::string tts_model_name_lower = fs::path(tts_model_path).filename().string();
        std::transform(tts_model_name_lower.begin(), tts_model_name_lower.end(), tts_model_name_lower.begin(), ::tolower);
        const bool prefer_qwen_talker_tokenizer = tts_model_name_lower.find("qwen-talker") != std::string::npos;
        tokenizer_model_path = choose_tokenizer(prefer_qwen_talker_tokenizer);
    }
    if (tokenizer_model_path.empty()) {
        tokenizer_model_path = model_dir + "/qwen3-tts-tokenizer-f16.gguf";
    }

    fprintf(stderr, "  TTS model path:       %s\n", tts_model_path.c_str());
    fprintf(stderr, "  Tokenizer model path: %s\n", tokenizer_model_path.c_str());

    tts_model_path_ = tts_model_path;
    tokenizer_model_path_ = tokenizer_model_path;
    decoder_model_path_ = tokenizer_model_path;
    encoder_loaded_ = false;
    speech_encoder_loaded_ = false;
    transformer_loaded_ = false;
    decoder_loaded_ = false;

    const char * low_mem_env = std::getenv("QWEN3_TTS_LOW_MEM");
    low_mem_mode_ = low_mem_env && low_mem_env[0] != '\0' && low_mem_env[0] != '0';
    if (low_mem_mode_) {
        fprintf(stderr, "  Low-memory mode enabled (lazy decoder + component unloads)\n");
    }

    fprintf(stderr, "Loading TTS model from %s...\n", tts_model_path.c_str());

    int64_t t_tokenizer_start = get_time_ms();
    {
        GGUFLoader loader;
        if (!loader.open(tts_model_path)) {
            error_msg_ = "Failed to open TTS model: " + loader.get_error();
            return false;
        }

        if (!tokenizer_.load_from_gguf(loader.get_ctx())) {
            error_msg_ = "Failed to load text tokenizer: " + tokenizer_.get_error();
            return false;
        }
        fprintf(stderr, "  Text tokenizer loaded: vocab_size=%d (%lld ms)\n",
                tokenizer_.get_config().vocab_size,
                (long long) (get_time_ms() - t_tokenizer_start));
    }
    log_memory_usage("load/after-tokenizer");

    fprintf(stderr, "  Speaker encoder: deferred (lazy load)\n");

    int64_t t_transformer_start = get_time_ms();
    if (!transformer_.load_model(tts_model_path)) {
        error_msg_ = "Failed to load TTS transformer: " + transformer_.get_error();
        return false;
    }
    transformer_loaded_ = true;
    fprintf(stderr, "  TTS transformer loaded: hidden_size=%d, n_layers=%d (%lld ms)\n",
            transformer_.get_config().hidden_size, transformer_.get_config().n_layers,
            (long long) (get_time_ms() - t_transformer_start));
    log_memory_usage("load/after-transformer");

    if (!low_mem_mode_) {
        fprintf(stderr, "Loading vocoder from %s...\n", tokenizer_model_path.c_str());
        int64_t t_decoder_start = get_time_ms();
        if (!audio_decoder_.load_model(tokenizer_model_path)) {
            error_msg_ = "Failed to load vocoder: " + audio_decoder_.get_error();
            return false;
        }
        decoder_loaded_ = true;
        fprintf(stderr, "  Vocoder loaded: sample_rate=%d, n_codebooks=%d (%lld ms)\n",
                audio_decoder_.get_config().sample_rate, audio_decoder_.get_config().n_codebooks,
                (long long) (get_time_ms() - t_decoder_start));
        log_memory_usage("load/after-vocoder");
    } else {
        fprintf(stderr, "  Vocoder: deferred (lazy load)\n");
    }

    models_loaded_ = true;

    int64_t t_end = get_time_ms();
    fprintf(stderr, "All models loaded in %lld ms\n", (long long) (t_end - t_start));
    log_memory_usage("load/end");

    return true;
}

std::vector<std::string> Qwen3TTS::get_available_speakers() const {
    std::vector<std::string> speakers;
    const auto & speaker_map = transformer_.get_config().speaker_id_map;
    speakers.reserve(speaker_map.size());
    for (const auto & it : speaker_map) {
        speakers.push_back(it.first);
    }
    return speakers;
}

tts_model_capabilities Qwen3TTS::get_model_capabilities() const {
    tts_model_capabilities caps;
    caps.loaded = models_loaded_;
    if (!models_loaded_) {
        return caps;
    }

    const auto & cfg = transformer_.get_config();
    caps.model_type = cfg.tts_model_type;
    caps.speaker_embedding_dim = cfg.hidden_size;
    caps.speaker_count = (int32_t) cfg.speaker_id_map.size();
    caps.supports_named_speakers = caps.speaker_count > 0;
    caps.supports_voice_clone = (cfg.tts_model_type == "base");

    if (cfg.has_supports_instruction) {
        caps.supports_instruction = cfg.supports_instruction;
    } else if (cfg.tts_model_type == "custom_voice") {
        caps.supports_instruction = cfg.hidden_size >= 2048;
    } else if (cfg.tts_model_type == "voice_design") {
        caps.supports_instruction = true;
    } else {
        caps.supports_instruction = false;
    }

    return caps;
}

} // namespace qwen3_tts
