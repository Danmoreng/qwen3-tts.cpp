#include "qwen3_tts.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>

namespace qwen3_tts {

namespace fs = std::filesystem;

namespace {

std::string shell_quote(const std::string & value) {
#ifdef _WIN32
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\\\"";
        else out.push_back(c);
    }
    out.push_back('"');
    return out;
#else
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') out += "'\\''";
        else out.push_back(c);
    }
    out.push_back('\'');
    return out;
#endif
}

std::string repo_root_from_source() {
    return fs::path(__FILE__).parent_path().parent_path().string();
}

std::string temp_path_for(const char * stem) {
    const auto base = fs::temp_directory_path() /
        (std::string("qwen3_tts_") + stem + "_" + std::to_string(std::rand()));
    return base.string();
}

} // namespace

Qwen3TTS::Qwen3TTS() = default;

Qwen3TTS::~Qwen3TTS() = default;

void Qwen3TTS::set_progress_callback(tts_progress_callback_t callback) {
    progress_callback_ = callback;
}

bool Qwen3TTS::validate_voice_clone_prompt(const voice_clone_prompt_asset & asset,
                                           voice_clone_prompt_validation * out) const {
    voice_clone_prompt_validation result;
    result.model_compatible = false;

    if (asset.format_version != 1) {
        result.error_msg = "Unsupported voice clone prompt format_version";
    } else if (asset.speaker_embedding.empty()) {
        result.error_msg = "Voice clone prompt is missing speaker_embedding";
    } else if (asset.speaker_embedding_dim <= 0 ||
               (int32_t) asset.speaker_embedding.size() != asset.speaker_embedding_dim) {
        result.error_msg = "Voice clone prompt speaker_embedding_dim does not match payload size";
    } else if (asset.prompt_mode == voice_clone_prompt_mode::reference_aware &&
               (asset.reference_text.empty() ||
                asset.reference_codebooks <= 0 ||
                asset.reference_frames <= 0 ||
                asset.reference_codes.empty())) {
        result.error_msg = "Reference-aware prompt is missing reference_text or reference_codes";
    } else if (asset.prompt_mode == voice_clone_prompt_mode::reference_aware &&
               (int32_t) asset.reference_codes.size() != asset.reference_frames * asset.reference_codebooks) {
        result.error_msg = "Reference-aware prompt reference_codes size does not match frames/codebooks";
    } else {
        result.valid = true;
        if (!models_loaded_) {
            result.model_compatible = true;
        } else {
            result.model_compatible = (asset.model_kind.empty() ||
                                       asset.model_kind == transformer_.get_config().tts_model_type) &&
                                      asset.speaker_embedding_dim == transformer_.get_config().hidden_size;
            if (!result.model_compatible) {
                result.error_msg = "Voice clone prompt is incompatible with the loaded model";
            }
        }
    }

    if (out) {
        *out = result;
    }
    return result.valid && result.model_compatible;
}

bool Qwen3TTS::create_voice_clone_prompt(const std::string & reference_audio,
                                         const std::string & reference_text,
                                         voice_clone_prompt_asset & asset) {
    asset = {};

    if (!models_loaded_) {
        error_msg_ = "Models not loaded";
        return false;
    }

    if (reference_audio.empty()) {
        error_msg_ = "Reference audio is required";
        return false;
    }

    asset.format_version = 1;
    asset.model_kind = transformer_.get_config().tts_model_type;
    asset.model_name = fs::path(tts_model_path_).filename().string();
    asset.speaker_embedding_dim = transformer_.get_config().hidden_size;

    if (reference_text.empty()) {
        asset.prompt_mode = voice_clone_prompt_mode::audio_only;
        if (!extract_speaker_embedding(reference_audio, asset.speaker_embedding, nullptr)) {
            return false;
        }
        return true;
    }

    const std::string repo_root = repo_root_from_source();
    const fs::path helper = fs::path(repo_root) / "scripts" / "create_voice_clone_prompt.py";
    if (!fs::exists(helper)) {
        error_msg_ = "Missing helper script for reference-aware prompt creation";
        return false;
    }

    const char * python_env = std::getenv("QWEN3_TTS_PYTHON");
#ifdef _WIN32
    const char * default_python = "python";
#else
    const char * default_python = "python3";
#endif
    const std::string python = python_env && python_env[0] ? python_env : default_python;
    const std::string output_json = temp_path_for("voice_clone_prompt") + ".json";

    std::ostringstream cmd;
    cmd << shell_quote(python)
        << " " << shell_quote(helper.string())
        << " --model-dir " << shell_quote(fs::path(tts_model_path_).parent_path().string())
        << " --reference-audio " << shell_quote(reference_audio)
        << " --reference-text " << shell_quote(reference_text)
        << " --output " << shell_quote(output_json);

    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        error_msg_ = "Reference-aware prompt helper failed; set QWEN3_TTS_PYTHON or install qwen-tts dependencies";
        std::error_code ec;
        fs::remove(output_json, ec);
        return false;
    }

    std::string load_error;
    if (!load_voice_clone_prompt_file(output_json, asset, &load_error)) {
        error_msg_ = "Failed to load helper-generated prompt asset: " + load_error;
        std::error_code ec;
        fs::remove(output_json, ec);
        return false;
    }

    std::error_code ec;
    fs::remove(output_json, ec);

    voice_clone_prompt_validation validation;
    if (!validate_voice_clone_prompt(asset, &validation)) {
        error_msg_ = validation.error_msg;
        return false;
    }

    return true;
}

} // namespace qwen3_tts
