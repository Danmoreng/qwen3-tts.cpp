#include "qwen3_tts.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>

namespace qwen3_tts {

static std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

static bool has_json_extension(const std::string & path) {
    const size_t pos = path.find_last_of('.');
    if (pos == std::string::npos) {
        return false;
    }
    const std::string ext = to_lower_ascii(path.substr(pos));
    return ext == ".json";
}

static bool parse_embedding_text(const std::string & text, std::vector<float> & embedding) {
    std::string cleaned = text;
    for (char & c : cleaned) {
        if (c == '[' || c == ']' || c == ',' || c == ';') {
            c = ' ';
        }
    }

    std::istringstream iss(cleaned);
    float value = 0.0f;
    embedding.clear();
    while (iss >> value) {
        embedding.push_back(value);
    }
    return !embedding.empty();
}

static std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

static std::string prompt_mode_to_string(voice_clone_prompt_mode mode) {
    return mode == voice_clone_prompt_mode::reference_aware ? "reference_aware" : "audio_only";
}

static bool prompt_mode_from_string(const std::string & value, voice_clone_prompt_mode & mode) {
    if (value == "audio_only") {
        mode = voice_clone_prompt_mode::audio_only;
        return true;
    }
    if (value == "reference_aware") {
        mode = voice_clone_prompt_mode::reference_aware;
        return true;
    }
    return false;
}

static size_t skip_ws(const std::string & text, size_t pos) {
    while (pos < text.size() && std::isspace((unsigned char) text[pos])) {
        ++pos;
    }
    return pos;
}

static bool find_json_key(const std::string & text, const char * key, size_t & value_pos) {
    const std::string needle = std::string("\"") + key + "\"";
    const size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }
    size_t colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return false;
    }
    value_pos = skip_ws(text, colon + 1);
    return value_pos < text.size();
}

static bool extract_json_string(const std::string & text,
                                const char * key,
                                std::string & out) {
    size_t pos = 0;
    if (!find_json_key(text, key, pos) || pos >= text.size() || text[pos] != '"') {
        return false;
    }
    ++pos;
    out.clear();
    bool escaped = false;
    for (; pos < text.size(); ++pos) {
        const char c = text[pos];
        if (escaped) {
            switch (c) {
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case '\\': out.push_back('\\'); break;
                case '"': out.push_back('"'); break;
                default: out.push_back(c); break;
            }
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            return true;
        }
        out.push_back(c);
    }
    return false;
}

template <typename T>
static bool extract_json_number(const std::string & text,
                                const char * key,
                                T & out) {
    size_t pos = 0;
    if (!find_json_key(text, key, pos)) {
        return false;
    }
    size_t end = pos;
    while (end < text.size()) {
        const char c = text[end];
        if (!(std::isdigit((unsigned char) c) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')) {
            break;
        }
        ++end;
    }
    if (end == pos) {
        return false;
    }
    std::istringstream iss(text.substr(pos, end - pos));
    iss >> out;
    return !iss.fail();
}

template <typename T>
static bool extract_json_array_numbers(const std::string & text,
                                       const char * key,
                                       std::vector<T> & out) {
    size_t pos = 0;
    if (!find_json_key(text, key, pos) || pos >= text.size() || text[pos] != '[') {
        return false;
    }
    ++pos;
    out.clear();
    while (pos < text.size()) {
        pos = skip_ws(text, pos);
        if (pos >= text.size()) {
            return false;
        }
        if (text[pos] == ']') {
            return true;
        }

        size_t end = pos;
        while (end < text.size()) {
            const char c = text[end];
            if (!(std::isdigit((unsigned char) c) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')) {
                break;
            }
            ++end;
        }
        if (end == pos) {
            return false;
        }

        std::istringstream iss(text.substr(pos, end - pos));
        T value {};
        iss >> value;
        if (iss.fail()) {
            return false;
        }
        out.push_back(value);

        pos = skip_ws(text, end);
        if (pos >= text.size()) {
            return false;
        }
        if (text[pos] == ',') {
            ++pos;
            continue;
        }
        if (text[pos] == ']') {
            return true;
        }
        return false;
    }
    return false;
}

bool load_speaker_embedding_file(const std::string & path,
                                 std::vector<float> & embedding) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "ERROR: Cannot open speaker embedding file: %s\n", path.c_str());
        return false;
    }

    std::string data((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    if (data.empty()) {
        fprintf(stderr, "ERROR: Speaker embedding file is empty: %s\n", path.c_str());
        return false;
    }

    if (has_json_extension(path) || data.find('[') != std::string::npos) {
        if (!parse_embedding_text(data, embedding)) {
            fprintf(stderr, "ERROR: Failed to parse speaker embedding JSON/text: %s\n", path.c_str());
            return false;
        }
        return true;
    }

    if (data.size() % sizeof(float) != 0) {
        fprintf(stderr, "ERROR: Speaker embedding binary size is not a multiple of 4 bytes: %s\n", path.c_str());
        return false;
    }

    embedding.resize(data.size() / sizeof(float));
    memcpy(embedding.data(), data.data(), data.size());
    return true;
}

bool save_speaker_embedding_file(const std::string & path,
                                 const std::vector<float> & embedding) {
    if (embedding.empty()) {
        fprintf(stderr, "ERROR: Refusing to save empty speaker embedding\n");
        return false;
    }

    if (has_json_extension(path)) {
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out) {
            fprintf(stderr, "ERROR: Cannot create speaker embedding JSON file: %s\n", path.c_str());
            return false;
        }
        out << std::setprecision(std::numeric_limits<float>::max_digits10);
        out << "[\n";
        for (size_t i = 0; i < embedding.size(); ++i) {
            out << "  " << embedding[i];
            if (i + 1 != embedding.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "]\n";
        return true;
    }

    std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out) {
        fprintf(stderr, "ERROR: Cannot create speaker embedding binary file: %s\n", path.c_str());
        return false;
    }
    out.write(reinterpret_cast<const char *>(embedding.data()),
              (std::streamsize) (embedding.size() * sizeof(float)));
    return out.good();
}

bool load_voice_clone_prompt_file(const std::string & path,
                                  voice_clone_prompt_asset & asset,
                                  std::string * error_msg) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error_msg) *error_msg = "Cannot open voice clone prompt file: " + path;
        return false;
    }

    std::string data((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    if (data.empty()) {
        if (error_msg) *error_msg = "Voice clone prompt file is empty: " + path;
        return false;
    }

    std::string prompt_mode;
    if (!extract_json_number(data, "format_version", asset.format_version) ||
        !extract_json_string(data, "prompt_mode", prompt_mode) ||
        !extract_json_string(data, "model_kind", asset.model_kind) ||
        !extract_json_string(data, "model_name", asset.model_name) ||
        !extract_json_number(data, "speaker_embedding_dim", asset.speaker_embedding_dim) ||
        !extract_json_array_numbers(data, "speaker_embedding", asset.speaker_embedding)) {
        if (error_msg) *error_msg = "Failed to parse required voice clone prompt fields";
        return false;
    }

    if (!prompt_mode_from_string(prompt_mode, asset.prompt_mode)) {
        if (error_msg) *error_msg = "Unsupported prompt_mode: " + prompt_mode;
        return false;
    }

    asset.reference_text.clear();
    asset.reference_codebooks = 0;
    asset.reference_frames = 0;
    asset.reference_codes.clear();

    if (asset.prompt_mode == voice_clone_prompt_mode::reference_aware) {
        if (!extract_json_string(data, "reference_text", asset.reference_text) ||
            !extract_json_number(data, "reference_codebooks", asset.reference_codebooks) ||
            !extract_json_number(data, "reference_frames", asset.reference_frames) ||
            !extract_json_array_numbers(data, "reference_codes", asset.reference_codes)) {
            if (error_msg) *error_msg = "Failed to parse reference-aware prompt fields";
            return false;
        }
    }

    return true;
}

bool save_voice_clone_prompt_file(const std::string & path,
                                  const voice_clone_prompt_asset & asset,
                                  std::string * error_msg) {
    if (asset.speaker_embedding.empty()) {
        if (error_msg) *error_msg = "Refusing to save voice clone prompt without speaker_embedding";
        return false;
    }

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        if (error_msg) *error_msg = "Cannot create voice clone prompt file: " + path;
        return false;
    }

    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    out << "{\n";
    out << "  \"format_version\": " << asset.format_version << ",\n";
    out << "  \"prompt_mode\": \"" << prompt_mode_to_string(asset.prompt_mode) << "\",\n";
    out << "  \"model_kind\": \"" << json_escape(asset.model_kind) << "\",\n";
    out << "  \"model_name\": \"" << json_escape(asset.model_name) << "\",\n";
    out << "  \"speaker_embedding_dim\": " << asset.speaker_embedding_dim << ",\n";
    out << "  \"speaker_embedding\": [\n";
    for (size_t i = 0; i < asset.speaker_embedding.size(); ++i) {
        out << "    " << asset.speaker_embedding[i];
        if (i + 1 != asset.speaker_embedding.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]";

    if (asset.prompt_mode == voice_clone_prompt_mode::reference_aware) {
        out << ",\n";
        out << "  \"reference_text\": \"" << json_escape(asset.reference_text) << "\",\n";
        out << "  \"reference_codebooks\": " << asset.reference_codebooks << ",\n";
        out << "  \"reference_frames\": " << asset.reference_frames << ",\n";
        out << "  \"reference_codes\": [\n";
        for (size_t i = 0; i < asset.reference_codes.size(); ++i) {
            out << "    " << asset.reference_codes[i];
            if (i + 1 != asset.reference_codes.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ]\n";
    } else {
        out << "\n";
    }

    out << "}\n";
    if (!out.good()) {
        if (error_msg) *error_msg = "Failed while writing voice clone prompt file: " + path;
        return false;
    }
    return true;
}

} // namespace qwen3_tts
