#include "qwen3_tts.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

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

} // namespace qwen3_tts
