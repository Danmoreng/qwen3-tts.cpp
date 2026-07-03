#include "qwen3_tts.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace qwen3_tts {
namespace {

std::string read_file_text(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

size_t find_key(const std::string & text, const char * key) {
    return text.find(std::string("\"") + key + "\"");
}

bool parse_int_field(const std::string & text, const char * key, int32_t & value) {
    const size_t key_pos = find_key(text, key);
    if (key_pos == std::string::npos) {
        return false;
    }
    const size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos) {
        return false;
    }
    size_t pos = colon + 1;
    while (pos < text.size() && std::isspace((unsigned char) text[pos])) {
        ++pos;
    }
    char * end = nullptr;
    const long parsed = std::strtol(text.c_str() + pos, &end, 10);
    if (end == text.c_str() + pos ||
        parsed < std::numeric_limits<int32_t>::min() ||
        parsed > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    value = (int32_t) parsed;
    return true;
}

bool find_array_bounds(const std::string & text, const char * key, size_t & begin, size_t & end) {
    const size_t key_pos = find_key(text, key);
    if (key_pos == std::string::npos) {
        return false;
    }
    begin = text.find('[', key_pos);
    if (begin == std::string::npos) {
        return false;
    }
    int depth = 0;
    for (size_t i = begin; i < text.size(); ++i) {
        if (text[i] == '[') {
            ++depth;
        } else if (text[i] == ']') {
            --depth;
            if (depth == 0) {
                end = i;
                return true;
            }
        }
    }
    return false;
}

template <typename T, typename ParseFn>
bool parse_number_array(const std::string & text, const char * key,
                        std::vector<T> & values, ParseFn parse) {
    size_t begin = 0;
    size_t end = 0;
    if (!find_array_bounds(text, key, begin, end)) {
        return false;
    }
    values.clear();
    size_t pos = begin + 1;
    while (pos < end) {
        while (pos < end && (std::isspace((unsigned char) text[pos]) || text[pos] == ',')) {
            ++pos;
        }
        if (pos >= end) {
            break;
        }
        char * parsed_end = nullptr;
        T value{};
        if (!parse(text.c_str() + pos, &parsed_end, value) || parsed_end == text.c_str() + pos) {
            ++pos;
            continue;
        }
        values.push_back(value);
        pos = (size_t) (parsed_end - text.c_str());
    }
    return !values.empty();
}

bool parse_float_value(const char * start, char ** end, float & value) {
    value = std::strtof(start, end);
    return *end != start;
}

bool parse_int32_value(const char * start, char ** end, int32_t & value) {
    const long parsed = std::strtol(start, end, 10);
    if (*end == start ||
        parsed < std::numeric_limits<int32_t>::min() ||
        parsed > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    value = (int32_t) parsed;
    return true;
}

bool parse_json_string_field(const std::string & text, const char * key, std::string & value) {
    const size_t key_pos = find_key(text, key);
    if (key_pos == std::string::npos) {
        return false;
    }
    const size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos) {
        return false;
    }
    size_t pos = text.find('"', colon + 1);
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;
    value.clear();
    while (pos < text.size()) {
        const char c = text[pos++];
        if (c == '"') {
            return true;
        }
        if (c != '\\') {
            value.push_back(c);
            continue;
        }
        if (pos >= text.size()) {
            return false;
        }
        const char esc = text[pos++];
        switch (esc) {
            case '"': value.push_back('"'); break;
            case '\\': value.push_back('\\'); break;
            case '/': value.push_back('/'); break;
            case 'b': value.push_back('\b'); break;
            case 'f': value.push_back('\f'); break;
            case 'n': value.push_back('\n'); break;
            case 'r': value.push_back('\r'); break;
            case 't': value.push_back('\t'); break;
            default: value.push_back(esc); break;
        }
    }
    return false;
}

void write_json_string(std::ostream & out, const std::string & value) {
    out << '"';
    for (const unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0') << (int) c
                        << std::dec << std::setfill(' ');
                } else {
                    out << (char) c;
                }
                break;
        }
    }
    out << '"';
}

template <typename T>
void write_number_array(std::ostream & out, const std::vector<T> & values, int per_line) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        if (per_line > 0 && i % (size_t) per_line == 0) {
            out << "\n    ";
        }
        out << values[i];
    }
    if (!values.empty()) {
        out << '\n';
    }
    out << "  ]";
}

} // namespace

bool load_icl_prompt_file(const std::string & path, icl_prompt & prompt) {
    const std::string text = read_file_text(path);
    if (text.empty()) {
        fprintf(stderr, "ERROR: ICL prompt file is empty or unreadable: %s\n", path.c_str());
        return false;
    }

    icl_prompt parsed;
    if (!parse_json_string_field(text, "reference_text", parsed.reference_text)) {
        parsed.reference_text.clear();
    }
    parse_number_array<int32_t>(text, "reference_token_ids", parsed.reference_token_ids, parse_int32_value);
    if (!parse_number_array<float>(text, "speaker_embedding", parsed.speaker_embedding, parse_float_value)) {
        fprintf(stderr, "ERROR: ICL prompt is missing speaker_embedding: %s\n", path.c_str());
        return false;
    }
    if (!parse_number_array<int32_t>(text, "codes", parsed.reference_codes.codes, parse_int32_value)) {
        fprintf(stderr, "ERROR: ICL prompt is missing reference codes: %s\n", path.c_str());
        return false;
    }
    parse_int_field(text, "frames", parsed.reference_codes.n_frames);
    parse_int_field(text, "codebooks", parsed.reference_codes.n_codebooks);
    if (parsed.reference_codes.n_codebooks <= 0) {
        parsed.reference_codes.n_codebooks = 16;
    }
    if (parsed.reference_codes.n_frames <= 0 &&
        parsed.reference_codes.n_codebooks > 0 &&
        !parsed.reference_codes.codes.empty()) {
        parsed.reference_codes.n_frames =
            (int32_t) (parsed.reference_codes.codes.size() /
                       (size_t) parsed.reference_codes.n_codebooks);
    }
    if (parsed.reference_text.empty() && parsed.reference_token_ids.empty()) {
        fprintf(stderr, "ERROR: ICL prompt requires reference_text or reference_token_ids: %s\n", path.c_str());
        return false;
    }

    prompt = std::move(parsed);
    return true;
}

bool save_icl_prompt_file(const std::string & path, const icl_prompt & prompt) {
    if (prompt.speaker_embedding.empty()) {
        fprintf(stderr, "ERROR: Refusing to save ICL prompt without speaker embedding\n");
        return false;
    }
    if (prompt.reference_codes.codes.empty()) {
        fprintf(stderr, "ERROR: Refusing to save ICL prompt without reference codes\n");
        return false;
    }
    if (prompt.reference_text.empty() && prompt.reference_token_ids.empty()) {
        fprintf(stderr, "ERROR: Refusing to save ICL prompt without reference text or token IDs\n");
        return false;
    }

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        fprintf(stderr, "ERROR: Cannot create ICL prompt file: %s\n", path.c_str());
        return false;
    }

    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    out << "{\n";
    out << "  \"format\": \"qwen3_tts_icl_prompt_v1\",\n";
    out << "  \"reference_text\": ";
    write_json_string(out, prompt.reference_text);
    out << ",\n";
    out << "  \"reference_token_ids\": ";
    write_number_array(out, prompt.reference_token_ids, 16);
    out << ",\n";
    out << "  \"speaker_embedding\": ";
    write_number_array(out, prompt.speaker_embedding, 8);
    out << ",\n";
    out << "  \"reference_codes\": {\n";
    out << "    \"frames\": " << prompt.reference_codes.n_frames << ",\n";
    out << "    \"codebooks\": " << prompt.reference_codes.n_codebooks << ",\n";
    out << "    \"codes\": ";
    write_number_array(out, prompt.reference_codes.codes, prompt.reference_codes.n_codebooks);
    out << "\n  }\n";
    out << "}\n";
    return out.good();
}

} // namespace qwen3_tts
