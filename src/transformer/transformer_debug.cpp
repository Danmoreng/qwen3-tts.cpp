#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace qwen3_tts {
namespace transformer_internal {

std::string normalize_speaker_name(const std::string & name) {
    std::string out = name;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return out;
}

int32_t parse_env_i32(const char * name, int32_t default_value, int32_t min_value, int32_t max_value) {
    const char * raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return default_value;
    }
    char * end_ptr = nullptr;
    long parsed = strtol(raw, &end_ptr, 10);
    if (!end_ptr || *end_ptr != '\0') {
        return default_value;
    }
    if (parsed < min_value) parsed = min_value;
    if (parsed > max_value) parsed = max_value;
    return (int32_t) parsed;
}

debug_trace_config init_debug_trace_config() {
    debug_trace_config cfg;
    const char * dir_env = std::getenv("QWEN3_TTS_DEBUG_DUMP_DIR");
    if (!dir_env || dir_env[0] == '\0') {
        return cfg;
    }

    cfg.dir = dir_env;
    cfg.max_frames = parse_env_i32("QWEN3_TTS_DEBUG_DUMP_MAX_FRAMES", 1, 1, 512);
    cfg.max_code_steps = parse_env_i32("QWEN3_TTS_DEBUG_DUMP_MAX_CODE_STEPS", 15, 1, 15);

    std::error_code ec;
    std::filesystem::create_directories(cfg.dir, ec);
    if (ec) {
        fprintf(stderr, "  Debug trace disabled: failed to create '%s' (%s)\n",
                cfg.dir.c_str(), ec.message().c_str());
        return {};
    }

    cfg.enabled = true;

    std::filesystem::path manifest_path = std::filesystem::path(cfg.dir) / "manifest.tsv";
    std::ofstream manifest(manifest_path.string(), std::ios::out | std::ios::trunc);
    if (manifest.is_open()) {
        manifest << "name\tdtype\tcount\tshape\n";
        manifest.close();
    }

    std::filesystem::path info_path = std::filesystem::path(cfg.dir) / "trace_info.txt";
    std::ofstream info(info_path.string(), std::ios::out | std::ios::trunc);
    if (info.is_open()) {
        info << "QWEN3_TTS_DEBUG_DUMP_MAX_FRAMES=" << cfg.max_frames << "\n";
        info << "QWEN3_TTS_DEBUG_DUMP_MAX_CODE_STEPS=" << cfg.max_code_steps << "\n";
        info.close();
    }

    fprintf(stderr, "  Debug trace enabled: %s\n", cfg.dir.c_str());
    return cfg;
}

const debug_trace_config & get_debug_trace_config() {
    static debug_trace_config cfg = init_debug_trace_config();
    return cfg;
}

bool debug_trace_should_dump_frame(const debug_trace_config & cfg, int32_t frame) {
    return cfg.enabled && frame >= 0 && frame < cfg.max_frames;
}

static std::string format_shape(const std::vector<int64_t> & shape) {
    std::ostringstream oss;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << "x";
        }
        oss << shape[i];
    }
    return oss.str();
}

static void debug_trace_append_manifest(const debug_trace_config & cfg,
                                        const std::string & name,
                                        const char * dtype,
                                        size_t count,
                                        const std::vector<int64_t> & shape) {
    if (!cfg.enabled) {
        return;
    }
    std::filesystem::path manifest_path = std::filesystem::path(cfg.dir) / "manifest.tsv";
    std::ofstream manifest(manifest_path.string(), std::ios::out | std::ios::app);
    if (!manifest.is_open()) {
        return;
    }
    manifest << name << "\t" << dtype << "\t" << count << "\t" << format_shape(shape) << "\n";
}

template <typename T>
static void debug_trace_write_bin_impl(const debug_trace_config & cfg,
                                       const std::string & name,
                                       const T * data,
                                       size_t count,
                                       const char * dtype,
                                       const std::vector<int64_t> & shape) {
    if (!cfg.enabled || !data || count == 0) {
        return;
    }

    std::filesystem::path out_path = std::filesystem::path(cfg.dir) / name;
    FILE * f = fopen(out_path.string().c_str(), "wb");
    if (!f) {
        return;
    }
    fwrite(data, sizeof(T), count, f);
    fclose(f);

    debug_trace_append_manifest(cfg, name, dtype, count, shape);
}

void debug_trace_write_bin(const debug_trace_config & cfg,
                           const std::string & name,
                           const float * data,
                           size_t count,
                           const char * dtype,
                           const std::vector<int64_t> & shape) {
    debug_trace_write_bin_impl(cfg, name, data, count, dtype, shape);
}

void debug_trace_write_bin(const debug_trace_config & cfg,
                           const std::string & name,
                           const int32_t * data,
                           size_t count,
                           const char * dtype,
                           const std::vector<int64_t> & shape) {
    debug_trace_write_bin_impl(cfg, name, data, count, dtype, shape);
}

void debug_trace_write_text_line(const debug_trace_config & cfg, const std::string & line) {
    if (!cfg.enabled) {
        return;
    }
    std::filesystem::path info_path = std::filesystem::path(cfg.dir) / "trace_info.txt";
    std::ofstream info(info_path.string(), std::ios::out | std::ios::app);
    if (!info.is_open()) {
        return;
    }
    info << line << "\n";
}

} // namespace transformer_internal
} // namespace qwen3_tts
