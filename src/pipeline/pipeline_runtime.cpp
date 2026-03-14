#include "pipeline/pipeline_internal.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace qwen3_tts {
namespace pipeline_internal {
namespace {

bool env_flag_enabled(const char * name) {
    const char * v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return false;
    }

    auto ieq = [](const char * a, const char * b) -> bool {
        if (!a || !b) {
            return false;
        }
        while (*a && *b) {
            if (std::tolower((unsigned char) *a) != std::tolower((unsigned char) *b)) {
                return false;
            }
            ++a;
            ++b;
        }
        return *a == '\0' && *b == '\0';
    };

    if (strcmp(v, "0") == 0) {
        return false;
    }
    if (ieq(v, "false") || ieq(v, "off") || ieq(v, "no")) {
        return false;
    }
    return true;
}

void ggml_log_callback_filtered(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;

    if (level == GGML_LOG_LEVEL_DEBUG && !env_flag_enabled("QWEN3_TTS_GGML_DEBUG")) {
        return;
    }

    if (text) {
        fputs(text, stderr);
        fflush(stderr);
    }
}

} // namespace

void configure_ggml_logging_once() {
    static bool configured = false;
    if (configured) {
        return;
    }
    configured = true;
    ggml_log_set(ggml_log_callback_filtered, nullptr);
}

int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool get_process_memory_snapshot(process_memory_snapshot & out) {
#ifdef __APPLE__
    mach_task_basic_info_data_t basic_info = {};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&basic_info), &basic_count) != KERN_SUCCESS) {
        return false;
    }
    out.rss_bytes = (uint64_t) basic_info.resident_size;

    task_vm_info_data_t vm_info = {};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm_info), &vm_count) == KERN_SUCCESS) {
        out.phys_footprint_bytes = (uint64_t) vm_info.phys_footprint;
    } else {
        out.phys_footprint_bytes = out.rss_bytes;
    }
    return true;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc = {};
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc),
                              sizeof(pmc))) {
        return false;
    }
    out.rss_bytes = (uint64_t) pmc.WorkingSetSize;
    out.phys_footprint_bytes = (uint64_t) pmc.PrivateUsage;
    return true;
#else
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return false;
    }
    out.rss_bytes = (uint64_t) usage.ru_maxrss * 1024ULL;
    out.phys_footprint_bytes = out.rss_bytes;
    return true;
#endif
}

std::string format_bytes(uint64_t bytes) {
    static const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    double val = (double) bytes;
    int unit = 0;
    while (val >= 1024.0 && unit < 4) {
        val /= 1024.0;
        ++unit;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", val, units[unit]);
    return std::string(buf);
}

void log_memory_usage(const char * label) {
    process_memory_snapshot mem;
    if (!get_process_memory_snapshot(mem)) {
        fprintf(stderr, "  [mem] %-24s unavailable\n", label);
        return;
    }
    fprintf(stderr, "  [mem] %-24s rss=%s  phys=%s\n",
            label, format_bytes(mem.rss_bytes).c_str(),
            format_bytes(mem.phys_footprint_bytes).c_str());
}

void resample_linear(const float * input, int input_len, int input_rate,
                     std::vector<float> & output, int output_rate) {
    double ratio = (double) input_rate / output_rate;
    int output_len = (int) ((double) input_len / ratio);
    output.resize(output_len);

    for (int i = 0; i < output_len; ++i) {
        double src_idx = i * ratio;
        int idx0 = (int) src_idx;
        int idx1 = idx0 + 1;
        double frac = src_idx - idx0;

        if (idx1 >= input_len) {
            output[i] = input[input_len - 1];
        } else {
            output[i] = (float) ((1.0 - frac) * input[idx0] + frac * input[idx1]);
        }
    }
}

} // namespace pipeline_internal
} // namespace qwen3_tts
