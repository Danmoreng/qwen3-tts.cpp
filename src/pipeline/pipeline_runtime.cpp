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

namespace {

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
    if (!input || input_len <= 0 || input_rate <= 0 || output_rate <= 0) {
        output.clear();
        return;
    }
    if (input_rate == output_rate) {
        output.assign(input, input + input_len);
        return;
    }

    const double src_per_dst = (double) input_rate / (double) output_rate;
    const int output_len = (int) ceil((double) input_len * (double) output_rate /
                                      (double) input_rate);
    output.resize(output_len);

    constexpr double pi = 3.14159265358979323846264338327950288;
    constexpr int zero_crossings = 24;
    const double cutoff = std::min(1.0, (double) output_rate / (double) input_rate);
    const double radius = (double) zero_crossings / cutoff;

    auto sinc = [](double x) -> double {
        if (fabs(x) < 1.0e-8) {
            return 1.0;
        }
        return sin(x) / x;
    };

    for (int i = 0; i < output_len; ++i) {
        const double center = (double) i * src_per_dst;
        const int left = (int) ceil(center - radius);
        const int right = (int) floor(center + radius);

        double sum = 0.0;
        double weight_sum = 0.0;
        for (int j = left; j <= right; ++j) {
            if (j < 0 || j >= input_len) {
                continue;
            }
            const double distance = center - (double) j;
            const double window_pos = fabs(distance) / radius;
            if (window_pos > 1.0) {
                continue;
            }

            const double window = 0.5 + 0.5 * cos(pi * window_pos);
            const double weight = cutoff * sinc(pi * cutoff * distance) * window;
            sum += (double) input[j] * weight;
            weight_sum += weight;
        }

        if (fabs(weight_sum) > 1.0e-12) {
            output[i] = (float) (sum / weight_sum);
        } else {
            const int nearest = std::max(0, std::min(input_len - 1, (int) llround(center)));
            output[i] = input[nearest];
        }
    }
}

} // namespace pipeline_internal
} // namespace qwen3_tts
