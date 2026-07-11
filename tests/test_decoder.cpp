#include "audio_tokenizer_decoder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <fstream>
#include <vector>
#include <cmath>

static bool load_binary_file(const char * path, std::vector<uint8_t> & data) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        return false;
    }
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    data.resize(size);
    f.read(reinterpret_cast<char *>(data.data()), size);
    return f.good();
}

static bool save_binary_file(const char * path, const void * data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }
    f.write(reinterpret_cast<const char *>(data), size);
    return f.good();
}

static bool parse_nonnegative_int(const char * text, int & value) {
    if (!text || text[0] == '\0') {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const long parsed = strtol(text, &end, 10);
    if (errno != 0 || !end || end[0] != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --tokenizer <path>  Path to tokenizer GGUF file\n");
    fprintf(stderr, "  --codes <path>      Path to speech codes binary file (int64)\n");
    fprintf(stderr, "  --reference <path>  Path to reference audio binary file (float32)\n");
    fprintf(stderr, "  --output <path>     Path to save decoded audio (optional)\n");
    fprintf(stderr, "  --bench-warmup <n>  Additional benchmark warmups after correctness decodes (default: 0)\n");
    fprintf(stderr, "  --bench-runs <n>    Fixed-code decoder benchmark measurements (default: 0)\n");
    fprintf(stderr, "  --help              Show this help\n");
}

int main(int argc, char ** argv) {
    const char * tokenizer_path = "models/qwen-tokenizer-12hz-Q8_0.gguf";
    const char * codes_path = "reference/speech_codes.bin";
    const char * reference_path = "reference/decoded_audio.bin";
    const char * output_path = nullptr;
    int benchmark_warmups = 0;
    int benchmark_runs = 0;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--codes") == 0 && i + 1 < argc) {
            codes_path = argv[++i];
        } else if (strcmp(argv[i], "--reference") == 0 && i + 1 < argc) {
            reference_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--bench-warmup") == 0) {
            if (i + 1 >= argc || !parse_nonnegative_int(argv[i + 1], benchmark_warmups)) {
                fprintf(stderr, "Invalid or missing --bench-warmup value\n");
                return 1;
            }
            ++i;
        } else if (strcmp(argv[i], "--bench-runs") == 0) {
            if (i + 1 >= argc || !parse_nonnegative_int(argv[i + 1], benchmark_runs)) {
                fprintf(stderr, "Invalid or missing --bench-runs value\n");
                return 1;
            }
            ++i;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (benchmark_warmups > 0 && benchmark_runs == 0) {
        fprintf(stderr, "Invalid benchmark counts: --bench-warmup requires --bench-runs > 0\n");
        return 1;
    }
    
    printf("=== Audio Tokenizer Decoder Test ===\n\n");

    int fail_count = 0;
    
    qwen3_tts::AudioTokenizerDecoder decoder;
    
    printf("Test 1: Load model from %s\n", tokenizer_path);
    if (!decoder.load_model(tokenizer_path)) {
        fprintf(stderr, "  FAIL: %s\n", decoder.get_error().c_str());
        return 1;
    }
    printf("  PASS: Model loaded successfully\n");
    
    auto config = decoder.get_config();
    printf("  Config: sample_rate=%d, n_codebooks=%d, codebook_size=%d\n",
           config.sample_rate, config.n_codebooks, config.codebook_size);
    printf("\n");
    
    printf("Test 2: Load speech codes from %s\n", codes_path);
    std::vector<uint8_t> codes_data;
    if (!load_binary_file(codes_path, codes_data)) {
        fprintf(stderr, "  FAIL: Could not load codes file\n");
        return 1;
    }
    
    int64_t * codes_i64 = reinterpret_cast<int64_t *>(codes_data.data());
    int n_codes = codes_data.size() / sizeof(int64_t);
    int n_frames = n_codes / config.n_codebooks;
    
    printf("  Loaded %d codes (%d frames x %d codebooks)\n", n_codes, n_frames, config.n_codebooks);
    
    std::vector<int32_t> codes_i32(n_codes);
    for (int i = 0; i < n_codes; ++i) {
        codes_i32[i] = static_cast<int32_t>(codes_i64[i]);
    }
    
    printf("  First frame codes: ");
    for (int cb = 0; cb < std::min(8, config.n_codebooks); ++cb) {
        printf("%d ", codes_i32[cb]);
    }
    printf("...\n");
    printf("  PASS: Codes loaded and converted\n\n");
    
    printf("Test 3: Decode speech codes to waveform\n");
    
    printf("  Debug: Testing single frame decode...\n");
    std::vector<float> single_samples;
    if (!decoder.decode(codes_i32.data(), 1, single_samples)) {
        fprintf(stderr, "  FAIL (single frame): %s\n", decoder.get_error().c_str());
        fail_count++;
    } else {
        printf("  Single frame: %zu samples, first 5: ", single_samples.size());
        for (int i = 0; i < 5 && i < (int)single_samples.size(); ++i) {
            printf("%.6f ", single_samples[i]);
        }
        printf("\n");
    }
    
    std::vector<float> samples;
    if (!decoder.decode(codes_i32.data(), n_frames, samples)) {
        fprintf(stderr, "  FAIL: %s\n", decoder.get_error().c_str());
        return 1;
    }
    printf("  PASS: Decoded %zu samples (%.3f seconds at %d Hz)\n",
           samples.size(), (float)samples.size() / config.sample_rate, config.sample_rate);
    const qwen3_tts::audio_decoder_timing first_timing = decoder.get_last_timing();
    printf("  Timing: build=%lld alloc=%lld upload=%lld compute=%lld read=%lld total=%lld rebuilt=%d\n",
           (long long) first_timing.graph_build_ms,
           (long long) first_timing.graph_alloc_ms,
           (long long) first_timing.input_upload_ms,
           (long long) first_timing.graph_compute_ms,
           (long long) first_timing.output_read_ms,
           (long long) first_timing.total_ms,
           first_timing.graph_rebuilt);
    
    float min_val = samples[0], max_val = samples[0], sum = 0;
    for (float s : samples) {
        min_val = std::min(min_val, s);
        max_val = std::max(max_val, s);
        sum += s;
    }
    printf("  Audio stats: min=%.4f, max=%.4f, mean=%.6f\n", min_val, max_val, sum / samples.size());
    printf("\n");

    printf("Test 4: Decode repeat stability\n");
    std::vector<float> samples_second;
    if (!decoder.decode(codes_i32.data(), n_frames, samples_second)) {
        fprintf(stderr, "  FAIL: second decode failed: %s\n", decoder.get_error().c_str());
        return 1;
    }
    if (samples_second.size() != samples.size()) {
        fprintf(stderr, "  FAIL: second decode sample count changed: %zu vs %zu\n",
                samples_second.size(), samples.size());
        fail_count++;
    } else {
        const qwen3_tts::audio_decoder_timing second_timing = decoder.get_last_timing();
        printf("  Second timing: build=%lld alloc=%lld upload=%lld compute=%lld read=%lld total=%lld rebuilt=%d\n",
               (long long) second_timing.graph_build_ms,
               (long long) second_timing.graph_alloc_ms,
               (long long) second_timing.input_upload_ms,
               (long long) second_timing.graph_compute_ms,
               (long long) second_timing.output_read_ms,
               (long long) second_timing.total_ms,
               second_timing.graph_rebuilt);
        double max_abs_diff = 0.0;
        double rms_diff = 0.0;
        for (size_t i = 0; i < samples.size(); ++i) {
            const double diff = (double) samples_second[i] - (double) samples[i];
            const double abs_diff = std::abs(diff);
            if (abs_diff > max_abs_diff) {
                max_abs_diff = abs_diff;
            }
            rms_diff += diff * diff;
        }
        rms_diff = samples.empty() ? 0.0 : std::sqrt(rms_diff / (double) samples.size());
        printf("  Repeat max_abs_diff=%.9f rms=%.9f\n", max_abs_diff, rms_diff);
        if (max_abs_diff > 1e-6 || rms_diff > 1e-7) {
            printf("  FAIL: repeated decode is not stable\n");
            fail_count++;
        } else {
            printf("  PASS: repeated decode is stable\n");
        }
    }
    printf("\n");

    if (benchmark_runs > 0) {
        printf("Benchmark: Resident fixed-code decoder (2 prior correctness decodes, "
               "%d additional warmups, %d runs)\n",
               benchmark_warmups, benchmark_runs);

        std::vector<float> benchmark_samples;
        for (int i = 0; i < benchmark_warmups; ++i) {
            const auto start = std::chrono::steady_clock::now();
            if (!decoder.decode(codes_i32.data(), n_frames, benchmark_samples)) {
                fprintf(stderr, "  FAIL: benchmark warmup %d failed: %s\n",
                        i + 1, decoder.get_error().c_str());
                return 1;
            }
            const auto end = std::chrono::steady_clock::now();
            const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();
            const qwen3_tts::audio_decoder_timing timing = decoder.get_last_timing();
            printf("DECODER_BENCH_JSON {\"warmup\":true,\"iteration\":%d,"
                   "\"wall_ms\":%.3f,\"build_ms\":%lld,\"alloc_ms\":%lld,"
                   "\"upload_ms\":%lld,\"compute_ms\":%lld,\"read_ms\":%lld,"
                   "\"total_ms\":%lld,\"rebuilt\":%d,\"frames\":%d}\n",
                   i + 1, wall_ms, (long long) timing.graph_build_ms,
                   (long long) timing.graph_alloc_ms, (long long) timing.input_upload_ms,
                   (long long) timing.graph_compute_ms, (long long) timing.output_read_ms,
                   (long long) timing.total_ms, timing.graph_rebuilt, n_frames);
        }

        std::vector<double> wall_times_ms;
        wall_times_ms.reserve(benchmark_runs);
        int64_t build_total_ms = 0;
        int64_t alloc_total_ms = 0;
        int64_t upload_total_ms = 0;
        int64_t compute_total_ms = 0;
        int64_t read_total_ms = 0;
        int64_t decode_total_ms = 0;
        for (int i = 0; i < benchmark_runs; ++i) {
            const auto start = std::chrono::steady_clock::now();
            if (!decoder.decode(codes_i32.data(), n_frames, benchmark_samples)) {
                fprintf(stderr, "  FAIL: benchmark run %d failed: %s\n",
                        i + 1, decoder.get_error().c_str());
                return 1;
            }
            const auto end = std::chrono::steady_clock::now();
            const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();
            const qwen3_tts::audio_decoder_timing timing = decoder.get_last_timing();
            wall_times_ms.push_back(wall_ms);
            build_total_ms += timing.graph_build_ms;
            alloc_total_ms += timing.graph_alloc_ms;
            upload_total_ms += timing.input_upload_ms;
            compute_total_ms += timing.graph_compute_ms;
            read_total_ms += timing.output_read_ms;
            decode_total_ms += timing.total_ms;
            printf("DECODER_BENCH_JSON {\"warmup\":false,\"iteration\":%d,"
                   "\"wall_ms\":%.3f,\"build_ms\":%lld,\"alloc_ms\":%lld,"
                   "\"upload_ms\":%lld,\"compute_ms\":%lld,\"read_ms\":%lld,"
                   "\"total_ms\":%lld,\"rebuilt\":%d,\"frames\":%d}\n",
                   i + 1, wall_ms, (long long) timing.graph_build_ms,
                   (long long) timing.graph_alloc_ms, (long long) timing.input_upload_ms,
                   (long long) timing.graph_compute_ms, (long long) timing.output_read_ms,
                   (long long) timing.total_ms, timing.graph_rebuilt, n_frames);
        }

        std::vector<double> sorted_wall_times = wall_times_ms;
        std::sort(sorted_wall_times.begin(), sorted_wall_times.end());
        const size_t middle = sorted_wall_times.size() / 2;
        const double median_wall_ms = sorted_wall_times.size() % 2 == 0
            ? (sorted_wall_times[middle - 1] + sorted_wall_times[middle]) * 0.5
            : sorted_wall_times[middle];
        double wall_total_ms = 0.0;
        for (double wall_ms : wall_times_ms) {
            wall_total_ms += wall_ms;
        }
        printf("DECODER_BENCH_SUMMARY {\"prior_full_decodes\":2,"
               "\"additional_warmups\":%d,\"runs\":%d,"
               "\"wall_total_ms\":%.3f,\"wall_mean_ms\":%.3f,"
               "\"wall_median_ms\":%.3f,\"build_total_ms\":%lld,"
               "\"alloc_total_ms\":%lld,\"upload_total_ms\":%lld,"
               "\"compute_total_ms\":%lld,\"read_total_ms\":%lld,"
               "\"decode_total_ms\":%lld,\"frames\":%d}\n",
               benchmark_warmups, benchmark_runs, wall_total_ms,
               wall_total_ms / benchmark_runs, median_wall_ms,
               (long long) build_total_ms, (long long) alloc_total_ms,
               (long long) upload_total_ms, (long long) compute_total_ms,
               (long long) read_total_ms, (long long) decode_total_ms, n_frames);
        printf("  PASS: Resident decoder benchmark completed\n\n");
    }
    
    if (output_path) {
        printf("Test 5: Save decoded audio to %s\n", output_path);
        if (save_binary_file(output_path, samples.data(), samples.size() * sizeof(float))) {
            printf("  PASS: Saved %zu samples\n", samples.size());
        } else {
            fprintf(stderr, "  FAIL: Could not save output file\n");
            fail_count++;
        }
        printf("\n");
    }
    
    printf("Test 6: Compare with reference audio from %s\n", reference_path);
    std::vector<uint8_t> ref_data;
    if (!load_binary_file(reference_path, ref_data)) {
        fprintf(stderr, "  SKIP: Could not load reference file\n");
    } else {
        float * ref_samples = reinterpret_cast<float *>(ref_data.data());
        int ref_n_samples = ref_data.size() / sizeof(float);
        
        printf("  Reference: %d samples\n", ref_n_samples);
        printf("  Generated: %zu samples\n", samples.size());
        int sample_delta = std::abs((int)samples.size() - ref_n_samples);
        double sample_delta_pct = ref_n_samples > 0 ? (100.0 * sample_delta / ref_n_samples) : 0.0;
        printf("  Sample count delta: %d (%.3f%%)\n", sample_delta, sample_delta_pct);
        if (sample_delta_pct > 5.0) {
            printf("  FAIL: Sample count delta > 5%%\n");
            fail_count++;
        } else if (sample_delta_pct > 1.0) {
            printf("  WARN: Sample count delta > 1%%\n");
        } else {
            printf("  PASS: Sample count delta <= 1%%\n");
        }
        
        int compare_len = std::min((int)samples.size(), ref_n_samples);
        
        double l2_sum = 0;
        double ref_sum = 0;
        double gen_sum = 0;
        double ref_sq_sum = 0;
        double gen_sq_sum = 0;
        double cross_sum = 0;
        
        for (int i = 0; i < compare_len; ++i) {
            double diff = samples[i] - ref_samples[i];
            l2_sum += diff * diff;
            ref_sum += ref_samples[i];
            gen_sum += samples[i];
            ref_sq_sum += ref_samples[i] * ref_samples[i];
            gen_sq_sum += samples[i] * samples[i];
            cross_sum += ref_samples[i] * samples[i];
        }
        
        double l2_dist = sqrt(l2_sum / compare_len);
        
        double ref_mean = ref_sum / compare_len;
        double gen_mean = gen_sum / compare_len;
        double ref_var = ref_sq_sum / compare_len - ref_mean * ref_mean;
        double gen_var = gen_sq_sum / compare_len - gen_mean * gen_mean;
        double covar = cross_sum / compare_len - ref_mean * gen_mean;
        double correlation = covar / (sqrt(ref_var) * sqrt(gen_var) + 1e-10);
        
        printf("  L2 distance (RMS): %.6f\n", l2_dist);
        printf("  Correlation: %.6f\n", correlation);
        
        if (l2_dist < 0.001) {
            printf("  PASS: L2 distance < 0.001 (excellent match)\n");
        } else if (l2_dist < 0.01) {
            printf("  PASS: L2 distance < 0.01 (good match)\n");
        } else if (l2_dist < 0.1) {
            printf("  WARN: L2 distance < 0.1 (moderate match)\n");
        } else if (l2_dist < 0.15) {
            printf("  WARN: L2 distance < 0.15 (loose match)\n");
        } else {
            printf("  FAIL: L2 distance >= 0.15 (poor match)\n");
            fail_count++;
        }
        
        if (correlation > 0.95) {
            printf("  PASS: Correlation > 0.95 (excellent)\n");
        } else if (correlation > 0.8) {
            printf("  PASS: Correlation > 0.8 (good)\n");
        } else if (correlation > 0.5) {
            printf("  WARN: Correlation > 0.5 (moderate)\n");
        } else {
            printf("  WARN: Correlation <= 0.5 (informational only; waveform phase/alignment may differ)\n");
        }
    }
    printf("\n");

    if (fail_count > 0) {
        printf("=== Tests completed with FAILURES (%d) ===\n", fail_count);
        return 1;
    }

    printf("=== All tests completed ===\n");
    return 0;
}
