#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qwen3_tts {
class Qwen3TTS;
struct tts_params;
struct tts_result;
struct voice_clone_prompt_asset;
namespace pipeline_internal {

struct ops {
    static tts_result synthesize_internal(Qwen3TTS & self,
                                          const std::string & text,
                                          const float * speaker_embedding,
                                          const voice_clone_prompt_asset * prompt_asset,
                                          const tts_params & params,
                                          tts_result & result);
};

struct process_memory_snapshot {
    uint64_t rss_bytes = 0;
    uint64_t phys_footprint_bytes = 0;
};

void configure_ggml_logging_once();
int64_t get_time_ms();
bool get_process_memory_snapshot(process_memory_snapshot & out);
std::string format_bytes(uint64_t bytes);
void log_memory_usage(const char * label);
void resample_linear(const float * input, int input_len, int input_rate,
                     std::vector<float> & output, int output_rate);

} // namespace pipeline_internal
} // namespace qwen3_tts
