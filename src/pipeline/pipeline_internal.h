#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace qwen3_tts {
class Qwen3TTS;
struct tts_params;
struct tts_result;
struct tts_streaming_params;
struct tts_audio_chunk;
using tts_audio_chunk_callback_t = std::function<bool(const tts_audio_chunk &)>;
namespace pipeline_internal {

struct ops {
    static tts_result synthesize_internal(Qwen3TTS & self,
                                          const std::string & text,
                                          const float * speaker_embedding,
                                          const tts_params & params,
                                          tts_result & result,
                                          const tts_streaming_params * streaming_params = nullptr,
                                          const tts_audio_chunk_callback_t * on_audio_chunk = nullptr);
};

struct process_memory_snapshot {
    uint64_t rss_bytes = 0;
    uint64_t phys_footprint_bytes = 0;
};

void configure_ggml_logging_once();
bool env_flag_enabled(const char * name);
int64_t get_time_ms();
bool get_process_memory_snapshot(process_memory_snapshot & out);
std::string format_bytes(uint64_t bytes);
void log_memory_usage(const char * label);
void resample_linear(const float * input, int input_len, int input_rate,
                     std::vector<float> & output, int output_rate);

} // namespace pipeline_internal
} // namespace qwen3_tts
