#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_cgraph;
struct gguf_context;

namespace qwen3_tts {
class TTSTransformer;
namespace transformer_internal {

struct debug_trace_config {
    bool enabled = false;
    std::string dir;
    int32_t max_frames = 1;
    int32_t max_code_steps = 15;
};

struct ops {
    static bool try_init_coreml_code_predictor(TTSTransformer & self, const std::string & model_path);
    static bool predict_codes_autoregressive_coreml(TTSTransformer & self,
                                                    const float * hidden,
                                                    int32_t codebook_0_token,
                                                    std::vector<int32_t> & output,
                                                    float temperature,
                                                    int32_t top_k,
                                                    int32_t trace_frame);

    static bool build_prefill_graph(TTSTransformer & self,
                                    const int32_t * text_tokens,
                                    int32_t n_tokens,
                                    const float * speaker_embd,
                                    int32_t language_id,
                                    std::vector<float> & prefill_embd,
                                    std::vector<float> & trailing_text_hidden,
                                    std::vector<float> & tts_pad_embed,
                                    const int32_t * instruct_tokens = nullptr,
                                    int32_t n_instruct_tokens = 0);

    static struct ggml_cgraph * build_prefill_forward_graph(TTSTransformer & self, int32_t n_tokens, int32_t n_past);
    static struct ggml_cgraph * build_step_graph(TTSTransformer & self, int32_t n_past);
    static bool project_text_tokens(TTSTransformer & self,
                                    const int32_t * text_tokens,
                                    int32_t n_tokens,
                                    std::vector<float> & output);
    static bool lookup_embedding_rows(TTSTransformer & self,
                                      struct ggml_tensor * embedding,
                                      const int32_t * token_ids,
                                      int32_t n_tokens,
                                      const char * input_name,
                                      const char * output_name,
                                      std::vector<float> & output);
    static bool lookup_single_embedding_row(TTSTransformer & self,
                                            struct ggml_tensor * embedding,
                                            int32_t token_id,
                                            float * out_row);
    static struct ggml_cgraph * build_code_pred_graph(TTSTransformer & self, int32_t n_prev_codes);
    static struct ggml_cgraph * build_code_pred_step_graph(TTSTransformer & self, int32_t n_past, int32_t generation_step);
    static struct ggml_cgraph * build_code_pred_prefill_graph(TTSTransformer & self);
    static void maybe_reserve_scheduler_graphs(TTSTransformer & self, int32_t prefill_len, int32_t required_ctx);
    static bool parse_config(TTSTransformer & self, struct gguf_context * ctx);
    static bool create_tensors(TTSTransformer & self, struct gguf_context * ctx);
    static bool load_tensor_data(TTSTransformer & self, const std::string & path, struct gguf_context * ctx);
};

std::string normalize_speaker_name(const std::string & name);
int32_t parse_env_i32(const char * name, int32_t default_value, int32_t min_value, int32_t max_value);
debug_trace_config init_debug_trace_config();
const debug_trace_config & get_debug_trace_config();
bool debug_trace_should_dump_frame(const debug_trace_config & cfg, int32_t frame);
void debug_trace_write_bin(const debug_trace_config & cfg,
                           const std::string & name,
                           const float * data,
                           size_t count,
                           const char * dtype,
                           const std::vector<int64_t> & shape);
void debug_trace_write_bin(const debug_trace_config & cfg,
                           const std::string & name,
                           const int32_t * data,
                           size_t count,
                           const char * dtype,
                           const std::vector<int64_t> & shape);
void debug_trace_write_text_line(const debug_trace_config & cfg, const std::string & line);

} // namespace transformer_internal
} // namespace qwen3_tts
