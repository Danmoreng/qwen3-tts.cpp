#pragma once

#include "audio_tokenizer_decoder.h"
#include "decoder_internal.h"

namespace qwen3_tts {

struct audio_decoder_private {
    audio_decoder_model model;
    audio_decoder_state state;
    std::string error_msg;
    std::vector<int32_t> codes_buf;
    std::vector<std::vector<int32_t>> codebook_input_bufs;
    std::vector<int32_t> positions_buf;
    std::vector<float> mask_buf;
    std::vector<int32_t> stream_codes_buf;
    std::vector<int32_t> stream_positions_buf;
    std::vector<int64_t> stream_rows_buf;
    std::vector<float> stream_mask_buf;
    audio_decoder_timing last_timing;
};

} // namespace qwen3_tts
