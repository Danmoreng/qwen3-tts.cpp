#pragma once

#include "audio_tokenizer_decoder.h"
#include "decoder_internal.h"

namespace qwen3_tts {

struct audio_decoder_private {
    audio_decoder_model model;
    audio_decoder_state state;
    int32_t n_threads = 0;
    std::string error_msg;
    std::vector<int32_t> codes_buf;
    std::vector<std::vector<int32_t>> codebook_input_bufs;
    std::vector<int32_t> positions_buf;
};

} // namespace qwen3_tts
