#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"

#include <cstdio>
#include <string>

namespace qwen3_tts {

bool transformer_internal::ops::parse_config(TTSTransformer & self, struct gguf_context * ctx) {
    auto & impl = self.impl_;
    auto get_u32_any = [&](std::initializer_list<const char *> keys, int32_t default_val) -> int32_t {
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx >= 0) {
                return (int32_t) gguf_get_val_u32(ctx, idx);
            }
        }
        return default_val;
    };

    auto get_f32_any = [&](std::initializer_list<const char *> keys, float default_val) -> float {
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx >= 0) {
                return gguf_get_val_f32(ctx, idx);
            }
        }
        return default_val;
    };

    auto get_str_any = [&](std::initializer_list<const char *> keys, const char * default_val) -> std::string {
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx >= 0 && gguf_get_kv_type(ctx, idx) == GGUF_TYPE_STRING) {
                const char * s = gguf_get_val_str(ctx, idx);
                if (s && s[0] != '\0') {
                    return std::string(s);
                }
            }
        }
        return std::string(default_val ? default_val : "");
    };

    auto get_bool_any = [&](std::initializer_list<const char *> keys, bool default_val, bool * found) -> bool {
        if (found) {
            *found = false;
        }
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx < 0) {
                continue;
            }

            const enum gguf_type type = gguf_get_kv_type(ctx, idx);
            if (found) {
                *found = true;
            }

            switch (type) {
                case GGUF_TYPE_BOOL:
                    return gguf_get_val_bool(ctx, idx);
                case GGUF_TYPE_UINT8:
                    return gguf_get_val_u8(ctx, idx) != 0;
                case GGUF_TYPE_INT8:
                    return gguf_get_val_i8(ctx, idx) != 0;
                case GGUF_TYPE_UINT16:
                    return gguf_get_val_u16(ctx, idx) != 0;
                case GGUF_TYPE_INT16:
                    return gguf_get_val_i16(ctx, idx) != 0;
                case GGUF_TYPE_UINT32:
                    return gguf_get_val_u32(ctx, idx) != 0;
                case GGUF_TYPE_INT32:
                    return gguf_get_val_i32(ctx, idx) != 0;
                case GGUF_TYPE_UINT64:
                    return gguf_get_val_u64(ctx, idx) != 0;
                case GGUF_TYPE_INT64:
                    return gguf_get_val_i64(ctx, idx) != 0;
                default:
                    fprintf(stderr, "  Warning: ignoring non-numeric metadata key '%s' for boolean parse\n", key);
                    if (found) {
                        *found = false;
                    }
                    break;
            }
        }
        return default_val;
    };

    auto & cfg = impl->model.config;
    auto load_mrope_section = [&](std::initializer_list<const char *> keys, int32_t out_section[3]) -> bool {
        for (const char * key : keys) {
            int64_t idx = gguf_find_key(ctx, key);
            if (idx < 0) {
                continue;
            }
            const int32_t * data = (const int32_t *) gguf_get_arr_data(ctx, idx);
            if (!data) {
                continue;
            }
            for (int i = 0; i < 3; ++i) {
                out_section[i] = data[i];
            }
            return true;
        }
        return false;
    };

    cfg.text_vocab_size = get_u32_any({
        "qwen3-tts.text.vocab_size",
        "qwen3-tts.text_vocab_size",
    }, 151936);
    cfg.text_embd_dim = get_u32_any({
        "qwen3-tts.text.embedding_dim",
        "qwen3-tts.text_hidden_size",
    }, 2048);
    cfg.hidden_size = get_u32_any({
        "qwen3-tts.talker.embedding_length",
        "qwen3-tts.embedding_length",
    }, 1024);
    cfg.n_layers = get_u32_any({
        "qwen3-tts.talker.block_count",
        "qwen3-tts.block_count",
    }, 28);
    cfg.n_attention_heads = get_u32_any({
        "qwen3-tts.talker.attention.head_count",
        "qwen3-tts.attention.head_count",
    }, 16);
    cfg.n_key_value_heads = get_u32_any({
        "qwen3-tts.talker.attention.head_count_kv",
        "qwen3-tts.attention.head_count_kv",
    }, 8);
    cfg.intermediate_size = get_u32_any({
        "qwen3-tts.talker.feed_forward_length",
        "qwen3-tts.feed_forward_length",
    }, 3072);
    cfg.head_dim = get_u32_any({
        "qwen3-tts.talker.attention.key_length",
        "qwen3-tts.attention.key_length",
    }, 128);
    cfg.rms_norm_eps = get_f32_any({
        "qwen3-tts.talker.attention.layer_norm_rms_epsilon",
        "qwen3-tts.attention.layer_norm_rms_epsilon",
    }, 1e-6f);
    cfg.rope_theta = get_f32_any({
        "qwen3-tts.talker.rope.freq_base",
        "qwen3-tts.rope.freq_base",
    }, 1000000.0f);

    cfg.codec_vocab_size = get_u32_any({
        "qwen3-tts.talker.codec_vocab_size",
        "qwen3-tts.vocab_size",
    }, 3072);
    cfg.n_codebooks = get_u32_any({
        "qwen3-tts.talker.num_codebooks",
        "qwen3-tts.num_code_groups",
    }, 16);

    cfg.code_pred_layers = get_u32_any({
        "qwen3-tts.code_pred.layer_count",
        "qwen3-tts.code_predictor.layer_count",
    }, 5);
    cfg.code_pred_vocab_size = get_u32_any({
        "qwen3-tts.code_pred.vocab_size",
        "qwen3-tts.code_predictor.vocab_size",
    }, 2048);
    cfg.code_pred_hidden_size = get_u32_any({
        "qwen3-tts.code_pred.embedding_length",
        "qwen3-tts.code_predictor.embedding_length",
    }, cfg.hidden_size);
    cfg.code_pred_intermediate_size = get_u32_any({
        "qwen3-tts.code_pred.feed_forward_length",
        "qwen3-tts.code_predictor.feed_forward_length",
    }, cfg.intermediate_size);
    cfg.code_pred_n_attention_heads = get_u32_any({
        "qwen3-tts.code_pred.attention.head_count",
        "qwen3-tts.code_predictor.attention.head_count",
    }, cfg.n_attention_heads);
    cfg.code_pred_n_key_value_heads = get_u32_any({
        "qwen3-tts.code_pred.attention.head_count_kv",
        "qwen3-tts.code_predictor.attention.head_count_kv",
    }, cfg.n_key_value_heads);
    cfg.code_pred_head_dim = get_u32_any({
        "qwen3-tts.code_pred.attention.key_length",
        "qwen3-tts.code_predictor.attention.key_length",
    }, cfg.head_dim);
    cfg.code_pred_rms_norm_eps = get_f32_any({
        "qwen3-tts.code_pred.attention.layer_norm_rms_epsilon",
        "qwen3-tts.code_predictor.attention.layer_norm_rms_epsilon",
    }, cfg.rms_norm_eps);
    cfg.code_pred_rope_theta = get_f32_any({
        "qwen3-tts.code_pred.rope.freq_base",
        "qwen3-tts.code_predictor.rope.freq_base",
    }, cfg.rope_theta);

    cfg.codec_pad_id = get_u32_any({
        "qwen3-tts.codec.pad_id",
    }, 2148);
    cfg.codec_bos_id = get_u32_any({
        "qwen3-tts.codec.bos_id",
    }, 2149);
    cfg.codec_eos_id = get_u32_any({
        "qwen3-tts.codec.eos_id",
        "qwen3-tts.codec.eos_token_id",
    }, 2150);

    cfg.tts_bos_token_id = get_u32_any({
        "qwen3-tts.tts_bos_token_id",
        "qwen3-tts.tts.bos_token_id",
        "qwen3-tts.tts.bos_id",
    }, 151672);
    cfg.tts_eos_token_id = get_u32_any({
        "qwen3-tts.tts_eos_token_id",
        "qwen3-tts.tts.eos_token_id",
        "qwen3-tts.tts.eos_id",
    }, 151673);
    cfg.tts_pad_token_id = get_u32_any({
        "qwen3-tts.tts_pad_token_id",
        "qwen3-tts.tts.pad_token_id",
        "qwen3-tts.tts.pad_id",
    }, 151671);

    cfg.codec_think_id = get_u32_any({
        "qwen3-tts.codec.think_id",
        "qwen3-tts.codec_think_id",
    }, 2154);
    cfg.codec_nothink_id = get_u32_any({
        "qwen3-tts.codec.nothink_id",
        "qwen3-tts.codec_nothink_id",
    }, 2155);
    cfg.codec_think_bos_id = get_u32_any({
        "qwen3-tts.codec.think_bos_id",
        "qwen3-tts.codec_think_bos_id",
    }, 2156);
    cfg.codec_think_eos_id = get_u32_any({
        "qwen3-tts.codec.think_eos_id",
        "qwen3-tts.codec_think_eos_id",
    }, 2157);

    cfg.english_language_id = get_u32_any({
        "qwen3-tts.language.english_id",
        "qwen3-tts.codec.language.english_id",
        "qwen3-tts.language_id",
    }, 2050);

    cfg.tts_model_type = transformer_internal::normalize_speaker_name(get_str_any({
        "qwen3-tts.tts_model_type",
    }, "base"));
    cfg.supports_instruction = get_bool_any({
        "qwen3-tts.supports_instruction",
        "qwen3-tts.instruction_supported",
        "qwen3-tts.instruct_supported",
    }, false, &cfg.has_supports_instruction);
    cfg.speaker_id_map.clear();

    fprintf(stderr, "  Codec IDs: pad=%d, bos=%d, eos=%d, think=%d, nothink=%d, think_bos=%d, think_eos=%d\n",
            cfg.codec_pad_id, cfg.codec_bos_id, cfg.codec_eos_id,
            cfg.codec_think_id, cfg.codec_nothink_id, cfg.codec_think_bos_id, cfg.codec_think_eos_id);
    fprintf(stderr, "  TTS model type: %s\n", cfg.tts_model_type.c_str());
    if (cfg.has_supports_instruction) {
        fprintf(stderr, "  Metadata supports_instruction: %s\n", cfg.supports_instruction ? "true" : "false");
    }

    cfg.use_mrope = load_mrope_section({
        "qwen3-tts.talker.rope.mrope_section",
        "qwen3-tts.rope.mrope_section",
    }, cfg.mrope_section);

    for (int i = 0; i < 3; ++i) {
        cfg.code_pred_mrope_section[i] = cfg.mrope_section[i];
    }
    cfg.code_pred_use_mrope = cfg.use_mrope;

    const bool code_pred_has_mrope = load_mrope_section({
        "qwen3-tts.code_predictor.rope.mrope_section",
        "qwen3-tts.code_pred.rope.mrope_section",
        "qwen3-tts.mtp.rope.mrope_section",
    }, cfg.code_pred_mrope_section);
    if (code_pred_has_mrope) {
        cfg.code_pred_use_mrope = true;
    }

    const int32_t code_pred_mrope_override = transformer_internal::parse_env_i32(
        "QWEN3_TTS_CODE_PRED_MROPE", -1, -1, 1);
    if (code_pred_mrope_override >= 0) {
        cfg.code_pred_use_mrope = code_pred_mrope_override != 0;
    }

    fprintf(stderr, "  Talker M-RoPE: %s [%d, %d, %d]\n",
            cfg.use_mrope ? "enabled" : "disabled",
            cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]);
    fprintf(stderr, "  Code predictor M-RoPE: %s [%d, %d, %d]%s\n",
            cfg.code_pred_use_mrope ? "enabled" : "disabled",
            cfg.code_pred_mrope_section[0], cfg.code_pred_mrope_section[1], cfg.code_pred_mrope_section[2],
            code_pred_mrope_override >= 0 ? " (env override)" :
            (code_pred_has_mrope ? " (predictor metadata)" :
             (cfg.use_mrope ? " (inherited from talker)" : "")));

    int64_t spk_count_idx = gguf_find_key(ctx, "qwen3-tts.speaker.count");
    int32_t spk_count = 0;
    if (spk_count_idx >= 0) {
        const enum gguf_type spk_count_type = gguf_get_kv_type(ctx, spk_count_idx);
        if (spk_count_type == GGUF_TYPE_UINT32) {
            spk_count = (int32_t) gguf_get_val_u32(ctx, spk_count_idx);
        } else if (spk_count_type == GGUF_TYPE_INT32) {
            spk_count = gguf_get_val_i32(ctx, spk_count_idx);
        }
    }
    if (spk_count > 0) {
        for (int32_t i = 0; i < spk_count; ++i) {
            char name_key[64];
            char id_key[64];
            snprintf(name_key, sizeof(name_key), "qwen3-tts.speaker.%d.name", i);
            snprintf(id_key, sizeof(id_key), "qwen3-tts.speaker.%d.id", i);
            int64_t name_idx = gguf_find_key(ctx, name_key);
            int64_t id_idx = gguf_find_key(ctx, id_key);
            if (name_idx < 0 || id_idx < 0) {
                continue;
            }
            if (gguf_get_kv_type(ctx, name_idx) != GGUF_TYPE_STRING) {
                continue;
            }
            const char * raw_name = gguf_get_val_str(ctx, name_idx);
            if (!raw_name || raw_name[0] == '\0') {
                continue;
            }

            int32_t spk_id = -1;
            const enum gguf_type id_type = gguf_get_kv_type(ctx, id_idx);
            if (id_type == GGUF_TYPE_UINT32) {
                spk_id = (int32_t) gguf_get_val_u32(ctx, id_idx);
            } else if (id_type == GGUF_TYPE_INT32) {
                spk_id = gguf_get_val_i32(ctx, id_idx);
            } else {
                continue;
            }
            if (spk_id < 0) {
                continue;
            }

            cfg.speaker_id_map[transformer_internal::normalize_speaker_name(raw_name)] = spk_id;
        }
    }

    if (!cfg.speaker_id_map.empty()) {
        fprintf(stderr, "  CustomVoice speakers loaded: %zu\n", cfg.speaker_id_map.size());
    }

    return true;
}

} // namespace qwen3_tts
