#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

namespace qwen3_tts {
namespace {

void reset_scheduler_reserve_state(tts_transformer_state & state) {
    state.sched_reserved = false;
    state.sched_reserve_failed = false;
    state.sched_reserved_ctx = 0;
    state.sched_reserved_prefill_len = 0;
    state.code_pred_sched_reserved = false;
    state.code_pred_sched_reserve_failed = false;
    state.talker_replay_ready = false;
    state.talker_replay_failed = false;
    state.talker_replay_n_kv_pad = 0;
    state.talker_replay_graph = nullptr;
    if (state.talker_replay_sched) {
        ggml_backend_sched_reset(state.talker_replay_sched);
    }
    state.code_pred_replay_ready = false;
    state.code_pred_replay_failed = false;
    state.code_pred_replay_graphs.clear();
    for (ggml_backend_sched_t replay_sched : state.code_pred_replay_scheds) {
        if (replay_sched) {
            ggml_backend_sched_reset(replay_sched);
        }
    }
}

bool env_flag_enabled(const char * name) {
#if defined(_MSC_VER)
    char * value = nullptr;
    size_t value_len = 0;
    if (_dupenv_s(&value, &value_len, name) != 0 || !value) {
        return false;
    }
    const bool enabled = value[0] != '\0' && value[0] != '0';
    std::free(value);
    return enabled;
#else
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && value[0] != '0';
#endif
}

void clear_code_pred_static_tensors(tts_transformer_state & state) {
    state.code_pred_prefill_pos = nullptr;
    state.code_pred_prefill_mask_tensor = nullptr;
    state.code_pred_step_pos.clear();
    state.code_pred_step_mask_tensors.clear();
}

} // namespace

bool TTSTransformer::init_kv_cache(int32_t n_ctx) {
    const auto & cfg = impl_->model.config;
    const bool use_f32_cache = env_flag_enabled("QWEN3_TTS_TALKER_KV_F32");
    const ggml_type cache_type = use_f32_cache ? GGML_TYPE_F32 : GGML_TYPE_F16;

    free_tts_kv_cache(impl_->state.cache);

    impl_->state.cache.n_ctx = n_ctx;
    impl_->state.cache.n_used = 0;
    impl_->state.cache.head_dim = cfg.head_dim;
    impl_->state.cache.n_kv_heads = cfg.n_key_value_heads;
    impl_->state.cache.n_layers = cfg.n_layers;
    reset_scheduler_reserve_state(impl_->state);

    const size_t n_tensors = cfg.n_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    impl_->state.cache.ctx = ggml_init(params);
    if (!impl_->state.cache.ctx) {
        error_msg_ = "Failed to create KV cache context";
        return false;
    }

    impl_->state.cache.k_cache.resize(cfg.n_layers);
    impl_->state.cache.v_cache.resize(cfg.n_layers);

    if (use_f32_cache) {
        fprintf(stderr, "  Talker KV cache: F32 (QWEN3_TTS_TALKER_KV_F32 enabled)\n");
    } else {
        fprintf(stderr, "  Talker KV cache: F16 (default)\n");
    }

    for (int il = 0; il < cfg.n_layers; ++il) {
        impl_->state.cache.k_cache[il] = ggml_new_tensor_3d(
            impl_->state.cache.ctx, cache_type,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(impl_->state.cache.k_cache[il], "k_cache_%d", il);

        impl_->state.cache.v_cache[il] = ggml_new_tensor_3d(
            impl_->state.cache.ctx, cache_type,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(impl_->state.cache.v_cache[il], "v_cache_%d", il);
    }

    impl_->state.cache.buffer = ggml_backend_alloc_ctx_tensors(impl_->state.cache.ctx, impl_->state.backend);
    if (!impl_->state.cache.buffer) {
        error_msg_ = "Failed to allocate KV cache buffer";
        return false;
    }

    return true;
}

void TTSTransformer::clear_kv_cache() {
    impl_->state.cache.n_used = 0;
}

bool TTSTransformer::init_code_pred_kv_cache(int32_t n_ctx) {
    const auto & cfg = impl_->model.config;

    free_tts_kv_cache(impl_->state.code_pred_cache);
    clear_code_pred_static_tensors(impl_->state);

    impl_->state.code_pred_cache.n_ctx = n_ctx;
    impl_->state.code_pred_cache.n_used = 0;
    impl_->state.code_pred_cache.head_dim = cfg.code_pred_head_dim;
    impl_->state.code_pred_cache.n_kv_heads = cfg.code_pred_n_key_value_heads;
    impl_->state.code_pred_cache.n_layers = cfg.code_pred_layers;
    reset_scheduler_reserve_state(impl_->state);
    impl_->state.code_pred_prefill_mask.clear();
    impl_->state.code_pred_step_masks.clear();
    impl_->state.code_pred_static_mask_n_ctx = 0;
    impl_->state.code_pred_graph_stats_logged.assign(15, 0);

    const size_t n_tensors = cfg.code_pred_layers * 2 + 2 + 15 * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    impl_->state.code_pred_cache.ctx = ggml_init(params);
    if (!impl_->state.code_pred_cache.ctx) {
        error_msg_ = "Failed to create code predictor KV cache context";
        return false;
    }

    impl_->state.code_pred_cache.k_cache.resize(cfg.code_pred_layers);
    impl_->state.code_pred_cache.v_cache.resize(cfg.code_pred_layers);
    impl_->state.code_pred_step_pos.assign(15, nullptr);
    impl_->state.code_pred_step_mask_tensors.assign(15, nullptr);

    for (int il = 0; il < cfg.code_pred_layers; ++il) {
        impl_->state.code_pred_cache.k_cache[il] = ggml_new_tensor_3d(
            impl_->state.code_pred_cache.ctx, GGML_TYPE_F16,
            cfg.code_pred_head_dim, n_ctx, cfg.code_pred_n_key_value_heads);
        ggml_format_name(impl_->state.code_pred_cache.k_cache[il], "code_pred_k_cache_%d", il);

        impl_->state.code_pred_cache.v_cache[il] = ggml_new_tensor_3d(
            impl_->state.code_pred_cache.ctx, GGML_TYPE_F16,
            cfg.code_pred_head_dim, n_ctx, cfg.code_pred_n_key_value_heads);
        ggml_format_name(impl_->state.code_pred_cache.v_cache[il], "code_pred_v_cache_%d", il);
    }

    impl_->state.code_pred_prefill_pos = ggml_new_tensor_1d(
        impl_->state.code_pred_cache.ctx, GGML_TYPE_I32, 2);
    ggml_set_name(impl_->state.code_pred_prefill_pos, "code_pred_prefill_pos_const");

    impl_->state.code_pred_prefill_mask_tensor = ggml_new_tensor_2d(
        impl_->state.code_pred_cache.ctx, GGML_TYPE_F16, n_ctx, 2);
    ggml_set_name(impl_->state.code_pred_prefill_mask_tensor, "code_pred_prefill_mask_const");

    for (int step = 1; step < 15; ++step) {
        impl_->state.code_pred_step_pos[step] = ggml_new_tensor_1d(
            impl_->state.code_pred_cache.ctx, GGML_TYPE_I32, 1);
        ggml_format_name(impl_->state.code_pred_step_pos[step], "code_pred_step_%02d_pos_const", step);

        impl_->state.code_pred_step_mask_tensors[step] = ggml_new_tensor_2d(
            impl_->state.code_pred_cache.ctx, GGML_TYPE_F16, n_ctx, 1);
        ggml_format_name(impl_->state.code_pred_step_mask_tensors[step],
                         "code_pred_step_%02d_mask_const", step);
    }

    impl_->state.code_pred_cache.buffer = ggml_backend_alloc_ctx_tensors(impl_->state.code_pred_cache.ctx, impl_->state.backend);
    if (!impl_->state.code_pred_cache.buffer) {
        error_msg_ = "Failed to allocate code predictor KV cache buffer";
        return false;
    }

    const int32_t prefill_pos[2] = {0, 1};
    ggml_backend_tensor_set(impl_->state.code_pred_prefill_pos, prefill_pos, 0, sizeof(prefill_pos));

    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    std::vector<ggml_fp16_t> prefill_mask((size_t) n_ctx * 2, neg_inf);
    prefill_mask[0] = zero;
    prefill_mask[(size_t) n_ctx] = zero;
    prefill_mask[(size_t) n_ctx + 1] = zero;
    ggml_backend_tensor_set(impl_->state.code_pred_prefill_mask_tensor, prefill_mask.data(), 0,
                            prefill_mask.size() * sizeof(ggml_fp16_t));

    std::vector<ggml_fp16_t> step_mask((size_t) n_ctx, neg_inf);
    for (int step = 1; step < 15; ++step) {
        const int32_t pos = step + 1;
        ggml_backend_tensor_set(impl_->state.code_pred_step_pos[step], &pos, 0, sizeof(pos));

        std::fill(step_mask.begin(), step_mask.end(), neg_inf);
        const int32_t max_unmasked = std::min<int32_t>(n_ctx - 1, step + 1);
        for (int32_t i = 0; i <= max_unmasked; ++i) {
            step_mask[(size_t) i] = zero;
        }
        ggml_backend_tensor_set(impl_->state.code_pred_step_mask_tensors[step], step_mask.data(), 0,
                                step_mask.size() * sizeof(ggml_fp16_t));
    }

    return true;
}

void TTSTransformer::clear_code_pred_kv_cache() {
    impl_->state.code_pred_cache.n_used = 0;
    static thread_local std::vector<uint8_t> zero_buf;

    size_t max_bytes = 0;
    for (int il = 0; il < impl_->state.code_pred_cache.n_layers; ++il) {
        if (impl_->state.code_pred_cache.k_cache[(size_t) il]) {
            max_bytes = std::max(max_bytes, ggml_nbytes(impl_->state.code_pred_cache.k_cache[(size_t) il]));
        }
        if (impl_->state.code_pred_cache.v_cache[(size_t) il]) {
            max_bytes = std::max(max_bytes, ggml_nbytes(impl_->state.code_pred_cache.v_cache[(size_t) il]));
        }
    }
    if (max_bytes == 0) {
        return;
    }
    zero_buf.assign(max_bytes, 0);

    for (int il = 0; il < impl_->state.code_pred_cache.n_layers; ++il) {
        struct ggml_tensor * k = impl_->state.code_pred_cache.k_cache[(size_t) il];
        struct ggml_tensor * v = impl_->state.code_pred_cache.v_cache[(size_t) il];
        if (k) {
            ggml_backend_tensor_set(k, zero_buf.data(), 0, ggml_nbytes(k));
        }
        if (v) {
            ggml_backend_tensor_set(v, zero_buf.data(), 0, ggml_nbytes(v));
        }
    }
}

void transformer_internal::ops::maybe_reserve_scheduler_graphs(TTSTransformer & self, int32_t prefill_len, int32_t required_ctx) {
    auto & impl = self.impl_;

    if (!impl->state.sched) {
        return;
    }
    if (impl->state.sched_reserve_failed) {
        return;
    }
    if (impl->state.code_pred_cache.n_ctx < 16) {
        return;
    }

    if (impl->state.sched_reserved &&
        impl->state.sched_reserved_ctx >= required_ctx &&
        impl->state.sched_reserved_prefill_len >= prefill_len) {
        return;
    }

    std::string first_failed_graph;
    auto reserve_graph = [&](struct ggml_cgraph * g, const char * name) -> bool {
        if (!g) {
            if (first_failed_graph.empty()) {
                first_failed_graph = name;
            }
            return false;
        }
        const bool ok = ggml_backend_sched_reserve(impl->state.sched, g);
        ggml_backend_sched_reset(impl->state.sched);
        if (!ok && first_failed_graph.empty()) {
            first_failed_graph = name;
        }
        return ok;
    };

    bool ok = true;
    ok &= reserve_graph(build_prefill_forward_graph(self, prefill_len, 0), "talker prefill");
    ok &= reserve_graph(build_step_graph(self, std::max<int32_t>(0, required_ctx - 1), true), "talker step");
    const bool use_dedicated_code_pred_sched =
        impl->state.code_pred_prefill_sched && impl->state.code_pred_step_sched;
    if (!use_dedicated_code_pred_sched) {
        ok &= reserve_graph(build_code_pred_prefill_graph(self, impl->state.hidden_bridge != nullptr), "code predictor prefill");

        for (int step = 1; step < 15; ++step) {
            char name[32];
            snprintf(name, sizeof(name), "code predictor step %d", step);
            ok &= reserve_graph(build_code_pred_step_graph(self, 15, step), name);
        }
    }

    if (ok) {
        impl->state.sched_reserved = true;
        impl->state.sched_reserve_failed = false;
        impl->state.sched_reserved_ctx = required_ctx;
        impl->state.sched_reserved_prefill_len = prefill_len;
    } else {
        impl->state.sched_reserved = false;
        impl->state.sched_reserve_failed = true;
        const char * graph_name = first_failed_graph.empty() ? "unknown graph" : first_failed_graph.c_str();
        fprintf(stderr,
                "  Scheduler reserve failed at %s; disabling reserve warmup and using dynamic graph allocation\n",
                graph_name);
    }
}

} // namespace qwen3_tts
