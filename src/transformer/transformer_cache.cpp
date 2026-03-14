#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace qwen3_tts {
namespace {

void reset_scheduler_reserve_state(tts_transformer_state & state) {
    state.sched_reserved = false;
    state.sched_reserve_failed = false;
    state.sched_reserved_ctx = 0;
    state.sched_reserved_prefill_len = 0;
}

} // namespace

bool TTSTransformer::init_kv_cache(int32_t n_ctx) {
    const auto & cfg = impl_->model.config;

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

    for (int il = 0; il < cfg.n_layers; ++il) {
        impl_->state.cache.k_cache[il] = ggml_new_tensor_3d(
            impl_->state.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(impl_->state.cache.k_cache[il], "k_cache_%d", il);

        impl_->state.cache.v_cache[il] = ggml_new_tensor_3d(
            impl_->state.cache.ctx, GGML_TYPE_F16,
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

    impl_->state.code_pred_cache.n_ctx = n_ctx;
    impl_->state.code_pred_cache.n_used = 0;
    impl_->state.code_pred_cache.head_dim = cfg.code_pred_head_dim;
    impl_->state.code_pred_cache.n_kv_heads = cfg.code_pred_n_key_value_heads;
    impl_->state.code_pred_cache.n_layers = cfg.code_pred_layers;
    reset_scheduler_reserve_state(impl_->state);

    const size_t n_tensors = cfg.code_pred_layers * 2;
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

    for (int il = 0; il < cfg.code_pred_layers; ++il) {
        impl_->state.code_pred_cache.k_cache[il] = ggml_new_tensor_3d(
            impl_->state.code_pred_cache.ctx, GGML_TYPE_F16,
            cfg.code_pred_head_dim, cfg.code_pred_n_key_value_heads, n_ctx);
        ggml_format_name(impl_->state.code_pred_cache.k_cache[il], "code_pred_k_cache_%d", il);

        impl_->state.code_pred_cache.v_cache[il] = ggml_new_tensor_3d(
            impl_->state.code_pred_cache.ctx, GGML_TYPE_F16,
            cfg.code_pred_head_dim, cfg.code_pred_n_key_value_heads, n_ctx);
        ggml_format_name(impl_->state.code_pred_cache.v_cache[il], "code_pred_v_cache_%d", il);
    }

    impl_->state.code_pred_cache.buffer = ggml_backend_alloc_ctx_tensors(impl_->state.code_pred_cache.ctx, impl_->state.backend);
    if (!impl_->state.code_pred_cache.buffer) {
        error_msg_ = "Failed to allocate code predictor KV cache buffer";
        return false;
    }

    return true;
}

void TTSTransformer::clear_code_pred_kv_cache() {
    impl_->state.code_pred_cache.n_used = 0;
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
    ok &= reserve_graph(build_step_graph(self, std::max<int32_t>(0, required_ctx - 1)), "talker step");
    ok &= reserve_graph(build_code_pred_prefill_graph(self), "code predictor prefill");

    for (int step = 1; step < 15; ++step) {
        char name[32];
        snprintf(name, sizeof(name), "code predictor step %d", step);
        ok &= reserve_graph(build_code_pred_step_graph(self, 15, step), name);
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
