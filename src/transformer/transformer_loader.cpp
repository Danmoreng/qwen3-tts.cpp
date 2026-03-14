#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "gguf_loader.h"
#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

namespace qwen3_tts {

void TTSTransformer::unload_model() {
    free_tts_kv_cache(impl_->state.cache);
    free_tts_kv_cache(impl_->state.code_pred_cache);
    free_transformer_model(impl_->model);

    impl_->coreml_code_predictor.unload();
    impl_->use_coreml_code_predictor = false;
    impl_->coreml_code_predictor_path.clear();
    impl_->skip_ggml_code_pred_layers = false;

    if (impl_->state.sched) {
        ggml_backend_sched_free(impl_->state.sched);
        impl_->state.sched = nullptr;
    }
    impl_->state.sched_reserved = false;
    impl_->state.sched_reserve_failed = false;
    impl_->state.sched_reserved_ctx = 0;
    impl_->state.sched_reserved_prefill_len = 0;
    if (impl_->state.backend) {
        release_preferred_backend(impl_->state.backend);
        impl_->state.backend = nullptr;
    }
    if (impl_->state.backend_cpu) {
        ggml_backend_free(impl_->state.backend_cpu);
        impl_->state.backend_cpu = nullptr;
    }

    impl_->state.compute_meta.clear();
    impl_->state.code_pred_mask.clear();
    last_hidden_.clear();
    impl_->embd_row_fp16_scratch.clear();
}

bool TTSTransformer::load_model(const std::string & model_path) {
    unload_model();

    impl_->skip_ggml_code_pred_layers = false;
#if defined(__APPLE__)
    const char * use_coreml_env = std::getenv("QWEN3_TTS_USE_COREML");
    bool coreml_disabled = false;
    if (use_coreml_env && use_coreml_env[0] != '\0') {
        std::string use_coreml = use_coreml_env;
        std::transform(use_coreml.begin(), use_coreml.end(), use_coreml.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        coreml_disabled = use_coreml == "0" || use_coreml == "false" ||
                          use_coreml == "off" || use_coreml == "no";
    }

    if (!coreml_disabled) {
        std::string coreml_path;
        const char * override_env = std::getenv("QWEN3_TTS_COREML_MODEL");
        if (override_env && override_env[0] != '\0') {
            coreml_path = override_env;
        } else {
            size_t slash = model_path.find_last_of("/\\");
            const std::string model_dir = (slash == std::string::npos) ? "." : model_path.substr(0, slash);
            coreml_path = model_dir + "/coreml/code_predictor.mlpackage";
        }

        struct stat st = {};
        if (stat(coreml_path.c_str(), &st) == 0) {
            impl_->skip_ggml_code_pred_layers = true;
        } else if (use_coreml_env && use_coreml_env[0] != '\0') {
            impl_->skip_ggml_code_pred_layers = true;
        }
    }
#endif

    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };

    struct gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open GGUF file: " + model_path;
        return false;
    }

    if (!transformer_internal::ops::parse_config(*this, ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    if (!transformer_internal::ops::create_tensors(*this, ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    if (!transformer_internal::ops::load_tensor_data(*this, model_path, ctx)) {
        free_transformer_model(impl_->model);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    if (!impl_->skip_ggml_code_pred_layers) {
        const auto & cfg = impl_->model.config;
        const bool projection_required = cfg.hidden_size > cfg.code_pred_hidden_size;
        const bool likely_legacy_1p7 = (cfg.hidden_size > 1024 &&
                                        impl_->model.code_pred_small_to_mtp_weight == nullptr);
        if ((projection_required || likely_legacy_1p7) &&
            impl_->model.code_pred_small_to_mtp_weight == nullptr) {
            error_msg_ =
                "Model is missing code_pred.small_to_mtp projection weights. "
                "Re-convert with the updated scripts/convert_tts_to_gguf.py.";
            free_transformer_model(impl_->model);
            gguf_free(ctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
    }

    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);

    impl_->state.backend = init_preferred_backend("TTSTransformer", &error_msg_);
    if (!impl_->state.backend) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(impl_->state.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  TTSTransformer backend: %s\n", device_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        impl_->state.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!impl_->state.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for TTSTransformer";
            return false;
        }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(impl_->state.backend);
    if (impl_->state.backend_cpu) {
        backends.push_back(impl_->state.backend_cpu);
    }
    impl_->state.sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), QWEN3_TTS_MAX_NODES, false, true);
    if (!impl_->state.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }

    impl_->state.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_MAX_NODES + ggml_graph_overhead());
    impl_->state.code_pred_compute_meta.resize(15);
    for (int i = 0; i < 15; ++i) {
        impl_->state.code_pred_compute_meta[i].resize(ggml_tensor_overhead() * QWEN3_TTS_MAX_NODES + ggml_graph_overhead());
    }

    if (!transformer_internal::ops::try_init_coreml_code_predictor(*this, model_path)) {
        return false;
    }

    return true;
}

bool transformer_internal::ops::try_init_coreml_code_predictor(TTSTransformer & self, const std::string & model_path) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;
    (void) model_path;
    impl->use_coreml_code_predictor = false;
    impl->coreml_code_predictor_path.clear();

    const char * use_coreml_env = std::getenv("QWEN3_TTS_USE_COREML");
    bool coreml_disabled = false;
    if (use_coreml_env && use_coreml_env[0] != '\0') {
        std::string use_coreml = use_coreml_env;
        std::transform(use_coreml.begin(), use_coreml.end(), use_coreml.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        coreml_disabled = use_coreml == "0" || use_coreml == "false" ||
                          use_coreml == "off" || use_coreml == "no";
    }

    if (coreml_disabled) {
        return true;
    }

#if !defined(__APPLE__)
    if (use_coreml_env && use_coreml_env[0] != '\0') {
        fprintf(stderr, "  CoreML code predictor requested but this build is not on Apple platform\n");
    }
    return true;
#else
    std::string coreml_path;
    const char * override_env = std::getenv("QWEN3_TTS_COREML_MODEL");
    if (override_env && override_env[0] != '\0') {
        coreml_path = override_env;
    } else {
        size_t slash = model_path.find_last_of("/\\");
        const std::string model_dir = (slash == std::string::npos) ? "." : model_path.substr(0, slash);
        coreml_path = model_dir + "/coreml/code_predictor.mlpackage";
    }

    if (!impl->coreml_code_predictor.load(coreml_path, impl->model.config.n_codebooks - 1)) {
        if (impl->skip_ggml_code_pred_layers) {
            error_msg = "CoreML code predictor load failed in strict mode: " + impl->coreml_code_predictor.get_error();
            return false;
        } else {
            fprintf(stderr, "  CoreML code predictor load failed: %s\n",
                    impl->coreml_code_predictor.get_error().c_str());
            fprintf(stderr, "  Falling back to GGML code predictor\n");
            return true;
        }
    }

    impl->use_coreml_code_predictor = true;
    impl->coreml_code_predictor_path = coreml_path;
    fprintf(stderr, "  CoreML code predictor enabled: %s\n", impl->coreml_code_predictor_path.c_str());
    return true;
#endif
}

} // namespace qwen3_tts
