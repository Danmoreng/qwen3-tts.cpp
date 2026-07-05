#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>

namespace qwen3_tts {

namespace {
struct shared_backend_state {
    ggml_backend_t backend = nullptr;
    int32_t ref_count = 0;
    backend_preference preference = backend_preference::auto_select;
    int32_t cpu_n_threads = 0;
    bool cpu_n_threads_explicit = false;
    std::string active_backend_name;
};

shared_backend_state & get_shared_backend_state() {
    static shared_backend_state state;
    return state;
}

int32_t sanitize_thread_count(int32_t n_threads) {
    return n_threads > 0 ? n_threads : default_cpu_thread_count();
}

void configure_cpu_backend_threads(ggml_backend_t backend) {
    if (!backend) {
        return;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        return;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (!reg) {
        return;
    }

    auto set_fn =
        (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_fn) {
        set_fn(backend, get_cpu_thread_count());
    }
}

ggml_backend_t init_backend_by_type(enum ggml_backend_dev_type type) {
    ggml_backend_t backend = ggml_backend_init_by_type(type, nullptr);
    if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        configure_cpu_backend_threads(backend);
    }
    return backend;
}

std::string backend_name(ggml_backend_t backend) {
    if (!backend) {
        return "";
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    const char * name = device ? ggml_backend_dev_name(device) : nullptr;
    return name ? name : "Unknown";
}
}

GGUFLoader::GGUFLoader() = default;

GGUFLoader::~GGUFLoader() {
    close();
}

ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg) {
    if (error_msg) error_msg->clear();

    auto & shared = get_shared_backend_state();
    if (shared.backend) {
        shared.ref_count++;
        return shared.backend;
    }

    ggml_backend_t backend = nullptr;
    if (shared.preference == backend_preference::cpu) {
        backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    } else if (shared.preference == backend_preference::cuda) {
        backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    } else {
        backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (!backend) {
            backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        }
        if (!backend) {
            backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL);
        }
        if (!backend) {
            backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
    }

    if (!backend && error_msg) {
        const char * name = component_name ? component_name : "component";
        if (shared.preference == backend_preference::cuda) {
            *error_msg = "CUDA backend was requested but could not be initialized for " + std::string(name);
        } else if (shared.preference == backend_preference::cpu) {
            *error_msg = "CPU backend could not be initialized for " + std::string(name);
        } else {
            *error_msg = "Failed to initialize backend (IGPU/GPU/ACCEL/CPU) for " + std::string(name);
        }
    }

    if (backend) {
        shared.backend = backend;
        shared.ref_count = 1;
        shared.active_backend_name = backend_name(backend);
        fprintf(stderr, "  Native backend preference: %d\n", static_cast<int>(shared.preference));
        fprintf(stderr, "  Native active backend: %s\n", shared.active_backend_name.c_str());
        if (ggml_backend_get_device(backend) &&
            ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            fprintf(stderr, "  Native CPU threads: %d\n", get_cpu_thread_count());
        }
    }

    return backend;
}

ggml_backend_t init_cpu_backend(const char * component_name, std::string * error_msg) {
    if (error_msg) error_msg->clear();

    ggml_backend_t backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!backend && error_msg) {
        const char * name = component_name ? component_name : "component";
        *error_msg = "Failed to initialize CPU backend for " + std::string(name);
    }
    return backend;
}

void release_preferred_backend(ggml_backend_t backend) {
    if (!backend) {
        return;
    }

    auto & shared = get_shared_backend_state();
    if (shared.backend == backend) {
        shared.ref_count--;
        if (shared.ref_count <= 0) {
            ggml_backend_free(shared.backend);
            shared.backend = nullptr;
            shared.ref_count = 0;
            shared.active_backend_name.clear();
        }
        return;
    }

    ggml_backend_free(backend);
}

bool set_backend_preference(backend_preference preference) {
    auto & shared = get_shared_backend_state();
    if (shared.backend && shared.preference != preference) {
        return false;
    }
    shared.preference = preference;
    return true;
}

int32_t default_cpu_thread_count() {
    int32_t n_threads = (int32_t) (std::thread::hardware_concurrency() / 2);
    return n_threads > 0 ? n_threads : 1;
}

int32_t default_parallel_thread_count() {
    int32_t n_threads = (int32_t) std::thread::hardware_concurrency();
    return n_threads > 0 ? n_threads : default_cpu_thread_count();
}

bool set_cpu_thread_count(int32_t n_threads, bool explicit_value) {
    auto & shared = get_shared_backend_state();
    shared.cpu_n_threads = sanitize_thread_count(n_threads);
    shared.cpu_n_threads_explicit = explicit_value;
    if (shared.backend) {
        configure_cpu_backend_threads(shared.backend);
    }
    return true;
}

int32_t get_cpu_thread_count() {
    auto & shared = get_shared_backend_state();
    if (shared.cpu_n_threads <= 0) {
        shared.cpu_n_threads = default_cpu_thread_count();
    }
    return shared.cpu_n_threads;
}

bool cpu_thread_count_is_explicit() {
    return get_shared_backend_state().cpu_n_threads_explicit;
}

backend_preference get_backend_preference() {
    return get_shared_backend_state().preference;
}

enum ggml_backend_dev_type get_preferred_backend_type() {
    switch (get_backend_preference()) {
        case backend_preference::cpu:
            return GGML_BACKEND_DEVICE_TYPE_CPU;
        case backend_preference::cuda:
            return GGML_BACKEND_DEVICE_TYPE_GPU;
        case backend_preference::auto_select:
        default:
            return GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
}

int32_t get_compiled_backend_mask() {
    int32_t mask = 1; // CPU
#ifdef QWEN3_TTS_CUDA_ENABLED
    mask |= 2; // CUDA
#endif
    return mask;
}

std::string get_active_backend_name() {
    return get_shared_backend_state().active_backend_name;
}

bool GGUFLoader::open(const std::string & path) {
    close();  // Close any previously opened file
    
    file_path_ = path;
    
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx_,
    };
    
    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) {
        error_msg_ = "Failed to open GGUF file: " + path;
        return false;
    }
    
    return true;
}

void GGUFLoader::close() {
    if (ctx_) {
        gguf_free(ctx_);
        ctx_ = nullptr;
    }
    if (meta_ctx_) {
        ggml_free(meta_ctx_);
        meta_ctx_ = nullptr;
    }
    file_path_.clear();
}

int64_t GGUFLoader::get_n_tensors() const {
    if (!ctx_) return 0;
    return gguf_get_n_tensors(ctx_);
}

const char * GGUFLoader::get_tensor_name(int64_t idx) const {
    if (!ctx_) return nullptr;
    return gguf_get_tensor_name(ctx_, idx);
}

enum ggml_type GGUFLoader::get_tensor_type(int64_t idx) const {
    if (!ctx_) return GGML_TYPE_F32;
    return gguf_get_tensor_type(ctx_, idx);
}

size_t GGUFLoader::get_tensor_offset(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_offset(ctx_, idx);
}

size_t GGUFLoader::get_tensor_size(int64_t idx) const {
    if (!ctx_) return 0;
    return gguf_get_tensor_size(ctx_, idx);
}

int32_t GGUFLoader::get_u32(const char * key, int32_t default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return (int32_t)gguf_get_val_u32(ctx_, idx);
}

float GGUFLoader::get_f32(const char * key, float default_val) const {
    if (!ctx_) return default_val;
    int64_t idx = gguf_find_key(ctx_, key);
    if (idx < 0) return default_val;
    return gguf_get_val_f32(ctx_, idx);
}

size_t GGUFLoader::get_data_offset() const {
    if (!ctx_) return 0;
    return gguf_get_data_offset(ctx_);
}

bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    enum ggml_backend_dev_type preferred_backend_type
) {
    ggml_backend_t backend = init_backend_by_type(preferred_backend_type);
    const backend_preference preference = get_backend_preference();
    const bool allow_fallback = preference == backend_preference::auto_select &&
        preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_CPU;
    if (!backend && allow_fallback) {
        if (preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        }
        if (!backend && preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_GPU) {
            backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        }
        if (!backend && preferred_backend_type != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL);
        }
        if (!backend) {
            backend = init_backend_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
    }
    if (!backend) {
        if (preference == backend_preference::cuda) {
            error_msg = "CUDA backend was requested but could not be initialized for GGUF tensor loader";
        } else if (preference == backend_preference::cpu) {
            error_msg = "CPU backend could not be initialized for GGUF tensor loader";
        } else {
            error_msg = "Failed to initialize backend for GGUF tensor loader";
        }
        return false;
    }
    
    // Allocate buffer for all tensors
    buffer = ggml_backend_alloc_ctx_tensors(model_ctx, backend);
    if (!buffer) {
        error_msg = "Failed to allocate tensor buffer";
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    
    // Open file for reading tensor data
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        error_msg = "Failed to open file for reading: " + path;
        ggml_backend_free(backend);
        return false;
    }
    
    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);
        
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            continue;  // Skip tensors not in our map
        }
        
        struct ggml_tensor * tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);
        
        read_buf.resize(nbytes);
        
#ifdef _WIN32
        if (_fseeki64(f, (int64_t)data_offset + (int64_t)offset, SEEK_SET) != 0) {
#else
        if (fseek(f, data_offset + offset, SEEK_SET) != 0) {
#endif
            error_msg = "Failed to seek to tensor data: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
        
        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            error_msg = "Failed to read tensor data: " + std::string(name);
            fclose(f);
            ggml_backend_free(backend);
            return false;
        }
        
        ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
    }
    
    fclose(f);
    ggml_backend_free(backend);
    
    return true;
}

void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer) {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

} // namespace qwen3_tts
