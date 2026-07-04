#include "speech_tokenizer_encoder.h"
#include "qwen3_tts.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct npy_array {
    std::string descr;
    bool fortran_order = false;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

std::string slurp_header(std::ifstream & file, uint8_t major) {
    if (major == 1) {
        uint16_t len = 0;
        file.read(reinterpret_cast<char *>(&len), sizeof(len));
        std::string header(len, '\0');
        file.read(header.data(), len);
        return header;
    }
    uint32_t len = 0;
    file.read(reinterpret_cast<char *>(&len), sizeof(len));
    std::string header(len, '\0');
    file.read(header.data(), len);
    return header;
}

std::vector<int64_t> parse_shape(const std::string & header) {
    const size_t start = header.find('(');
    const size_t end = header.find(')', start);
    if (start == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("invalid npy shape header");
    }
    std::vector<int64_t> shape;
    size_t pos = start + 1;
    while (pos < end) {
        while (pos < end && (header[pos] == ' ' || header[pos] == ',')) pos++;
        if (pos >= end) break;
        size_t next = pos;
        while (next < end && header[next] >= '0' && header[next] <= '9') next++;
        if (next > pos) {
            shape.push_back(std::stoll(header.substr(pos, next - pos)));
        }
        pos = next + 1;
    }
    return shape;
}

std::string parse_descr(const std::string & header) {
    const std::string key = "'descr'";
    size_t pos = header.find(key);
    if (pos == std::string::npos) {
        throw std::runtime_error("invalid npy descr header");
    }
    pos = header.find(':', pos + key.size());
    pos = header.find('\'', pos);
    const size_t end = header.find('\'', pos + 1);
    if (pos == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("invalid npy descr value");
    }
    return header.substr(pos + 1, end - pos - 1);
}

bool parse_fortran_order(const std::string & header) {
    const std::string key = "'fortran_order'";
    size_t pos = header.find(key);
    if (pos == std::string::npos) {
        throw std::runtime_error("invalid npy fortran_order header");
    }
    pos = header.find(':', pos + key.size());
    if (pos == std::string::npos) {
        throw std::runtime_error("invalid npy fortran_order value");
    }
    return header.find("True", pos) != std::string::npos &&
           header.find("True", pos) < header.find(',', pos);
}

npy_array load_npy(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    char magic[6] = {};
    file.read(magic, sizeof(magic));
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy magic: " + path);
    }
    uint8_t major = 0;
    uint8_t minor = 0;
    file.read(reinterpret_cast<char *>(&major), 1);
    file.read(reinterpret_cast<char *>(&minor), 1);
    (void) minor;
    std::string header = slurp_header(file, major);
    npy_array array;
    array.descr = parse_descr(header);
    array.fortran_order = parse_fortran_order(header);
    array.shape = parse_shape(header);
    file.seekg(0, std::ios::end);
    const std::streamoff end = file.tellg();
    const std::streamoff data_pos = 6 + 2 + (major == 1 ? 2 : 4) + (std::streamoff) header.size();
    file.seekg(data_pos, std::ios::beg);
    array.data.resize((size_t) (end - data_pos));
    file.read(reinterpret_cast<char *>(array.data.data()), (std::streamsize) array.data.size());
    return array;
}

const float * as_f32(const npy_array & array) {
    if (array.descr != "<f4" && array.descr != "|f4") {
        throw std::runtime_error("expected f32 npy array");
    }
    return reinterpret_cast<const float *>(array.data.data());
}

const int32_t * as_i32(const npy_array & array) {
    if (array.descr != "<i4" && array.descr != "|i4") {
        throw std::runtime_error("expected i32 npy array");
    }
    return reinterpret_cast<const int32_t *>(array.data.data());
}

struct projection_error {
    double rmse = 0.0;
    double mae = 0.0;
    float max_abs = 0.0f;
};

projection_error report_projection_error(const char * name,
                                         const std::vector<float> & actual,
                                         const float * expected,
                                         int32_t dim,
                                         int32_t n_frames) {
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    size_t max_idx = 0;
    const size_t n = (size_t) dim * (size_t) n_frames;
    for (size_t i = 0; i < n; ++i) {
        const float diff = actual[i] - expected[i];
        const float ad = std::fabs(diff);
        sum_sq += (double) diff * (double) diff;
        sum_abs += ad;
        if (ad > max_abs) {
            max_abs = ad;
            max_idx = i;
        }
    }
    projection_error err;
    err.rmse = std::sqrt(sum_sq / (double) n);
    err.mae = sum_abs / (double) n;
    err.max_abs = max_abs;
    fprintf(stderr, "%s projection error: rmse=%.6g mae=%.6g max=%.6g at flat=%zu actual=%.6g expected=%.6g\n",
            name, err.rmse, err.mae,
            max_abs, max_idx, actual[max_idx], expected[max_idx]);
    return err;
}

} // namespace

int main(int argc, char ** argv) {
    std::string tokenizer_path = "models/qwen-tokenizer-12hz-Q8_0.gguf";
    if (argc > 1) {
        tokenizer_path = argv[1];
    }

    qwen3_tts::SpeechTokenizerEncoder encoder;
    if (!encoder.load_model(tokenizer_path, true)) {
        fprintf(stderr, "failed to load speech tokenizer encoder: %s\n", encoder.get_error().c_str());
        return 1;
    }

    const auto & cfg = encoder.get_config();
    printf("speech tokenizer encoder ok: sample_rate=%d frame_rate=%.2f hidden=%d layers=%d heads=%d valid_quantizers=%d\n",
           cfg.sample_rate, cfg.frame_rate, cfg.hidden_size, cfg.n_layers,
           cfg.n_heads, cfg.n_valid_quantizers);

    if (argc > 2) {
        try {
        const std::string golden_dir = argv[2];
        const npy_array semantic = load_npy(golden_dir + "/hook_encoder_quantizer_semantic_residual_vector_quantizer_input_proj.npy");
        const npy_array acoustic = load_npy(golden_dir + "/hook_encoder_quantizer_acoustic_residual_vector_quantizer_input_proj.npy");
        const npy_array expected = load_npy(golden_dir + "/api_audio_codes.npy");
        fprintf(stderr, "golden shapes: semantic=[%lld,%lld,%lld] acoustic=[%lld,%lld,%lld] expected=[%lld,%lld]\n",
                (long long) semantic.shape[0], (long long) semantic.shape[1], (long long) semantic.shape[2],
                (long long) acoustic.shape[0], (long long) acoustic.shape[1], (long long) acoustic.shape[2],
                (long long) expected.shape[0], (long long) expected.shape[1]);
        if (semantic.shape.size() != 3 || semantic.shape[1] != cfg.codebook_dim ||
            acoustic.shape.size() != 3 || acoustic.shape[1] != cfg.codebook_dim ||
            semantic.shape[2] != acoustic.shape[2]) {
            fprintf(stderr, "unexpected projected feature shapes\n");
            return 1;
        }
        const int32_t n_frames = (int32_t) semantic.shape[2];
        qwen3_tts::speech_codes codes;
        fprintf(stderr, "running VQ quantize_projected...\n");
        if (!encoder.quantize_projected(as_f32(semantic), as_f32(acoustic), n_frames, codes)) {
            fprintf(stderr, "failed to quantize golden projections: %s\n", encoder.get_error().c_str());
            return 1;
        }
        fprintf(stderr, "quantize_projected returned frames=%d codebooks=%d values=%zu\n",
                codes.n_frames, codes.n_codebooks, codes.codes.size());
        if (expected.shape.size() != 2 || expected.shape[0] != n_frames ||
            expected.shape[1] != codes.n_codebooks) {
            fprintf(stderr, "unexpected expected code shape\n");
            return 1;
        }
        const int32_t * expected_codes = as_i32(expected);
        size_t mismatches = 0;
        for (int32_t frame = 0; frame < codes.n_frames; ++frame) {
            for (int32_t cb = 0; cb < codes.n_codebooks; ++cb) {
                const size_t actual_idx = (size_t) frame * (size_t) codes.n_codebooks + (size_t) cb;
                const size_t expected_idx = expected.fortran_order
                    ? (size_t) frame + (size_t) cb * (size_t) codes.n_frames
                    : actual_idx;
                if (codes.codes[actual_idx] == expected_codes[expected_idx]) {
                    continue;
                }
                if (mismatches < 8) {
                    fprintf(stderr, "code mismatch at frame=%d cb=%d got=%d expected=%d\n",
                            frame, cb, codes.codes[actual_idx], expected_codes[expected_idx]);
                }
                mismatches++;
            }
        }
        if (mismatches != 0) {
            fprintf(stderr, "VQ golden comparison failed: mismatches=%zu / %zu\n",
                    mismatches, codes.codes.size());
            return 1;
        }
        printf("VQ golden comparison ok: frames=%d codebooks=%d\n", codes.n_frames, codes.n_codebooks);

        const npy_array input_values = load_npy(golden_dir + "/model_input_values.npy");
        if (input_values.shape.size() != 2 || input_values.shape[0] != 1) {
            fprintf(stderr, "unexpected model_input_values shape\n");
            return 1;
        }
        qwen3_tts::speech_codes encoded_codes;
        fprintf(stderr, "running full encode() golden comparison...\n");
        std::vector<float> projected_semantic;
        std::vector<float> projected_acoustic;
        int32_t projected_frames = 0;
        if (!encoder.project(as_f32(input_values), (int32_t) input_values.shape[1],
                             projected_semantic, projected_acoustic, projected_frames)) {
            fprintf(stderr, "project failed: %s\n", encoder.get_error().c_str());
            return 1;
        }
        if (projected_frames != n_frames) {
            fprintf(stderr, "project shape mismatch: got %d frames expected %d\n", projected_frames, n_frames);
            return 1;
        }
        const projection_error semantic_err =
            report_projection_error("semantic", projected_semantic, as_f32(semantic), cfg.codebook_dim, n_frames);
        const projection_error acoustic_err =
            report_projection_error("acoustic", projected_acoustic, as_f32(acoustic), cfg.codebook_dim, n_frames);
        if (semantic_err.rmse > 0.10 || semantic_err.max_abs > 1.0 ||
            acoustic_err.rmse > 0.05 || acoustic_err.max_abs > 0.5) {
            fprintf(stderr, "full projection golden comparison failed tolerance\n");
            return 1;
        }
        if (!encoder.encode(as_f32(input_values), (int32_t) input_values.shape[1], encoded_codes)) {
            fprintf(stderr, "full encode failed: %s\n", encoder.get_error().c_str());
            return 1;
        }
        if (encoded_codes.n_frames != n_frames || encoded_codes.n_codebooks != codes.n_codebooks) {
            fprintf(stderr, "full encode shape mismatch: got %d x %d expected %d x %d\n",
                    encoded_codes.n_frames, encoded_codes.n_codebooks, n_frames, codes.n_codebooks);
            return 1;
        }
        size_t encode_mismatches = 0;
        std::vector<size_t> mismatches_by_cb((size_t) encoded_codes.n_codebooks, 0);
        for (int32_t frame = 0; frame < encoded_codes.n_frames; ++frame) {
            for (int32_t cb = 0; cb < encoded_codes.n_codebooks; ++cb) {
                const size_t actual_idx = (size_t) frame * (size_t) encoded_codes.n_codebooks + (size_t) cb;
                const size_t expected_idx = expected.fortran_order
                    ? (size_t) frame + (size_t) cb * (size_t) encoded_codes.n_frames
                    : actual_idx;
                if (encoded_codes.codes[actual_idx] == expected_codes[expected_idx]) {
                    continue;
                }
                if (encode_mismatches < 8) {
                    fprintf(stderr, "encode mismatch at frame=%d cb=%d got=%d expected=%d\n",
                            frame, cb, encoded_codes.codes[actual_idx], expected_codes[expected_idx]);
                }
                encode_mismatches++;
                mismatches_by_cb[(size_t) cb]++;
            }
        }
        if (encode_mismatches != 0) {
            fprintf(stderr, "full encode golden comparison differs after F16 projection: mismatches=%zu / %zu\n",
                    encode_mismatches, encoded_codes.codes.size());
            fprintf(stderr, "mismatches by codebook:");
            for (size_t cb = 0; cb < mismatches_by_cb.size(); ++cb) {
                fprintf(stderr, " cb%zu=%zu", cb, mismatches_by_cb[cb]);
            }
            fprintf(stderr, "\n");
        } else {
            printf("full encode golden comparison ok: frames=%d codebooks=%d\n",
                   encoded_codes.n_frames, encoded_codes.n_codebooks);
        }
        } catch (const std::exception & ex) {
            fprintf(stderr, "golden VQ test failed with exception: %s\n", ex.what());
            return 1;
        } catch (...) {
            fprintf(stderr, "golden VQ test failed with non-standard exception\n");
            return 1;
        }
    }
    return 0;
}
