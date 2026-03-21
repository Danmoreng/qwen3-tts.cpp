#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace qwen3_tts {
namespace transformer_internal {

namespace {

int32_t argmax_sampling(const float * data, int32_t n) {
    int32_t max_idx = 0;
    float max_val = data[0];
    for (int32_t i = 1; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

} // namespace

int32_t sample_token_inplace(float * logits,
                             int32_t vocab_size,
                             float temperature,
                             int32_t top_k,
                             float top_p,
                             std::vector<float> & probs,
                             std::vector<int32_t> & sorted_indices,
                             std::mt19937 & rng) {
    if (!logits || vocab_size <= 0) {
        return 0;
    }

    if (temperature <= 0.0f) {
        return argmax_sampling(logits, vocab_size);
    }

    for (int32_t i = 0; i < vocab_size; ++i) {
        logits[i] /= temperature;
    }

    if (top_k > 0 && top_k < vocab_size) {
        sorted_indices.resize((size_t) vocab_size);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::partial_sort(sorted_indices.begin(), sorted_indices.begin() + top_k, sorted_indices.end(),
                          [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        const float threshold = logits[sorted_indices[(size_t) top_k - 1]];
        for (int32_t i = 0; i < vocab_size; ++i) {
            if (logits[i] < threshold) {
                logits[i] = -INFINITY;
            }
        }
    }

    const float max_logit = *std::max_element(logits, logits + vocab_size);
    probs.resize((size_t) vocab_size);
    double sum = 0.0;
    for (int32_t i = 0; i < vocab_size; ++i) {
        probs[(size_t) i] = expf(logits[i] - max_logit);
        sum += probs[(size_t) i];
    }

    if (!(sum > 0.0) || !std::isfinite(sum)) {
        return argmax_sampling(logits, vocab_size);
    }

    for (int32_t i = 0; i < vocab_size; ++i) {
        probs[(size_t) i] = (float) (probs[(size_t) i] / sum);
    }

    if (std::isfinite(top_p) && top_p < 1.0f) {
        const float clamped_top_p = std::max(0.0f, top_p);
        sorted_indices.resize((size_t) vocab_size);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [&](int32_t a, int32_t b) { return probs[(size_t) a] > probs[(size_t) b]; });

        double kept_sum = 0.0;
        double cumulative = 0.0;
        int32_t cutoff_rank = vocab_size - 1;
        for (int32_t rank = 0; rank < vocab_size; ++rank) {
            const int32_t idx = sorted_indices[(size_t) rank];
            cumulative += probs[(size_t) idx];
            kept_sum += probs[(size_t) idx];
            if (rank == 0 || cumulative >= clamped_top_p) {
                cutoff_rank = rank;
                break;
            }
        }

        for (int32_t rank = cutoff_rank + 1; rank < vocab_size; ++rank) {
            probs[(size_t) sorted_indices[(size_t) rank]] = 0.0f;
        }

        if (kept_sum > 0.0) {
            const float inv_kept_sum = (float) (1.0 / kept_sum);
            for (int32_t rank = 0; rank <= cutoff_rank; ++rank) {
                const int32_t idx = sorted_indices[(size_t) rank];
                probs[(size_t) idx] *= inv_kept_sum;
            }
        }
    }

    std::discrete_distribution<int32_t> dist(probs.begin(), probs.begin() + vocab_size);
    return dist(rng);
}

} // namespace transformer_internal
} // namespace qwen3_tts
