#include "audio_tokenizer_encoder.h"
#include "encoder/encoder_state_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

static constexpr float k_pi = 3.14159265358979323846f;

void compute_mel_filterbank_slaney(float * filterbank, int n_mels, int n_fft,
                                   int sample_rate, float f_min, float f_max) {
    auto hz_to_mel_slaney = [](float hz) -> float {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = min_log_hz / f_sp;
        const float logstep = logf(6.4f) / 27.0f;

        if (hz < min_log_hz) {
            return hz / f_sp;
        }
        return min_log_mel + logf(hz / min_log_hz) / logstep;
    };

    auto mel_to_hz_slaney = [](float mel) -> float {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = min_log_hz / f_sp;
        const float logstep = logf(6.4f) / 27.0f;

        if (mel < min_log_mel) {
            return f_sp * mel;
        }
        return min_log_hz * expf(logstep * (mel - min_log_mel));
    };

    const float mel_min = hz_to_mel_slaney(f_min);
    const float mel_max = hz_to_mel_slaney(f_max);
    const int n_fft_bins = n_fft / 2 + 1;

    std::vector<float> mel_points(n_mels + 2);
    std::vector<float> hz_points(n_mels + 2);
    std::vector<float> fft_freqs(n_fft_bins);

    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        hz_points[i] = mel_to_hz_slaney(mel_points[i]);
    }

    for (int i = 0; i < n_fft_bins; ++i) {
        fft_freqs[i] = (float) i * sample_rate / n_fft;
    }

    memset(filterbank, 0, (size_t) n_mels * (size_t) n_fft_bins * sizeof(float));

    for (int m = 0; m < n_mels; ++m) {
        const float f_left = hz_points[m];
        const float f_center = hz_points[m + 1];
        const float f_right = hz_points[m + 2];
        const float enorm = 2.0f / (f_right - f_left);

        for (int k = 0; k < n_fft_bins; ++k) {
            const float freq = fft_freqs[k];
            if (freq >= f_left && freq <= f_center) {
                if (f_center > f_left) {
                    filterbank[m * n_fft_bins + k] = enorm * (freq - f_left) / (f_center - f_left);
                }
            } else if (freq > f_center && freq <= f_right) {
                if (f_right > f_center) {
                    filterbank[m * n_fft_bins + k] = enorm * (f_right - freq) / (f_right - f_center);
                }
            }
        }
    }
}

void compute_dft(const float * input, float * real, float * imag, int n) {
    for (int k = 0; k < n; ++k) {
        real[k] = 0.0f;
        imag[k] = 0.0f;
        for (int t = 0; t < n; ++t) {
            const float angle = -2.0f * k_pi * k * t / n;
            real[k] += input[t] * cosf(angle);
            imag[k] += input[t] * sinf(angle);
        }
    }
}

void compute_centered_window(float * window, int n_fft, int win_length) {
    memset(window, 0, (size_t) n_fft * sizeof(float));

    const int offset = (n_fft - win_length) / 2;
    for (int i = 0; i < win_length; ++i) {
        window[offset + i] = 0.5f * (1.0f - cosf(2.0f * k_pi * i / win_length));
    }
}

} // namespace

namespace qwen3_tts {

void encoder_internal::ops::init_frontend_cache(AudioTokenizerEncoder & self) {
    auto & model = self.impl_->model;
    auto & state = self.impl_->state;

    const int n_fft_bins = model.config.n_fft / 2 + 1;
    state.mel_filterbank.resize((size_t) model.config.n_mels * (size_t) n_fft_bins);
    compute_mel_filterbank_slaney(state.mel_filterbank.data(), model.config.n_mels, model.config.n_fft,
                                  model.config.sample_rate, model.config.f_min, model.config.f_max);

    state.stft_window.resize(model.config.n_fft);
    compute_centered_window(state.stft_window.data(), model.config.n_fft, model.config.win_length);
}

bool encoder_internal::ops::compute_mel_spectrogram(AudioTokenizerEncoder & self,
                                                    const float * samples,
                                                    int32_t n_samples,
                                                    std::vector<float> & mel,
                                                    int32_t & n_frames) {
    const auto & cfg = self.impl_->model.config;
    const auto & state = self.impl_->state;

    const int padding = (cfg.n_fft - cfg.hop_length) / 2;
    const int padded_length = n_samples + 2 * padding;

    std::vector<float> padded(padded_length);
    for (int i = 0; i < padded_length; ++i) {
        int src_idx = 0;
        if (i < padding) {
            src_idx = padding - i;
        } else if (i >= padding + n_samples) {
            src_idx = 2 * n_samples - (i - padding) - 2;
        } else {
            src_idx = i - padding;
        }
        src_idx = std::max(0, std::min(n_samples - 1, src_idx));
        padded[i] = samples[src_idx];
    }

    n_frames = (padded_length - cfg.n_fft) / cfg.hop_length + 1;
    if (n_frames <= 0) {
        self.error_msg_ = "Audio too short for mel spectrogram";
        return false;
    }

    const int n_fft_bins = cfg.n_fft / 2 + 1;
    if (state.mel_filterbank.empty() || state.stft_window.empty()) {
        self.error_msg_ = "Speaker encoder frontend cache is not initialized";
        return false;
    }

    mel.resize((size_t) cfg.n_mels * (size_t) n_frames);

    std::vector<float> frame(cfg.n_fft, 0.0f);
    std::vector<float> fft_real(cfg.n_fft);
    std::vector<float> fft_imag(cfg.n_fft);
    std::vector<float> magnitude(n_fft_bins);

    for (int32_t f = 0; f < n_frames; ++f) {
        const int start = f * cfg.hop_length;

        for (int i = 0; i < cfg.n_fft; ++i) {
            frame[i] = padded[start + i] * state.stft_window[i];
        }

        compute_dft(frame.data(), fft_real.data(), fft_imag.data(), cfg.n_fft);

        for (int k = 0; k < n_fft_bins; ++k) {
            magnitude[k] = sqrtf(fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k] + 1e-9f);
        }

        for (int m = 0; m < cfg.n_mels; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < n_fft_bins; ++k) {
                sum += state.mel_filterbank[m * n_fft_bins + k] * magnitude[k];
            }
            mel[m * n_frames + f] = logf(std::max(sum, 1e-5f));
        }
    }

    return true;
}

} // namespace qwen3_tts
