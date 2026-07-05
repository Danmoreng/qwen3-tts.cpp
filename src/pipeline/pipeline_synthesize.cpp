#include "qwen3_tts.h"
#include "gguf_loader.h"
#include "pipeline/pipeline_internal.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace qwen3_tts {
using pipeline_internal::format_bytes;
using pipeline_internal::get_process_memory_snapshot;
using pipeline_internal::get_time_ms;
using pipeline_internal::env_flag_enabled;
using pipeline_internal::log_memory_usage;
using pipeline_internal::ops;
using pipeline_internal::process_memory_snapshot;
using pipeline_internal::resample_linear;

namespace {

constexpr int32_t qwen3_tts_codec_hop_length = 1920;
constexpr const char * qwen3_tts_speech_encoder_cuda_env = "QWEN3_TTS_SPEECH_ENCODER_CUDA";

bool speech_encoder_cuda_enabled() {
    const char * value = std::getenv(qwen3_tts_speech_encoder_cuda_env);
    return !value || value[0] == '\0' ||
           env_flag_enabled(qwen3_tts_speech_encoder_cuda_env);
}

bool write_codes_file(const std::string & path,
                      const std::vector<int32_t> & codes,
                      int32_t n_frames,
                      int32_t n_codebooks,
                      std::string & error) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        error = "Failed to open speech-code dump: " + path;
        return false;
    }
    out << "{\n  \"frames\": " << n_frames
        << ",\n  \"codebooks\": " << n_codebooks
        << ",\n  \"codes\": [";
    for (size_t i = 0; i < codes.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        if (n_codebooks > 0 && i % (size_t) n_codebooks == 0) {
            out << "\n    ";
        }
        out << codes[i];
    }
    out << "\n  ]\n}\n";
    if (!out) {
        error = "Failed to write speech-code dump: " + path;
        return false;
    }
    return true;
}

uint64_t fnv1a_append(uint64_t hash, const void * data, size_t bytes) {
    const uint8_t * ptr = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= (uint64_t) ptr[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t hash_reference_samples(const float * samples, int32_t n_samples) {
    uint64_t hash = 1469598103934665603ULL;
    hash = fnv1a_append(hash, &n_samples, sizeof(n_samples));
    if (samples && n_samples > 0) {
        hash = fnv1a_append(hash, samples, (size_t) n_samples * sizeof(float));
    }
    return hash;
}

int32_t duration_sec_to_codec_frames(const audio_decoder_config & cfg,
                                     float duration_sec,
                                     int32_t min_frames) {
    if (duration_sec <= 0.0f) {
        return std::max<int32_t>(0, min_frames);
    }
    int32_t frames = (int32_t) (duration_sec * (float) cfg.sample_rate /
                                (float) qwen3_tts_codec_hop_length + 0.5f);
    return std::max<int32_t>(min_frames, frames);
}

class text_span_estimator {
public:
    void reset(const std::string & text, int32_t sample_rate) {
        text_ = &text;
        sample_rate_ = sample_rate > 0 ? sample_rate : 24000;
        segments_.clear();
        predicted_total_samples_ = 0;
        build_segments(text);
        assign_segment_timeline();
    }

    void span_for_samples(int64_t start_sample,
                          int64_t end_sample,
                          int32_t & start_byte,
                          int32_t & end_byte,
                          int32_t & kind,
                          float & confidence) const {
        start_byte = -1;
        end_byte = -1;
        kind = TTS_TEXT_ALIGNMENT_NONE;
        confidence = 0.0f;
        if (!text_ || text_->empty() || segments_.empty() || predicted_total_samples_ <= 0) {
            return;
        }

        const int64_t midpoint_sample = start_sample + std::max<int64_t>(0, end_sample - start_sample) / 2;
        const size_t segment_index = segment_index_for_sample(midpoint_sample);
        start_byte = (int32_t) segments_[segment_index].start_byte;
        end_byte = (int32_t) segments_[segment_index].end_byte;
        if (end_byte < start_byte) {
            end_byte = start_byte;
        }
        kind = TTS_TEXT_ALIGNMENT_ESTIMATED;
        confidence = 0.55f;
    }

private:
    struct text_segment {
        size_t start_byte = 0;
        size_t end_byte = 0;
        double seconds = 0.0;
        int64_t start_sample = 0;
        int64_t end_sample = 0;
    };

    static size_t utf8_char_len(char c) {
        const unsigned char uc = (unsigned char) c;
        if ((uc & 0x80) == 0) return 1;
        if ((uc & 0xE0) == 0xC0) return 2;
        if ((uc & 0xF0) == 0xE0) return 3;
        if ((uc & 0xF8) == 0xF0) return 4;
        return 1;
    }

    static bool is_ascii_space(char c) {
        const unsigned char uc = (unsigned char) c;
        return uc <= 0x20 || uc == 0x7F;
    }

    static bool is_ascii_strong_boundary(char c) {
        return c == '.' || c == '!' || c == '?' || c == '\n' || c == '\r';
    }

    static bool is_ascii_weak_boundary(char c) {
        return c == ',' || c == ';' || c == ':';
    }

    static bool is_ascii_punctuation(char c) {
        const unsigned char uc = (unsigned char) c;
        return (uc < 128) && (
            (uc >= 33 && uc <= 47) ||
            (uc >= 58 && uc <= 64) ||
            (uc >= 91 && uc <= 96) ||
            (uc >= 123 && uc <= 126));
    }

    void build_segments(const std::string & text) {
        size_t segment_start = std::string::npos;
        int32_t words_in_segment = 0;
        bool in_word = false;

        for (size_t i = 0; i < text.size();) {
            const size_t len = utf8_char_len(text[i]);
            const size_t next = std::min(text.size(), i + len);
            const bool ascii = len == 1;
            const char c = text[i];
            const bool space = ascii && is_ascii_space(c);
            const bool punctuation = ascii && is_ascii_punctuation(c);
            const bool strong_boundary = ascii && is_ascii_strong_boundary(c);
            const bool weak_boundary = ascii && is_ascii_weak_boundary(c);

            if (segment_start == std::string::npos && !space) {
                segment_start = i;
            }

            if (space || punctuation) {
                in_word = false;
            } else if (!in_word) {
                words_in_segment++;
                in_word = true;
            }

            const bool breakable = space || punctuation;
            const bool full_phrase = breakable && words_in_segment >= 5;
            const bool long_phrase = breakable &&
                segment_start != std::string::npos &&
                next - segment_start >= 48;
            const bool boundary =
                segment_start != std::string::npos &&
                (strong_boundary ||
                 (weak_boundary && words_in_segment >= 2) ||
                 full_phrase ||
                 long_phrase);

            if (boundary) {
                const double pause = strong_boundary ? 0.28 : (weak_boundary ? 0.16 : 0.04);
                append_segment(text, segment_start, next, pause);
                segment_start = std::string::npos;
                words_in_segment = 0;
                in_word = false;
            }

            i = next;
        }

        if (segment_start != std::string::npos) {
            append_segment(text, segment_start, text.size(), 0.12);
        }
        if (segments_.empty() && !text.empty()) {
            append_segment(text, 0, text.size(), 0.0);
        }
    }

    void append_segment(const std::string & text, size_t raw_start, size_t raw_end, double pause_sec) {
        size_t start = raw_start;
        size_t end = std::min(raw_end, text.size());
        while (start < end && is_ascii_space(text[start])) {
            start++;
        }
        while (end > start && is_ascii_space(text[end - 1])) {
            end--;
        }
        if (start >= end) {
            return;
        }

        int32_t chars = 0;
        int32_t words = 0;
        bool in_word = false;
        for (size_t i = start; i < end;) {
            const size_t len = utf8_char_len(text[i]);
            const bool ascii = len == 1;
            const char c = text[i];
            const bool separator = ascii && (is_ascii_space(c) || is_ascii_punctuation(c));
            if (!separator && !in_word) {
                words++;
                in_word = true;
            } else if (separator) {
                in_word = false;
            }
            chars++;
            i = std::min(end, i + len);
        }

        text_segment segment;
        segment.start_byte = start;
        segment.end_byte = end;
        segment.seconds = std::max(0.28,
                                   (double) std::max(words, 1) * 0.34 +
                                   (double) chars * 0.018 +
                                   pause_sec);
        segments_.push_back(segment);
    }

    void assign_segment_timeline() {
        double total_seconds = 0.0;
        for (const text_segment & segment : segments_) {
            total_seconds += segment.seconds;
        }
        total_seconds = std::max(0.35, total_seconds);
        predicted_total_samples_ = (int64_t) (total_seconds * (double) sample_rate_ + 0.5);

        double cursor_seconds = 0.0;
        for (size_t i = 0; i < segments_.size(); ++i) {
            text_segment & segment = segments_[i];
            segment.start_sample = (int64_t) (cursor_seconds * (double) sample_rate_ + 0.5);
            cursor_seconds += segment.seconds;
            segment.end_sample = (i + 1 == segments_.size())
                ? predicted_total_samples_
                : (int64_t) (cursor_seconds * (double) sample_rate_ + 0.5);
            if (segment.end_sample <= segment.start_sample) {
                segment.end_sample = segment.start_sample + 1;
            }
        }
    }

    size_t segment_index_for_sample(int64_t sample) const {
        if (segments_.empty()) {
            return 0;
        }
        const int64_t clamped = std::max<int64_t>(0, sample);
        for (size_t i = 0; i < segments_.size(); ++i) {
            if (clamped < segments_[i].end_sample) {
                return i;
            }
        }
        return segments_.size() - 1;
    }

    const std::string * text_ = nullptr;
    int32_t sample_rate_ = 24000;
    int64_t predicted_total_samples_ = 0;
    std::vector<text_segment> segments_;
};

class chunked_audio_stream {
public:
    bool init(AudioTokenizerDecoder * decoder,
              int32_t n_codebooks,
              int32_t chunk_frames,
              int32_t left_context_frames,
              const tts_audio_chunk_callback_t * callback,
              bool collect_audio,
              std::vector<float> * collected_audio,
              const text_span_estimator * text_estimator,
              std::string * error) {
        decoder_ = decoder;
        n_codebooks_ = n_codebooks;
        chunk_frames_ = std::max<int32_t>(1, chunk_frames);
        left_context_frames_ = std::max<int32_t>(0, left_context_frames);
        callback_ = callback;
        collect_audio_ = collect_audio;
        collected_audio_ = collected_audio;
        text_estimator_ = text_estimator;
        error_ = error;
        codes_.clear();
        total_frames_ = 0;
        emit_start_frame_ = 0;
        preloaded_frames_ = 0;
        emitted_samples_ = 0;
        cancelled_ = false;
        return decoder_ && n_codebooks_ > 0 && callback_ && *callback_;
    }

    void preload_context(const int32_t * codes, int32_t n_frames) {
        if (!codes || n_frames <= 0) {
            return;
        }
        const int32_t seed_frames = std::min<int32_t>(n_frames, left_context_frames_);
        if (seed_frames <= 0) {
            return;
        }
        const int32_t start_frame = n_frames - seed_frames;
        const int32_t * tail = codes + (size_t) start_frame * n_codebooks_;
        codes_.insert(codes_.end(), tail, tail + (size_t) seed_frames * n_codebooks_);
        total_frames_ = seed_frames;
        emit_start_frame_ = seed_frames;
        preloaded_frames_ = seed_frames;
    }

    bool push_frame(const int32_t * frame_codes) {
        if (!frame_codes) {
            set_error("Streaming frame callback received null codes");
            return false;
        }
        codes_.insert(codes_.end(), frame_codes, frame_codes + n_codebooks_);
        total_frames_++;
        while (total_frames_ - emit_start_frame_ >= chunk_frames_) {
            if (!emit_one(emit_start_frame_ + chunk_frames_)) {
                return false;
            }
        }
        return true;
    }

    bool flush() {
        if (total_frames_ > emit_start_frame_) {
            return emit_one(total_frames_);
        }
        return true;
    }

    bool cancelled() const {
        return cancelled_;
    }

    int64_t decode_ms() const {
        return decode_ms_;
    }

private:
    void set_error(const std::string & msg) {
        if (error_) {
            *error_ = msg;
        }
    }

    bool emit_one(int32_t end_frame) {
        const int32_t ctx = (emit_start_frame_ - left_context_frames_ > 0)
            ? left_context_frames_
            : emit_start_frame_;
        const int32_t slice_start = emit_start_frame_ - ctx;
        const int32_t slice_frames = end_frame - slice_start;
        if (slice_frames <= 0) {
            emit_start_frame_ = end_frame;
            return true;
        }

        std::vector<int32_t> slice((size_t) slice_frames * n_codebooks_);
        const size_t offset = (size_t) slice_start * n_codebooks_;
        std::copy(codes_.begin() + offset,
                  codes_.begin() + offset + slice.size(),
                  slice.begin());

        std::vector<float> decoded;
        const int64_t t_decode_start = get_time_ms();
        if (!decoder_->decode(slice.data(), slice_frames, decoded)) {
            set_error("Streaming vocoder decode failed: " + decoder_->get_error());
            return false;
        }
        decoder_->clear_decode_cache();
        decode_ms_ += get_time_ms() - t_decode_start;

        const size_t drop = slice_frames > 0
            ? (size_t) ((double) ctx / (double) slice_frames * (double) decoded.size() + 0.5)
            : 0;
        if (drop > decoded.size()) {
            set_error("Streaming vocoder context trim is out of range");
            return false;
        }
        const float * emit = decoded.data() + drop;
        const int32_t n_emit = (int32_t) (decoded.size() - drop);
        if (n_emit > 0) {
            tts_audio_chunk chunk;
            chunk.samples = emit;
            chunk.n_samples = n_emit;
            chunk.sample_rate = decoder_->get_config().sample_rate;
            chunk.start_sample = emitted_samples_;
            chunk.end_sample = emitted_samples_ + n_emit;
            chunk.start_frame = std::max<int32_t>(0, emit_start_frame_ - preloaded_frames_);
            chunk.end_frame = std::max<int32_t>(chunk.start_frame, end_frame - preloaded_frames_);
            if (text_estimator_) {
                text_estimator_->span_for_samples(chunk.start_sample,
                                                  chunk.end_sample,
                                                  chunk.start_text_byte,
                                                  chunk.end_text_byte,
                                                  chunk.text_alignment_kind,
                                                  chunk.confidence);
            }
            if (collect_audio_ && collected_audio_) {
                collected_audio_->insert(collected_audio_->end(), emit, emit + n_emit);
            }
            if (!(*callback_)(chunk)) {
                cancelled_ = true;
                set_error("Streaming audio callback requested cancellation");
                return false;
            }
            emitted_samples_ = chunk.end_sample;
        }
        emit_start_frame_ = end_frame;
        return true;
    }

    AudioTokenizerDecoder * decoder_ = nullptr;
    int32_t n_codebooks_ = 0;
    int32_t chunk_frames_ = 1;
    int32_t left_context_frames_ = 0;
    int32_t total_frames_ = 0;
    int32_t emit_start_frame_ = 0;
    int32_t preloaded_frames_ = 0;
    int64_t emitted_samples_ = 0;
    const tts_audio_chunk_callback_t * callback_ = nullptr;
    bool collect_audio_ = false;
    std::vector<float> * collected_audio_ = nullptr;
    const text_span_estimator * text_estimator_ = nullptr;
    std::string * error_ = nullptr;
    bool cancelled_ = false;
    int64_t decode_ms_ = 0;
    std::vector<int32_t> codes_;
};

} // namespace

tts_result Qwen3TTS::synthesize(const std::string & text,
                                const tts_params & params) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (!params.speaker.empty()) {
        std::vector<float> speaker_embedding;
        if (!transformer_.get_named_speaker_embedding(params.speaker, speaker_embedding)) {
            result.error_msg = "Failed to resolve speaker '" + params.speaker + "': " + transformer_.get_error();
            return result;
        }
        if (params.print_progress) {
            fprintf(stderr, "Using named speaker: %s (%zu floats)\n",
                    params.speaker.c_str(), speaker_embedding.size());
        }
        return ops::synthesize_internal(*this, text, speaker_embedding.data(), params, result);
    }

    const std::string & model_type = transformer_.get_config().tts_model_type;
    if (model_type == "custom_voice") {
        result.error_msg = "CustomVoice model requires --speaker, --reference, or --speaker-embedding";
        return result;
    }

    return ops::synthesize_internal(*this, text, nullptr, params, result);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                           const std::string & reference_audio,
                                           const tts_params & params) {
    tts_result result;

    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        result.error_msg = "Failed to load reference audio: " + reference_audio;
        return result;
    }

    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        fprintf(stderr, "Resampling audio from %d Hz to %d Hz...\n", ref_sample_rate, target_rate);
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int) ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }

    return synthesize_with_voice(text, ref_samples.data(), (int32_t) ref_samples.size(), params);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                           const float * ref_samples, int32_t n_ref_samples,
                                           const tts_params & params) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    int64_t t_encode_start = get_time_ms();
    std::vector<float> speaker_embedding;
    tts_params effective_params = params;
    const bool needs_reference_codes =
        !effective_params.reference_codes.has_value() &&
        (!effective_params.reference_text.empty() ||
         !effective_params.reference_token_ids.empty());
    const bool speech_encoder_cuda = speech_encoder_cuda_enabled();
    const uint64_t sample_hash = hash_reference_samples(ref_samples, n_ref_samples);
    const bool cache_hit =
        voice_prompt_cache_.valid &&
        voice_prompt_cache_.sample_hash == sample_hash &&
        voice_prompt_cache_.n_samples == n_ref_samples &&
        voice_prompt_cache_.reference_text == effective_params.reference_text &&
        voice_prompt_cache_.reference_token_ids == effective_params.reference_token_ids &&
        voice_prompt_cache_.has_auto_reference_codes == needs_reference_codes &&
        (!needs_reference_codes || voice_prompt_cache_.reference_codes_cuda == speech_encoder_cuda) &&
        (!needs_reference_codes || voice_prompt_cache_.reference_codes.has_value());

    if (cache_hit) {
        speaker_embedding = voice_prompt_cache_.speaker_embedding;
        if (needs_reference_codes) {
            effective_params.reference_codes = voice_prompt_cache_.reference_codes;
        }
        if (params.print_progress || params.print_timing) {
            fprintf(stderr, "Voice prompt cache hit: reused speaker embedding%s\n",
                    needs_reference_codes ? " and reference speech codes" : "");
        }
    } else {
        if (!encoder_loaded_) {
            if (speaker_encoder_model_path_.empty()) {
                result.error_msg = "Internal error: missing TTS model path for lazy encoder load";
                return result;
            }
            int64_t t_encoder_load_start = get_time_ms();
            if (!audio_encoder_.load_model(speaker_encoder_model_path_)) {
                result.error_msg = "Failed to load speaker encoder: " + audio_encoder_.get_error();
                return result;
            }
            encoder_loaded_ = true;
            if (params.print_timing) {
                fprintf(stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                        (long long) (get_time_ms() - t_encoder_load_start));
                log_memory_usage("voice/after-encoder-load");
            }
        }

        if (!audio_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
            result.error_msg = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
            return result;
        }
        if (needs_reference_codes) {
            if (tokenizer_model_path_.empty()) {
                result.error_msg = "Internal error: missing tokenizer model path for speech tokenizer encoder";
                return result;
            }
            if (speech_encoder_loaded_ && speech_encoder_cuda_enabled_ != speech_encoder_cuda) {
                speech_encoder_.unload_model();
                speech_encoder_loaded_ = false;
                speech_encoder_cuda_enabled_ = false;
            }
            if (!speech_encoder_loaded_) {
                const int64_t t_speech_encoder_load_start = get_time_ms();
                if (!speech_encoder_.load_model(tokenizer_model_path_, !speech_encoder_cuda)) {
                    result.error_msg = "Failed to load speech tokenizer encoder: " + speech_encoder_.get_error();
                    return result;
                }
                speech_encoder_loaded_ = true;
                speech_encoder_cuda_enabled_ = speech_encoder_cuda;
                if (params.print_timing) {
                    fprintf(stderr, "  Speech tokenizer encoder lazy-loaded in %lld ms\n",
                            (long long) (get_time_ms() - t_speech_encoder_load_start));
                    log_memory_usage("voice/after-speech-encoder-load");
                }
            }
            speech_codes reference_codes;
            if (!speech_encoder_.encode(ref_samples, n_ref_samples, reference_codes)) {
                result.error_msg = "Failed to tokenize reference audio: " + speech_encoder_.get_error();
                return result;
            }
            if (params.print_progress) {
                fprintf(stderr, "Reference audio tokenized: %d frames x %d codebooks\n",
                        reference_codes.n_frames, reference_codes.n_codebooks);
            }
            effective_params.reference_codes = std::move(reference_codes);
        }

        voice_prompt_cache_.valid = true;
        voice_prompt_cache_.sample_hash = sample_hash;
        voice_prompt_cache_.n_samples = n_ref_samples;
        voice_prompt_cache_.reference_text = effective_params.reference_text;
        voice_prompt_cache_.reference_token_ids = effective_params.reference_token_ids;
        voice_prompt_cache_.has_auto_reference_codes = needs_reference_codes;
        voice_prompt_cache_.reference_codes_cuda = needs_reference_codes && speech_encoder_cuda;
        voice_prompt_cache_.speaker_embedding = speaker_embedding;
        voice_prompt_cache_.reference_codes = needs_reference_codes
            ? effective_params.reference_codes
            : std::optional<speech_codes>();
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;

    const int expected_dim = transformer_.get_config().hidden_size;
    if ((int) speaker_embedding.size() != expected_dim) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Speaker embedding dimension mismatch after extraction: got %zu, expected %d",
                 speaker_embedding.size(), expected_dim);
        result.error_msg = buf;
        return result;
    }

    if (params.print_progress) {
        fprintf(stderr, "Speaker embedding extracted: %zu floats\n", speaker_embedding.size());
    }

    return ops::synthesize_internal(*this, text, speaker_embedding.data(), effective_params, result);
}

tts_result Qwen3TTS::synthesize_with_speaker_embedding(const std::string & text,
                                                       const std::vector<float> & speaker_embedding,
                                                       const tts_params & params) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    const int expected_dim = transformer_.get_config().hidden_size;
    if ((int) speaker_embedding.size() != expected_dim) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Speaker embedding dimension mismatch: got %zu, expected %d",
                 speaker_embedding.size(), expected_dim);
        result.error_msg = buf;
        return result;
    }

    if (params.print_progress) {
        fprintf(stderr, "Using provided speaker embedding: %zu floats\n", speaker_embedding.size());
    }

    result.t_encode_ms = 0;
    return ops::synthesize_internal(*this, text, speaker_embedding.data(), params, result);
}

tts_result Qwen3TTS::synthesize_streaming(const std::string & text,
                                          const tts_audio_chunk_callback_t & on_audio_chunk,
                                          const tts_streaming_params & stream_params) {
    tts_result result;
    const tts_params & params = stream_params.generation;

    if (!on_audio_chunk) {
        result.error_msg = "Streaming audio callback is not set";
        return result;
    }
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (!params.speaker.empty()) {
        std::vector<float> speaker_embedding;
        if (!transformer_.get_named_speaker_embedding(params.speaker, speaker_embedding)) {
            result.error_msg = "Failed to resolve speaker '" + params.speaker + "': " + transformer_.get_error();
            return result;
        }
        return ops::synthesize_internal(*this, text, speaker_embedding.data(), params, result,
                                        &stream_params, &on_audio_chunk);
    }

    const std::string & model_type = transformer_.get_config().tts_model_type;
    if (model_type == "custom_voice") {
        result.error_msg = "CustomVoice model requires --speaker, --reference, or --speaker-embedding";
        return result;
    }

    return ops::synthesize_internal(*this, text, nullptr, params, result,
                                    &stream_params, &on_audio_chunk);
}

tts_result Qwen3TTS::synthesize_with_voice_streaming(
    const std::string & text,
    const std::string & reference_audio,
    const tts_audio_chunk_callback_t & on_audio_chunk,
    const tts_streaming_params & stream_params) {
    tts_result result;

    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        result.error_msg = "Failed to load reference audio: " + reference_audio;
        return result;
    }

    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        fprintf(stderr, "Resampling audio from %d Hz to %d Hz...\n", ref_sample_rate, target_rate);
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int) ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }

    return synthesize_with_voice_streaming(text, ref_samples.data(), (int32_t) ref_samples.size(),
                                           on_audio_chunk, stream_params);
}

tts_result Qwen3TTS::synthesize_with_voice_streaming(
    const std::string & text,
    const float * ref_samples,
    int32_t n_ref_samples,
    const tts_audio_chunk_callback_t & on_audio_chunk,
    const tts_streaming_params & stream_params) {
    tts_result result;
    const tts_params & params = stream_params.generation;

    if (!on_audio_chunk) {
        result.error_msg = "Streaming audio callback is not set";
        return result;
    }
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    int64_t t_encode_start = get_time_ms();
    std::vector<float> speaker_embedding;
    tts_params effective_params = params;
    const bool needs_reference_codes =
        !effective_params.reference_codes.has_value() &&
        (!effective_params.reference_text.empty() ||
         !effective_params.reference_token_ids.empty());
    const bool speech_encoder_cuda = speech_encoder_cuda_enabled();
    const uint64_t sample_hash = hash_reference_samples(ref_samples, n_ref_samples);
    const bool cache_hit =
        voice_prompt_cache_.valid &&
        voice_prompt_cache_.sample_hash == sample_hash &&
        voice_prompt_cache_.n_samples == n_ref_samples &&
        voice_prompt_cache_.reference_text == effective_params.reference_text &&
        voice_prompt_cache_.reference_token_ids == effective_params.reference_token_ids &&
        voice_prompt_cache_.has_auto_reference_codes == needs_reference_codes &&
        (!needs_reference_codes || voice_prompt_cache_.reference_codes_cuda == speech_encoder_cuda) &&
        (!needs_reference_codes || voice_prompt_cache_.reference_codes.has_value());

    if (cache_hit) {
        speaker_embedding = voice_prompt_cache_.speaker_embedding;
        if (needs_reference_codes) {
            effective_params.reference_codes = voice_prompt_cache_.reference_codes;
        }
    } else {
        if (!encoder_loaded_) {
            if (speaker_encoder_model_path_.empty()) {
                result.error_msg = "Internal error: missing TTS model path for lazy encoder load";
                return result;
            }
            if (!audio_encoder_.load_model(speaker_encoder_model_path_)) {
                result.error_msg = "Failed to load speaker encoder: " + audio_encoder_.get_error();
                return result;
            }
            encoder_loaded_ = true;
        }

        if (!audio_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
            result.error_msg = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
            return result;
        }
        if (needs_reference_codes) {
            if (tokenizer_model_path_.empty()) {
                result.error_msg = "Internal error: missing tokenizer model path for speech tokenizer encoder";
                return result;
            }
            if (speech_encoder_loaded_ && speech_encoder_cuda_enabled_ != speech_encoder_cuda) {
                speech_encoder_.unload_model();
                speech_encoder_loaded_ = false;
                speech_encoder_cuda_enabled_ = false;
            }
            if (!speech_encoder_loaded_) {
                if (!speech_encoder_.load_model(tokenizer_model_path_, !speech_encoder_cuda)) {
                    result.error_msg = "Failed to load speech tokenizer encoder: " + speech_encoder_.get_error();
                    return result;
                }
                speech_encoder_loaded_ = true;
                speech_encoder_cuda_enabled_ = speech_encoder_cuda;
            }
            speech_codes reference_codes;
            if (!speech_encoder_.encode(ref_samples, n_ref_samples, reference_codes)) {
                result.error_msg = "Failed to tokenize reference audio: " + speech_encoder_.get_error();
                return result;
            }
            effective_params.reference_codes = std::move(reference_codes);
        }

        voice_prompt_cache_.valid = true;
        voice_prompt_cache_.sample_hash = sample_hash;
        voice_prompt_cache_.n_samples = n_ref_samples;
        voice_prompt_cache_.reference_text = effective_params.reference_text;
        voice_prompt_cache_.reference_token_ids = effective_params.reference_token_ids;
        voice_prompt_cache_.has_auto_reference_codes = needs_reference_codes;
        voice_prompt_cache_.reference_codes_cuda = needs_reference_codes && speech_encoder_cuda;
        voice_prompt_cache_.speaker_embedding = speaker_embedding;
        voice_prompt_cache_.reference_codes = needs_reference_codes
            ? effective_params.reference_codes
            : std::optional<speech_codes>();
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;

    const int expected_dim = transformer_.get_config().hidden_size;
    if ((int) speaker_embedding.size() != expected_dim) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Speaker embedding dimension mismatch after extraction: got %zu, expected %d",
                 speaker_embedding.size(), expected_dim);
        result.error_msg = buf;
        return result;
    }

    return ops::synthesize_internal(*this, text, speaker_embedding.data(), effective_params, result,
                                    &stream_params, &on_audio_chunk);
}

tts_result Qwen3TTS::synthesize_with_speaker_embedding_streaming(
    const std::string & text,
    const std::vector<float> & speaker_embedding,
    const tts_audio_chunk_callback_t & on_audio_chunk,
    const tts_streaming_params & stream_params) {
    tts_result result;

    if (!on_audio_chunk) {
        result.error_msg = "Streaming audio callback is not set";
        return result;
    }
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    const int expected_dim = transformer_.get_config().hidden_size;
    if ((int) speaker_embedding.size() != expected_dim) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Speaker embedding dimension mismatch: got %zu, expected %d",
                 speaker_embedding.size(), expected_dim);
        result.error_msg = buf;
        return result;
    }

    result.t_encode_ms = 0;
    return ops::synthesize_internal(*this, text, speaker_embedding.data(), stream_params.generation, result,
                                    &stream_params, &on_audio_chunk);
}

bool Qwen3TTS::extract_speaker_embedding(const std::string & reference_audio,
                                         std::vector<float> & speaker_embedding,
                                         int64_t * encode_time_ms) {
    if (!models_loaded_ && !encoder_loaded_) {
        error_msg_ = "Models not loaded";
        return false;
    }

    std::vector<float> ref_samples;
    int ref_sample_rate = 0;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        error_msg_ = "Failed to load reference audio: " + reference_audio;
        return false;
    }

    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        fprintf(stderr, "Resampling audio from %d Hz to %d Hz...\n", ref_sample_rate, target_rate);
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int) ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }

    if (!encoder_loaded_) {
        if (speaker_encoder_model_path_.empty()) {
            error_msg_ = "Internal error: missing TTS model path for lazy encoder load";
            return false;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(speaker_encoder_model_path_)) {
            error_msg_ = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return false;
        }
        encoder_loaded_ = true;
        fprintf(stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                (long long) (get_time_ms() - t_encoder_load_start));
        log_memory_usage("voice/after-encoder-load");
    }

    const int64_t t_encode_start = get_time_ms();
    if (!audio_encoder_.encode(ref_samples.data(), (int32_t) ref_samples.size(), speaker_embedding)) {
        error_msg_ = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return false;
    }

    const int expected_dim = transformer_loaded_
        ? transformer_.get_config().hidden_size
        : audio_encoder_.get_config().embedding_dim;
    if ((int) speaker_embedding.size() != expected_dim) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Speaker embedding dimension mismatch after extraction: got %zu, expected %d",
                 speaker_embedding.size(), expected_dim);
        error_msg_ = buf;
        return false;
    }

    if (encode_time_ms) {
        *encode_time_ms = get_time_ms() - t_encode_start;
    }
    return true;
}

bool Qwen3TTS::extract_icl_prompt(const std::string & reference_audio,
                                  const std::string & reference_text,
                                  icl_prompt & prompt,
                                  int64_t * encode_time_ms) {
    if (tts_model_path_.empty() || tokenizer_model_path_.empty()) {
        error_msg_ = "ICL prompt extraction encoders not loaded";
        return false;
    }
    if (!encoder_loaded_ && speaker_encoder_model_path_.empty()) {
        error_msg_ = "Speaker encoder not loaded";
        return false;
    }
    if (reference_text.empty()) {
        error_msg_ = "ICL prompt extraction requires reference text";
        return false;
    }

    std::vector<float> ref_samples;
    int ref_sample_rate = 0;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        error_msg_ = "Failed to load reference audio: " + reference_audio;
        return false;
    }

    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        fprintf(stderr, "Resampling audio from %d Hz to %d Hz...\n", ref_sample_rate, target_rate);
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int) ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }

    const int64_t t_encode_start = get_time_ms();
    if (!encoder_loaded_) {
        if (speaker_encoder_model_path_.empty()) {
            error_msg_ = "Internal error: missing TTS model path for lazy encoder load";
            return false;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(speaker_encoder_model_path_)) {
            error_msg_ = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return false;
        }
        encoder_loaded_ = true;
        fprintf(stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                (long long) (get_time_ms() - t_encoder_load_start));
        log_memory_usage("icl/after-encoder-load");
    }

    icl_prompt extracted;
    fprintf(stderr, "  Encoding ICL speaker embedding...\n");
    if (!audio_encoder_.encode(ref_samples.data(), (int32_t) ref_samples.size(),
                               extracted.speaker_embedding)) {
        error_msg_ = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return false;
    }
    fprintf(stderr, "  ICL speaker embedding encoded: %zu floats\n",
            extracted.speaker_embedding.size());

    const int expected_dim = audio_encoder_.get_config().embedding_dim;
    if ((int) extracted.speaker_embedding.size() != expected_dim) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Speaker embedding dimension mismatch after ICL extraction: got %zu, expected %d",
                 extracted.speaker_embedding.size(), expected_dim);
        error_msg_ = buf;
        return false;
    }
    fprintf(stderr, "  ICL speaker embedding validated\n");

    if (tokenizer_model_path_.empty()) {
        error_msg_ = "Internal error: missing tokenizer model path for speech tokenizer encoder";
        return false;
    }
    const bool speech_encoder_cuda = speech_encoder_cuda_enabled();
    if (speech_encoder_loaded_ && speech_encoder_cuda_enabled_ != speech_encoder_cuda) {
        speech_encoder_.unload_model();
        speech_encoder_loaded_ = false;
        speech_encoder_cuda_enabled_ = false;
    }
    if (!speech_encoder_loaded_) {
        const int64_t t_speech_encoder_load_start = get_time_ms();
        if (!speech_encoder_.load_model(tokenizer_model_path_, !speech_encoder_cuda)) {
            error_msg_ = "Failed to load speech tokenizer encoder: " + speech_encoder_.get_error();
            return false;
        }
        speech_encoder_loaded_ = true;
        speech_encoder_cuda_enabled_ = speech_encoder_cuda;
        fprintf(stderr, "  Speech tokenizer encoder lazy-loaded in %lld ms\n",
                (long long) (get_time_ms() - t_speech_encoder_load_start));
        log_memory_usage("icl/after-speech-encoder-load");
    }

    if (!speech_encoder_.encode(ref_samples.data(), (int32_t) ref_samples.size(),
                                extracted.reference_codes)) {
        error_msg_ = "Failed to tokenize reference audio: " + speech_encoder_.get_error();
        return false;
    }
    if (!models_loaded_) {
        speech_encoder_.unload_model();
        speech_encoder_loaded_ = false;
        speech_encoder_cuda_enabled_ = false;
    }

    if (!text_tokenizer_loaded_) {
        const int64_t t_tokenizer_start = get_time_ms();
        GGUFLoader loader;
        if (!loader.open(tts_model_path_)) {
            error_msg_ = "Failed to open TTS model for text tokenizer: " + loader.get_error();
            return false;
        }
        if (!tokenizer_.load_from_gguf(loader.get_ctx())) {
            error_msg_ = "Failed to load text tokenizer: " + tokenizer_.get_error();
            return false;
        }
        text_tokenizer_loaded_ = true;
        fprintf(stderr, "  Text tokenizer lazy-loaded in %lld ms\n",
                (long long) (get_time_ms() - t_tokenizer_start));
    }

    extracted.reference_text = reference_text;
    extracted.reference_token_ids = tokenizer_.encode_reference_for_tts(reference_text);
    if (extracted.reference_token_ids.empty()) {
        error_msg_ = "Failed to tokenize reference text";
        return false;
    }

    if (encode_time_ms) {
        *encode_time_ms = get_time_ms() - t_encode_start;
    }
    prompt = std::move(extracted);
    return true;
}

tts_result pipeline_internal::ops::synthesize_internal(Qwen3TTS & self,
                                                       const std::string & text,
                                                       const float * speaker_embedding,
                                                       const tts_params & params,
                                                       tts_result & result,
                                                       const tts_streaming_params * streaming_params,
                                                       const tts_audio_chunk_callback_t * on_audio_chunk) {
    int64_t t_total_start = get_time_ms() - result.t_encode_ms;
    auto sample_memory = [&](const char * stage) {
        process_memory_snapshot mem;
        if (!get_process_memory_snapshot(mem)) {
            return;
        }
        if (result.mem_rss_start_bytes == 0) {
            result.mem_rss_start_bytes = mem.rss_bytes;
            result.mem_phys_start_bytes = mem.phys_footprint_bytes;
        }
        result.mem_rss_end_bytes = mem.rss_bytes;
        result.mem_phys_end_bytes = mem.phys_footprint_bytes;
        if (mem.rss_bytes > result.mem_rss_peak_bytes) {
            result.mem_rss_peak_bytes = mem.rss_bytes;
        }
        if (mem.phys_footprint_bytes > result.mem_phys_peak_bytes) {
            result.mem_phys_peak_bytes = mem.phys_footprint_bytes;
        }
        if (params.print_timing) {
            fprintf(stderr, "  [mem] %-24s rss=%s  phys=%s\n",
                    stage,
                    format_bytes(mem.rss_bytes).c_str(),
                    format_bytes(mem.phys_footprint_bytes).c_str());
        }
    };
    sample_memory("synth/start");

    int64_t t_tokenize_start = get_time_ms();
    std::vector<int32_t> text_tokens = self.tokenizer_.encode_for_tts(text);
    std::vector<int32_t> instruct_tokens;
    if (!params.instruction.empty()) {
        instruct_tokens = self.tokenizer_.encode_instruct(params.instruction);
    }
    std::vector<int32_t> reference_tokens;
    if (params.reference_codes.has_value()) {
        if (!params.reference_token_ids.empty()) {
            reference_tokens = params.reference_token_ids;
        } else if (params.reference_text.empty()) {
            result.error_msg = "ICL reference codes require reference_text";
            return result;
        } else {
            reference_tokens = self.tokenizer_.encode_reference_for_tts(params.reference_text);
        }
    }
    result.t_tokenize_ms = get_time_ms() - t_tokenize_start;
    sample_memory("synth/after-tokenize");

    if (text_tokens.empty()) {
        result.error_msg = "Failed to tokenize text";
        return result;
    }
    if (!params.instruction.empty() && instruct_tokens.empty()) {
        result.error_msg = "Failed to tokenize instruction";
        return result;
    }
    if (params.reference_codes.has_value() && reference_tokens.empty()) {
        result.error_msg = "Failed to tokenize reference text";
        return result;
    }

    if (params.print_progress) {
        fprintf(stderr, "Text tokenized: %zu tokens\n", text_tokens.size());
        if (!instruct_tokens.empty()) {
            fprintf(stderr, "Instruction tokenized: %zu tokens\n", instruct_tokens.size());
        }
        if (!reference_tokens.empty()) {
            fprintf(stderr, "Reference text tokenized: %zu tokens\n", reference_tokens.size());
        }
        fprintf(stderr, "  Tokens: ");
        for (size_t i = 0; i < std::min(text_tokens.size(), (size_t) 10); ++i) {
            fprintf(stderr, "%d ", text_tokens[i]);
        }
        if (text_tokens.size() > 10) fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }

    int64_t t_generate_start = get_time_ms();
    if (!self.transformer_loaded_) {
        int64_t t_reload_start = get_time_ms();
        if (!self.transformer_.load_model(self.tts_model_path_)) {
            result.error_msg = "Failed to reload TTS transformer: " + self.transformer_.get_error();
            return result;
        }
        self.transformer_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Transformer reloaded in %lld ms\n",
                    (long long) (get_time_ms() - t_reload_start));
            sample_memory("synth/after-transformer-reload");
        }
    }
    self.transformer_.clear_kv_cache();

    speech_codes reference_codes;
    const speech_codes * reference_codes_ptr = nullptr;
    if (params.reference_codes.has_value()) {
        if (!speaker_embedding) {
            result.error_msg = "ICL reference codes require a speaker embedding";
            return result;
        }
        reference_codes = *params.reference_codes;
        const int32_t model_codebooks = self.transformer_.get_config().n_codebooks;
        if (reference_codes.n_codebooks == 0) {
            reference_codes.n_codebooks = model_codebooks;
        } else if (reference_codes.n_codebooks != model_codebooks) {
            result.error_msg = "ICL reference codebook count does not match loaded model";
            return result;
        }
        if (reference_codes.codes.empty() ||
            reference_codes.codes.size() % (size_t) reference_codes.n_codebooks != 0) {
            result.error_msg = "ICL reference code count is not divisible by codebook count";
            return result;
        }
        const int32_t inferred_frames =
            (int32_t) (reference_codes.codes.size() / (size_t) reference_codes.n_codebooks);
        if (reference_codes.n_frames == 0) {
            reference_codes.n_frames = inferred_frames;
        } else if (reference_codes.n_frames != inferred_frames) {
            result.error_msg = "ICL reference frame count does not match code count";
            return result;
        }
        reference_codes_ptr = &reference_codes;
    }

    const bool streaming = streaming_params && on_audio_chunk && *on_audio_chunk;
    chunked_audio_stream stream;
    text_span_estimator stream_text_estimator;
    std::string stream_error;
    if (streaming) {
        if (!self.streaming_decoder_loaded_) {
            int64_t t_decoder_load_start = get_time_ms();
            if (self.decoder_model_path_.empty()) {
                result.error_msg = "Internal error: missing vocoder model path";
                return result;
            }
            if (!self.streaming_audio_decoder_.load_model_dedicated(self.decoder_model_path_)) {
                result.error_msg = "Failed to load streaming vocoder: " + self.streaming_audio_decoder_.get_error();
                return result;
            }
            self.streaming_decoder_loaded_ = true;
            if (params.print_timing) {
                fprintf(stderr, "  Vocoder lazy-loaded for streaming in %lld ms\n",
                        (long long) (get_time_ms() - t_decoder_load_start));
                sample_memory("synth/after-vocoder-load");
            }
        }
        const audio_decoder_config & decoder_cfg = self.streaming_audio_decoder_.get_config();
        const int32_t chunk_frames = duration_sec_to_codec_frames(decoder_cfg, streaming_params->chunk_sec, 1);
        const int32_t left_context_frames =
            duration_sec_to_codec_frames(decoder_cfg, streaming_params->left_context_sec, 0);
        stream_text_estimator.reset(text, decoder_cfg.sample_rate);
        if (!stream.init(&self.streaming_audio_decoder_, self.transformer_.get_config().n_codebooks,
                         chunk_frames, left_context_frames, on_audio_chunk,
                         streaming_params->collect_audio, &result.audio,
                         &stream_text_estimator, &stream_error)) {
            result.error_msg = "Failed to initialize streaming decoder";
            return result;
        }
        if (reference_codes_ptr) {
            stream.preload_context(reference_codes_ptr->codes.data(), reference_codes_ptr->n_frames);
        }
    }

    std::vector<int32_t> generated_codes;
    tts_code_frame_callback_t frame_callback;
    if (streaming) {
        frame_callback = [&](const int32_t * frame_codes, int32_t frame_codebooks, int32_t frame_index) {
            if (self.progress_callback_) {
                self.progress_callback_(frame_index + 1, params.max_audio_tokens);
            }
            if (frame_codebooks != self.transformer_.get_config().n_codebooks) {
                stream_error = "Streaming frame codebook count mismatch";
                return false;
            }
            return stream.push_frame(frame_codes);
        };
    }
    if (!self.transformer_.generate(text_tokens.data(), (int32_t) text_tokens.size(),
                                    speaker_embedding, params.max_audio_tokens, generated_codes,
                                    params.language_id, params.repetition_penalty,
                                    params.temperature, params.top_k, params.top_p, params.seed,
                                    instruct_tokens.empty() ? nullptr : instruct_tokens.data(),
                                    (int32_t) instruct_tokens.size(),
                                    reference_tokens.empty() ? nullptr : reference_tokens.data(),
                                    (int32_t) reference_tokens.size(),
                                    reference_codes_ptr ? reference_codes_ptr->codes.data() : nullptr,
                                    reference_codes_ptr ? reference_codes_ptr->n_frames : 0,
                                    reference_codes_ptr ? reference_codes_ptr->n_codebooks : 0,
                                    streaming ? &frame_callback : nullptr)) {
        result.error_msg = stream_error.empty()
            ? "Failed to generate speech codes: " + self.transformer_.get_error()
            : stream_error;
        return result;
    }
    if (streaming && !stream.flush()) {
        result.error_msg = stream_error.empty()
            ? "Streaming vocoder flush failed"
            : stream_error;
        return result;
    }
    result.t_generate_ms = get_time_ms() - t_generate_start;
    sample_memory("synth/after-generate");

    int n_codebooks = self.transformer_.get_config().n_codebooks;
    int n_frames = (int) generated_codes.size() / n_codebooks;

    if (params.print_progress) {
        fprintf(stderr, "Speech codes generated: %d frames x %d codebooks\n", n_frames, n_codebooks);
    }

    if (n_frames == 0) {
        result.error_msg = "No speech codes generated";
        return result;
    }
    if (!params.dump_generated_codes_path.empty()) {
        if (!write_codes_file(params.dump_generated_codes_path, generated_codes,
                              n_frames, n_codebooks, result.error_msg)) {
            return result;
        }
    }

    if (streaming) {
        if (!params.dump_decoder_codes_path.empty()) {
            std::vector<int32_t> decoder_codes;
            const int32_t decoder_frames = reference_codes_ptr ? n_frames + reference_codes_ptr->n_frames : n_frames;
            const std::vector<int32_t> * dump_codes = &generated_codes;
            if (reference_codes_ptr) {
                decoder_codes.reserve(reference_codes_ptr->codes.size() + generated_codes.size());
                decoder_codes.insert(decoder_codes.end(),
                                     reference_codes_ptr->codes.begin(),
                                     reference_codes_ptr->codes.end());
                decoder_codes.insert(decoder_codes.end(), generated_codes.begin(), generated_codes.end());
                dump_codes = &decoder_codes;
            }
            if (!write_codes_file(params.dump_decoder_codes_path, *dump_codes,
                                  decoder_frames, n_codebooks, result.error_msg)) {
                return result;
            }
        }

        result.sample_rate = self.streaming_audio_decoder_.get_config().sample_rate;
        result.t_decode_ms = stream.decode_ms();

        if (self.low_mem_mode_) {
            self.transformer_.unload_model();
            self.transformer_loaded_ = false;
            sample_memory("synth/after-transformer-unload");
            self.streaming_audio_decoder_.unload_model();
            self.streaming_decoder_loaded_ = false;
            sample_memory("synth/after-vocoder-unload");
        }

        result.decode_frames = n_frames;
        result.decode_samples = (int64_t) result.audio.size();
        result.success = true;
        result.t_total_ms = get_time_ms() - t_total_start;
        sample_memory("synth/end");

        if (params.print_timing) {
            const double audio_sec = result.sample_rate > 0
                ? (double) result.audio.size() / (double) result.sample_rate : 0.0;
            const double wall_sec = (double) result.t_total_ms / 1000.0;
            const double realtime_factor = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
            const double x_realtime = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
            fprintf(stderr, "\nTiming:\n");
            fprintf(stderr, "  Tokenization:    %lld ms\n", (long long) result.t_tokenize_ms);
            fprintf(stderr, "  Speaker encode:  %lld ms\n", (long long) result.t_encode_ms);
            fprintf(stderr, "  Code+streaming:  %lld ms\n", (long long) result.t_generate_ms);
            fprintf(stderr, "  Streaming decode:%lld ms\n", (long long) result.t_decode_ms);
            fprintf(stderr, "  Total:           %lld ms\n", (long long) result.t_total_ms);
            fprintf(stderr, "  Collected audio: %.2f s%s\n", audio_sec,
                    streaming_params->collect_audio ? "" : " (disabled)");
            fprintf(stderr, "  Throughput:      %.2fx realtime (RTF=%.3f)\n", x_realtime, realtime_factor);
        }

        return result;
    }

    if (self.low_mem_mode_) {
        self.transformer_.unload_model();
        self.transformer_loaded_ = false;
        sample_memory("synth/after-transformer-unload");
    }

    int64_t t_decode_start = get_time_ms();
    if (!self.decoder_loaded_) {
        int64_t t_decoder_load_start = get_time_ms();
        if (self.decoder_model_path_.empty()) {
            result.error_msg = "Internal error: missing vocoder model path";
            return result;
        }
        if (!self.audio_decoder_.load_model(self.decoder_model_path_)) {
            result.error_msg = "Failed to load vocoder: " + self.audio_decoder_.get_error();
            return result;
        }
        self.decoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Vocoder lazy-loaded in %lld ms\n",
                    (long long) (get_time_ms() - t_decoder_load_start));
            sample_memory("synth/after-vocoder-load");
        }
    }

    std::vector<int32_t> decoder_codes;
    const audio_decoder_config & decoder_cfg = self.audio_decoder_.get_config();
    int32_t reference_context_frames = 0;
    if (reference_codes_ptr) {
        const int32_t max_context_frames =
            duration_sec_to_codec_frames(decoder_cfg, params.vocoder_left_context_sec, 0);
        reference_context_frames = std::min<int32_t>(reference_codes_ptr->n_frames, max_context_frames);
    }
    const int32_t decoder_frames = n_frames + reference_context_frames;
    const int32_t * decoder_code_data = generated_codes.data();
    if (reference_context_frames > 0) {
        const size_t context_codes = (size_t) reference_context_frames * (size_t) n_codebooks;
        decoder_codes.reserve(context_codes + generated_codes.size());
        const int32_t context_start_frame = reference_codes_ptr->n_frames - reference_context_frames;
        const auto ref_begin = reference_codes_ptr->codes.begin() +
            (size_t) context_start_frame * (size_t) n_codebooks;
        decoder_codes.insert(decoder_codes.end(), ref_begin, ref_begin + context_codes);
        decoder_codes.insert(decoder_codes.end(), generated_codes.begin(), generated_codes.end());
        decoder_code_data = decoder_codes.data();
    }
    if (!params.dump_decoder_codes_path.empty()) {
        const auto & dump_codes = reference_context_frames > 0 ? decoder_codes : generated_codes;
        if (!write_codes_file(params.dump_decoder_codes_path, dump_codes,
                              decoder_frames, n_codebooks, result.error_msg)) {
            return result;
        }
    }

    if (!self.audio_decoder_.decode(decoder_code_data, decoder_frames, result.audio)) {
        result.error_msg = "Failed to decode speech codes: " + self.audio_decoder_.get_error();
        return result;
    }
    const audio_decoder_timing & decoder_timing = self.audio_decoder_.get_last_timing();
    result.t_decode_graph_build_ms = decoder_timing.graph_build_ms;
    result.t_decode_graph_alloc_ms = decoder_timing.graph_alloc_ms;
    result.t_decode_input_upload_ms = decoder_timing.input_upload_ms;
    result.t_decode_graph_compute_ms = decoder_timing.graph_compute_ms;
    result.t_decode_output_read_ms = decoder_timing.output_read_ms;
    result.decode_graph_rebuilt = decoder_timing.graph_rebuilt;
    result.decode_frames = decoder_timing.n_frames;
    result.decode_samples = decoder_timing.n_samples;
    if (reference_context_frames > 0) {
        const int64_t cut = (int64_t) reference_context_frames *
            (int64_t) qwen3_tts_codec_hop_length;
        if (cut < 0 || cut > (int64_t) result.audio.size()) {
            result.error_msg = "ICL reference trim is out of range";
            return result;
        }
        result.audio.erase(result.audio.begin(), result.audio.begin() + cut);
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    sample_memory("synth/after-decode");

    if (self.low_mem_mode_) {
        self.audio_decoder_.unload_model();
        self.decoder_loaded_ = false;
        sample_memory("synth/after-vocoder-unload");
    }

    result.sample_rate = self.audio_decoder_.get_config().sample_rate;
    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;
    sample_memory("synth/end");

    if (params.print_timing) {
        const double audio_sec = result.sample_rate > 0
            ? (double) result.audio.size() / (double) result.sample_rate : 0.0;
        const double wall_sec = (double) result.t_total_ms / 1000.0;
        const double realtime_factor = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
        const double x_realtime = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenization:    %lld ms\n", (long long) result.t_tokenize_ms);
        fprintf(stderr, "  Speaker encode:  %lld ms\n", (long long) result.t_encode_ms);
        fprintf(stderr, "  Code generation: %lld ms\n", (long long) result.t_generate_ms);
        fprintf(stderr, "  Vocoder decode:  %lld ms\n", (long long) result.t_decode_ms);
        fprintf(stderr, "    graph build:   %lld ms%s\n",
                (long long) result.t_decode_graph_build_ms,
                result.decode_graph_rebuilt ? " (rebuilt)" : " (cached)");
        fprintf(stderr, "    graph alloc:   %lld ms\n", (long long) result.t_decode_graph_alloc_ms);
        fprintf(stderr, "    input upload:  %lld ms\n", (long long) result.t_decode_input_upload_ms);
        fprintf(stderr, "    graph compute: %lld ms\n", (long long) result.t_decode_graph_compute_ms);
        fprintf(stderr, "    output read:   %lld ms\n", (long long) result.t_decode_output_read_ms);
        fprintf(stderr, "    frames/samples:%d / %lld\n",
                result.decode_frames, (long long) result.decode_samples);
        fprintf(stderr, "  Total:           %lld ms\n", (long long) result.t_total_ms);
        fprintf(stderr, "  Audio duration:  %.2f s\n", audio_sec);
        fprintf(stderr, "  Throughput:      %.2fx realtime (RTF=%.3f)\n", x_realtime, realtime_factor);
        fprintf(stderr, "\nMemory:\n");
        fprintf(stderr, "  RSS start/end:   %s -> %s\n",
                format_bytes(result.mem_rss_start_bytes).c_str(),
                format_bytes(result.mem_rss_end_bytes).c_str());
        fprintf(stderr, "  RSS peak:        %s\n",
                format_bytes(result.mem_rss_peak_bytes).c_str());
        fprintf(stderr, "  Phys start/end:  %s -> %s\n",
                format_bytes(result.mem_phys_start_bytes).c_str(),
                format_bytes(result.mem_phys_end_bytes).c_str());
        fprintf(stderr, "  Phys peak:       %s\n",
                format_bytes(result.mem_phys_peak_bytes).c_str());
    }

    return result;
}

} // namespace qwen3_tts
