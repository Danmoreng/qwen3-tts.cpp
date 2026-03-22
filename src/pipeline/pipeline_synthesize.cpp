#include "qwen3_tts.h"
#include "pipeline/pipeline_internal.h"

#include <algorithm>
#include <cstdio>

namespace qwen3_tts {
using pipeline_internal::format_bytes;
using pipeline_internal::get_process_memory_snapshot;
using pipeline_internal::get_time_ms;
using pipeline_internal::log_memory_usage;
using pipeline_internal::ops;
using pipeline_internal::process_memory_snapshot;
using pipeline_internal::resample_linear;

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
        return ops::synthesize_internal(*this, text, speaker_embedding.data(), nullptr, params, result);
    }

    if (transformer_.get_config().tts_model_type == "custom_voice") {
        result.error_msg = "CustomVoice model requires --speaker, --reference, or --speaker-embedding";
        return result;
    }

    return ops::synthesize_internal(*this, text, nullptr, nullptr, params, result);
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

    if (!encoder_loaded_) {
        if (tts_model_path_.empty()) {
            result.error_msg = "Internal error: missing TTS model path for lazy encoder load";
            return result;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(tts_model_path_)) {
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

    int64_t t_encode_start = get_time_ms();
    std::vector<float> speaker_embedding;

    if (!audio_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
        result.error_msg = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return result;
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

    return ops::synthesize_internal(*this, text, speaker_embedding.data(), nullptr, params, result);
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
    return ops::synthesize_internal(*this, text, speaker_embedding.data(), nullptr, params, result);
}

tts_result Qwen3TTS::synthesize_with_voice_clone_prompt(const std::string & text,
                                                        const voice_clone_prompt_asset & asset,
                                                        const tts_params & params) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    voice_clone_prompt_validation validation;
    if (!validate_voice_clone_prompt(asset, &validation)) {
        result.error_msg = validation.error_msg;
        return result;
    }

    if (params.print_progress) {
        fprintf(stderr, "Using voice clone prompt: mode=%s, speaker_embedding=%zu floats, ref_frames=%d\n",
                asset.prompt_mode == voice_clone_prompt_mode::reference_aware ? "reference_aware" : "audio_only",
                asset.speaker_embedding.size(),
                asset.reference_frames);
    }

    result.t_encode_ms = 0;
    return ops::synthesize_internal(*this, text, asset.speaker_embedding.data(), &asset, params, result);
}

bool Qwen3TTS::extract_speaker_embedding(const std::string & reference_audio,
                                         std::vector<float> & speaker_embedding,
                                         int64_t * encode_time_ms) {
    if (!models_loaded_) {
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
        if (tts_model_path_.empty()) {
            error_msg_ = "Internal error: missing TTS model path for lazy encoder load";
            return false;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(tts_model_path_)) {
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

    const int expected_dim = transformer_.get_config().hidden_size;
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

tts_result pipeline_internal::ops::synthesize_internal(Qwen3TTS & self,
                                                       const std::string & text,
                                                       const float * speaker_embedding,
                                                       const voice_clone_prompt_asset * prompt_asset,
                                                       const tts_params & params,
                                                       tts_result & result) {
    int64_t t_total_start = get_time_ms();
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

    if (params.print_progress) {
        fprintf(stderr, "Text tokenized: %zu tokens\n", text_tokens.size());
        if (!instruct_tokens.empty()) {
            fprintf(stderr, "Instruction tokenized: %zu tokens\n", instruct_tokens.size());
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

    std::vector<int32_t> speech_codes;
    if (!self.transformer_.generate(text_tokens.data(), (int32_t) text_tokens.size(),
                                    speaker_embedding, params.max_audio_tokens, speech_codes,
                                    params.language_id, params.repetition_penalty,
                                    params.temperature, params.top_k,
                                    instruct_tokens.empty() ? nullptr : instruct_tokens.data(),
                                    (int32_t) instruct_tokens.size())) {
        result.error_msg = "Failed to generate speech codes: " + self.transformer_.get_error();
        return result;
    }
    result.t_generate_ms = get_time_ms() - t_generate_start;
    sample_memory("synth/after-generate");

    int n_codebooks = self.transformer_.get_config().n_codebooks;
    int n_frames = (int) speech_codes.size() / n_codebooks;

    if (params.print_progress) {
        fprintf(stderr, "Speech codes generated: %d frames x %d codebooks\n", n_frames, n_codebooks);
    }

    if (n_frames == 0) {
        result.error_msg = "No speech codes generated";
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

    std::vector<int32_t> decode_codes = speech_codes;
    int32_t decode_frames = n_frames;
    int32_t trim_ref_frames = 0;
    if (prompt_asset &&
        prompt_asset->prompt_mode == voice_clone_prompt_mode::reference_aware &&
        !prompt_asset->reference_codes.empty()) {
        trim_ref_frames = prompt_asset->reference_frames;
        decode_codes.reserve(prompt_asset->reference_codes.size() + speech_codes.size());
        decode_codes.insert(decode_codes.begin(),
                            prompt_asset->reference_codes.begin(),
                            prompt_asset->reference_codes.end());
        decode_frames += trim_ref_frames;
        if (params.print_progress) {
            fprintf(stderr, "Prepending %d reference frames to vocoder decode\n", trim_ref_frames);
        }
    }

    if (!self.audio_decoder_.decode(decode_codes.data(), decode_frames, result.audio)) {
        result.error_msg = "Failed to decode speech codes: " + self.audio_decoder_.get_error();
        return result;
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    sample_memory("synth/after-decode");

    if (trim_ref_frames > 0 && !result.audio.empty()) {
        const size_t trim_samples = std::min(
            result.audio.size(),
            (size_t) ((double) trim_ref_frames * result.audio.size() / std::max(decode_frames, 1)));
        result.audio.erase(result.audio.begin(), result.audio.begin() + (std::ptrdiff_t) trim_samples);
    }

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
