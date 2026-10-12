#pragma once

#include <cstdio>
#include <filesystem>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace qwen3_tts {

// Paths arrive as UTF-8 (from the CLI's UTF-8 argv and from the C bridge). On Windows the
// narrow fopen / std::filesystem constructors read them in the ANSI code page instead, so a
// Japanese or Chinese directory name opens the wrong file. Convert to wide first.
#ifdef _WIN32
inline std::wstring utf8_to_wide(const std::string & utf8) {
    if (utf8.empty()) {
        return std::wstring();
    }
    const int length = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), (int) utf8.size(), nullptr, 0);
    std::wstring wide(length > 0 ? (size_t) length : 0, L'\0');
    if (length > 0) {
        MultiByteToWideChar(CP_UTF8, 0, utf8.data(), (int) utf8.size(), &wide[0], length);
    }
    return wide;
}

inline std::filesystem::path utf8_fs_path(const std::string & utf8) {
    return std::filesystem::path(utf8_to_wide(utf8));
}

inline FILE * utf8_fopen(const std::string & utf8, const char * mode) {
    return _wfopen(utf8_to_wide(utf8).c_str(), utf8_to_wide(mode).c_str());
}
#else
inline std::filesystem::path utf8_fs_path(const std::string & utf8) {
    return std::filesystem::path(utf8);
}

inline FILE * utf8_fopen(const std::string & utf8, const char * mode) {
    return fopen(utf8.c_str(), mode);
}
#endif

} // namespace qwen3_tts
