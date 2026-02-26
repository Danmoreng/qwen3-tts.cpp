cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

# Timing instrumentation option
option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_TTS_TIMING)
    add_compile_definitions(QWEN3_TTS_TIMING)
endif()

# CoreML code predictor bridge (macOS only)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

# Find threads
find_package(Threads REQUIRED)

# GGML location/build mode
set(QWEN3_TTS_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml" CACHE PATH "Path to GGML source directory")
option(QWEN3_TTS_EMBED_GGML "Build GGML as a CMake subdirectory instead of linking prebuilt libs" ON)
set(QWEN3_TTS_GGML_BUILD_DIR "${QWEN3_TTS_GGML_DIR}/build" CACHE PATH "Path to prebuilt GGML build directory (used when QWEN3_TTS_EMBED_GGML=OFF)")

option(QWEN3_TTS_CUDA "Enable CUDA backend via GGML" OFF)
if(QWEN3_TTS_CUDA)
    set(GGML_CUDA ON CACHE BOOL "Enable GGML CUDA backend" FORCE)
endif()

set(GGML_DIR "${QWEN3_TTS_GGML_DIR}")
set(GGML_BUILD_DIR "${QWEN3_TTS_GGML_BUILD_DIR}")

if(NOT EXISTS "${GGML_DIR}/include/ggml.h")
    message(FATAL_ERROR "GGML headers not found in '${GGML_DIR}'. Set -DQWEN3_TTS_GGML_DIR=<path-to-ggml>.")
endif()

if(QWEN3_TTS_EMBED_GGML)
    if(NOT EXISTS "${GGML_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "GGML CMakeLists.txt not found in '${GGML_DIR}'.")
    endif()
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${GGML_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")
endif()

function(qwen3_link_ggml target_name)
    if(QWEN3_TTS_EMBED_GGML)
        target_link_libraries(${target_name} PUBLIC ggml ggml-base)
        if(QWEN3_TTS_CUDA)
            target_link_libraries(${target_name} PUBLIC ggml-cuda)
        else()
            target_link_libraries(${target_name} PUBLIC ggml-cpu)
        endif()
    else()
        target_link_directories(${target_name} PUBLIC ${GGML_BUILD_DIR}/src)
        target_link_libraries(${target_name} PUBLIC ggml ggml-base)
        if(QWEN3_TTS_CUDA)
            target_link_libraries(${target_name} PUBLIC ggml-cuda)
        else()
            target_link_libraries(${target_name} PUBLIC ggml-cpu)
        endif()
    endif()
endfunction()

# Text tokenizer library
add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
    src/gguf_loader.cpp
)
target_include_directories(text_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
qwen3_link_ggml(text_tokenizer)
target_link_libraries(text_tokenizer PUBLIC Threads::Threads)

# TTS Transformer library (GGML-based + optional CoreML bridge)
set(TTS_TRANSFORMER_SOURCES
    src/gguf_loader.cpp
    src/tts_transformer.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor.mm)
    set_source_files_properties(src/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor_stub.cpp)
endif()

add_library(tts_transformer STATIC
    ${TTS_TRANSFORMER_SOURCES}
)
target_include_directories(tts_transformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
qwen3_link_ggml(tts_transformer)
target_link_libraries(tts_transformer PUBLIC Threads::Threads)
if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(tts_transformer PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()

# Audio tokenizer encoder library (GGML-based)
add_library(audio_tokenizer_encoder STATIC
    src/audio_tokenizer_encoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
qwen3_link_ggml(audio_tokenizer_encoder)
target_link_libraries(audio_tokenizer_encoder PUBLIC Threads::Threads)

# Audio tokenizer decoder library (GGML-based)
add_library(audio_tokenizer_decoder STATIC
    src/audio_tokenizer_decoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
qwen3_link_ggml(audio_tokenizer_decoder)
target_link_libraries(audio_tokenizer_decoder PUBLIC Threads::Threads)

# Qwen3 TTS library (full pipeline)
add_library(qwen3_tts STATIC
    src/qwen3_tts.cpp
)
target_include_directories(qwen3_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(qwen3_tts PUBLIC
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
    Threads::Threads
)
if(WIN32)
    target_link_libraries(qwen3_tts PUBLIC Psapi)
endif()

# CLI executable
add_executable(qwen3-tts-cli
    src/main.cpp
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3_tts
)

# Test executable for tokenizer
add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    text_tokenizer
    Threads::Threads
)

# Test executable for audio encoder
add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_tokenizer_encoder
    Threads::Threads
)

# Test executable for transformer
add_executable(test_transformer
    tests/test_transformer.cpp
)
target_link_libraries(test_transformer PRIVATE
    tts_transformer
    Threads::Threads
)

# Test executable for audio decoder
add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    audio_tokenizer_decoder
    Threads::Threads
)

# Install targets
install(TARGETS text_tokenizer tts_transformer audio_tokenizer_encoder audio_tokenizer_decoder qwen3_tts qwen3-tts-cli
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES 
    src/gguf_loader.h 
    src/text_tokenizer.h 
    src/tts_transformer.h 
    src/coreml_code_predictor.h
    src/audio_tokenizer_encoder.h 
    src/audio_tokenizer_decoder.h 
    src/qwen3_tts.h
    DESTINATION include
)

# CTest support
enable_testing()
add_test(NAME tokenizer_test
    COMMAND test_tokenizer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME encoder_test
    COMMAND test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf --audio clone.wav --reference reference/ref_audio_embedding.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME transformer_test
    COMMAND test_transformer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME decoder_test
    COMMAND test_decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
