cmake_minimum_required(VERSION 3.22)
project(smollm_jni)

set(CMAKE_CXX_STANDARD 17)

# Disable extras we don't need — keeps the build small and fast
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp llama_build)

add_library(smollm_jni SHARED smollm_jni.cpp)

target_link_libraries(smollm_jni
    llama
    common
    log
    android
)

target_include_directories(smollm_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/common
)
