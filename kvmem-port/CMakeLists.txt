cmake_minimum_required(VERSION 3.21)
project(kvmem_prismml_cuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# KVMem adapter lives in the kvmem-llama.cpp repo, out-of-tree.
set(KVMEM_ROOT "/home/nbuser/ai/llama/kvmem/kvmem-llama.cpp" CACHE PATH "kvmem_llamacpp checkout")

# this CMakeLists supports two layouts:
#   in-repo:      <checkout>/llama.cpp/kvmem-port/CMakeLists.txt   (llama.cpp root = parent dir)
#   standalone:   < somewhere >/kvmem-port/CMakeLists.txt with the llama.cpp checkout at < somewhere >/llama.cpp
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../src/llama-kvmem-factory.h)
    set(LLAMA_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
else()
    set(LLAMA_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp)
endif()
if (NOT EXISTS ${LLAMA_CPP_ROOT}/CMakeLists.txt)
    message(FATAL_ERROR "llama.cpp tree with kvmem patches not found at ${LLAMA_CPP_ROOT}")
endif()

# the prismml llama.cpp tree must already carry the kvmem patches (LLAMA_KVMEM option)
set(LLAMA_KVMEM     ON            CACHE BOOL "" FORCE)
set(LLAMA_KVMEM_ROOT "${KVMEM_ROOT}" CACHE PATH "" FORCE)

set(GGML_CUDA            ON    CACHE BOOL "" FORCE)
# RTX 5070 (Blackwell, sm_120); also feeds the kvmem stagein target's CUDA_ARCHITECTURES
set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "" FORCE)

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_ROOT} ${CMAKE_BINARY_DIR}/llama)

add_executable(llama-kvmem-cli
    ${KVMEM_ROOT}/tools/llama-kvmem-cli.cpp
    ${KVMEM_ROOT}/tools/kvmem-spec.cpp)
target_include_directories(llama-kvmem-cli PRIVATE
    ${KVMEM_ROOT}/src/adapter
    ${LLAMA_CPP_ROOT}/include
    ${LLAMA_CPP_ROOT}/common)
target_link_libraries(llama-kvmem-cli PRIVATE llama llama-common)
target_compile_features(llama-kvmem-cli PRIVATE cxx_std_17)

add_executable(llama-kvmem-server
    ${KVMEM_ROOT}/tools/llama-kvmem-server.cpp
    ${KVMEM_ROOT}/tools/kvmem-spec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kvmem-vision-prismml.cpp
    ${KVMEM_ROOT}/tools/kvmem-responses.cpp
    ${KVMEM_ROOT}/tools/kvmem-responses-stream.cpp
    ${LLAMA_CPP_ROOT}/tools/server/server-common.cpp
    ${LLAMA_CPP_ROOT}/tools/server/server-chat.cpp)
target_include_directories(llama-kvmem-server PRIVATE
    ${KVMEM_ROOT}/kvmem/include
    ${KVMEM_ROOT}/src/adapter
    ${KVMEM_ROOT}/tools
    ${LLAMA_CPP_ROOT}/include
    ${LLAMA_CPP_ROOT}/common
    ${LLAMA_CPP_ROOT}/tools/server
    ${LLAMA_CPP_ROOT}/tools/mtmd
    ${LLAMA_CPP_ROOT}/vendor
    ${LLAMA_CPP_ROOT}/vendor/cpp-httplib)
target_link_libraries(llama-kvmem-server PRIVATE llama llama-common cpp-httplib mtmd kvmem)
target_compile_features(llama-kvmem-server PRIVATE cxx_std_17)
target_compile_definitions(llama-kvmem-server PRIVATE KVMEM_ENABLE_NVME=$<BOOL:${KVMEM_ENABLE_NVME}>)
