cmake_minimum_required(VERSION 3.20)
project(noether LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NOETHER_USE_CUDA "Build CUDA kernels (requires nvcc)" ON)
option(NOETHER_USE_OPENMP "Enable OpenMP CPU parallelism" ON)
option(NOETHER_BUILD_TESTS "Build CTest suite" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── CUDA detection ────────────────────────────────────────────────────────────
set(NOETHER_HAS_CUDA FALSE)
if(NOETHER_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
        set(NOETHER_HAS_CUDA TRUE)
        message(STATUS "CUDA: ${CMAKE_CUDA_COMPILER} (sm_75/80/86/89)")
    else()
        message(WARNING "NOETHER_USE_CUDA=ON but nvcc not found — building CPU-only")
    endif()
endif()

# ── Engine binary ─────────────────────────────────────────────────────────────
set(NOETHER_SOURCES main.cpp)
if(NOETHER_HAS_CUDA)
    list(APPEND NOETHER_SOURCES core/cuda/kernels.cu)
endif()

add_executable(noether ${NOETHER_SOURCES})
target_include_directories(noether PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(noether PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)

if(NOETHER_HAS_CUDA)
    target_compile_definitions(noether PRIVATE NOETHER_USE_CUDA)
endif()

if(NOETHER_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(noether PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(noether PRIVATE NOETHER_USE_OPENMP)
        message(STATUS "OpenMP: enabled")
    endif()
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(NOETHER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
