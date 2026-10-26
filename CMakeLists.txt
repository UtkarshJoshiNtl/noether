cmake_minimum_required(VERSION 3.20)
project(noether LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NOETHER_BUILD_TESTS "Build CTest suite" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(noether_core INTERFACE)
target_include_directories(noether_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(noether_core INTERFACE cxx_std_20)

if(NOETHER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
