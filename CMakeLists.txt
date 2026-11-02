cmake_minimum_required(VERSION 3.16)
project(locality-aware-spmv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "CMAKE_BUILD_TYPE not set, defaulting to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Performance numbers are only comparable in Release.
# LTO stays off: the timed loop must not see the SpMV body, or the
# compiler can delete repeated iterations.
function(spmv_warnings target)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -fno-lto)
endfunction()

function(spmv_optimize target)
  target_compile_options(${target} PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:RelWithDebInfo>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-march=native>
    $<$<CONFIG:RelWithDebInfo>:-g>
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:Debug>:-g>
  )
endfunction()

add_library(spmv_core
  src/mm_io.cpp
  src/spmv.cpp
  src/dense.cpp
  src/stream.cpp
  src/peak.cpp
  src/machine.cpp
  src/traffic.cpp
  src/check.cpp
  src/benchmark.cpp
)
target_include_directories(spmv_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
spmv_warnings(spmv_core)
spmv_optimize(spmv_core)

add_executable(spmv src/main.cpp)
target_link_libraries(spmv PRIVATE spmv_core)
spmv_warnings(spmv)
target_link_options(spmv PRIVATE -fno-lto)

enable_testing()
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE spmv_core)
spmv_warnings(test_correctness)
target_link_options(test_correctness PRIVATE -fno-lto)
configure_file(tests/source_dir.hpp.in ${CMAKE_BINARY_DIR}/generated/source_dir.hpp @ONLY)
target_include_directories(test_correctness PRIVATE ${CMAKE_BINARY_DIR}/generated)
add_test(NAME correctness COMMAND test_correctness)

set_target_properties(spmv_core spmv test_correctness PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
