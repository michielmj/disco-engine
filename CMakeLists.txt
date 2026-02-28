# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(disco LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── macOS minimum deployment target ───────────────────────────────────────────
# std::variant (used in Headers) requires macOS 10.14+; 10.15 is a safe baseline.
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum macOS deployment version")
endif()

# ── Shared toolchain setup ────────────────────────────────────────────────────
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _numpy_missing
)
if(_numpy_missing)
    message(FATAL_ERROR "NumPy not found — add numpy to build-system.requires")
endif()
message(STATUS "NumPy include dir: ${NUMPY_INCLUDE_DIR}")

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()

find_program(CYTHON_EXECUTABLE NAMES cython cython3 REQUIRED)

find_package(OpenMP COMPONENTS C)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found — Cython prange targets will be parallel")
else()
    message(WARNING "OpenMP not found — Cython targets will be single-threaded")
endif()

# ── Shared macros ─────────────────────────────────────────────────────────────
include(cmake/extension_helpers.cmake)

# ── Subpackages ───────────────────────────────────────────────────────────────
add_subdirectory(src/disco/event_queue)
# add_subdirectory(src/toolbox/future_package)   ← one line to add a new one
