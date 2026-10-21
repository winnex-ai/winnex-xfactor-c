cmake_minimum_required(VERSION 3.20)
project(winnex_xfactor_c VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# winnex-xfactor-c — native Ψ (latent coupling + phase activation) engine.
#
# Reuses the winnex-nano native sources (safetensors, tensor_ops, x_factor,
# spectral_tokenizer) WITHOUT duplicating them, following the project pattern
# (winnex-nano reuses winnex-madhava the same way).
# ============================================================================

if(NOT DEFINED NANO_ROOT)
    set(NANO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../winnex-nano")
endif()

if(NOT EXISTS "${NANO_ROOT}/CMakeLists.txt")
    message(STATUS "winnex-xfactor-c: winnex-nano sources not found locally — fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(
        winnex_nano_src
        GIT_REPOSITORY https://github.com/winnex-ai/winnex-nano.git
        # v0.1.11 was stale (PyPI ships 0.1.24) and its forward.cpp still had
        # the RoPE/bias/softmax defects fixed in the 2026-10-21 session. Pinned
        # to a tag that contains those fixes.
        GIT_TAG v0.1.24
    )
    FetchContent_Populate(winnex_nano_src)
    set(NANO_ROOT "${winnex_nano_src_SOURCE_DIR}")
endif()
message(STATUS "winnex-xfactor-c: reusing winnex-nano at ${NANO_ROOT}")

# --- Reused winnex-nano core (referenced, not copied) -----------------------
add_library(nano_reused STATIC
    ${NANO_ROOT}/src/safetensors.cpp
    ${NANO_ROOT}/src/tensor_ops.cpp
    ${NANO_ROOT}/src/x_factor.cpp
    ${NANO_ROOT}/src/spectral_tokenizer.cpp
)
target_include_directories(nano_reused PUBLIC ${NANO_ROOT}/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    target_compile_options(nano_reused PRIVATE -O3 -mavx2 -mfma -funroll-loops)
endif()
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nano_reused PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- winnex-xfactor-c core ---------------------------------------------------
add_library(winnex_xfactor_c
    src/derive_weights.cpp
    src/latent_coupling.cpp
    src/attention_forward.cpp
    src/psi_forward.cpp
)
target_include_directories(winnex_xfactor_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# OpenCL headers — needed only for the vendored CL/cl.h types; the loader is
# resolved at RUNTIME via dlopen, so there is no link-time OpenCL dependency.
#
# STANDALONE BUILD (2026-10-21). This used to point at
# `${NANO_ROOT}/../winnex-madhava/vendor`, a path that only exists when the
# sibling winnex-madhava checkout sits next to winnex-nano. In a clean clone of
# this repository — which is exactly what CI builds — that path does not exist,
# and the compile dies on `#include <CL/cl.h>` (reproduced: fatal error,
# CL/cl.h: No such file or directory).
#
# Resolution order, most self-contained first:
#   1. vendored copy next to this repo (developer working in the monorepo)
#   2. fetch winnex-madhava (it vendors CL/cl.h) — same mechanism the
#      winnex-nano build already uses to obtain the shared kernels
#   3. system OpenCL headers, if the distro ships them
set(XFC_OPENCL_INCLUDE "")
if(EXISTS "${NANO_ROOT}/../winnex-madhava/vendor/CL/cl.h")
    set(XFC_OPENCL_INCLUDE "${NANO_ROOT}/../winnex-madhava/vendor")
    message(STATUS "winnex-xfactor-c: OpenCL headers from local winnex-madhava")
else()
    message(STATUS "winnex-xfactor-c: no local winnex-madhava — fetching it for the OpenCL headers")
    include(FetchContent)
    FetchContent_Declare(
        winnex_madhava_hdr
        GIT_REPOSITORY https://github.com/winnex-ai/winnex-madhava.git
        GIT_TAG v1.9.17
        GIT_SHALLOW TRUE
    )
    FetchContent_Populate(winnex_madhava_hdr)
    if(EXISTS "${winnex_madhava_hdr_SOURCE_DIR}/vendor/CL/cl.h")
        set(XFC_OPENCL_INCLUDE "${winnex_madhava_hdr_SOURCE_DIR}/vendor")
        message(STATUS "winnex-xfactor-c: OpenCL headers from fetched winnex-madhava")
    else()
        find_path(OPENCL_INCLUDE_DIR CL/cl.h)
        if(OPENCL_INCLUDE_DIR)
            set(XFC_OPENCL_INCLUDE "${OPENCL_INCLUDE_DIR}")
            message(STATUS "winnex-xfactor-c: OpenCL headers from the system")
        else()
            message(FATAL_ERROR
                "winnex-xfactor-c: CL/cl.h not found. Install opencl-headers "
                "(Debian/Ubuntu) or OpenCL-Headers (vcpkg), or place a "
                "winnex-madhava checkout next to this repository.")
        endif()
    endif()
endif()
target_include_directories(winnex_xfactor_c PRIVATE ${XFC_OPENCL_INCLUDE})
target_link_libraries(winnex_xfactor_c PUBLIC nano_reused)
target_link_libraries(winnex_xfactor_c PRIVATE ${CMAKE_DL_LIBS})

# --- Python bindings (pybind11) ---------------------------------------------
option(WINNEX_XFACTOR_C_BUILD_PYTHON "Build the pybind11 Python module" ON)
if(WINNEX_XFACTOR_C_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_winnex_xfactor_c bindings/py_module.cpp)
    target_link_libraries(_winnex_xfactor_c PRIVATE winnex_xfactor_c)
    if(DEFINED SKBUILD_PLATLIB_DIR)
        install(TARGETS _winnex_xfactor_c DESTINATION ${SKBUILD_PLATLIB_DIR})
    endif()
endif()

# --- Tests ------------------------------------------------------------------
option(WINNEX_XFACTOR_C_BUILD_TESTS "Build the C++ tests" ON)
if(WINNEX_XFACTOR_C_BUILD_TESTS)
    enable_testing()
    add_executable(test_derive_preserves tests/test_derive_preserves.cpp)
    target_link_libraries(test_derive_preserves PRIVATE winnex_xfactor_c)
    add_test(NAME derive_preserves COMMAND test_derive_preserves)

    add_executable(test_latent_coupling tests/test_latent_coupling.cpp)
    target_link_libraries(test_latent_coupling PRIVATE winnex_xfactor_c)
    add_test(NAME latent_coupling COMMAND test_latent_coupling)

    add_executable(test_attention_forward tests/test_attention_forward.cpp)
    target_link_libraries(test_attention_forward PRIVATE winnex_xfactor_c)
    add_test(NAME attention_forward COMMAND test_attention_forward)

    add_executable(test_spectral tests/test_spectral.cpp)
    target_link_libraries(test_spectral PRIVATE winnex_xfactor_c)
    add_test(NAME spectral COMMAND test_spectral)

    add_executable(test_position tests/test_position.cpp)
    target_link_libraries(test_position PRIVATE winnex_xfactor_c)
    add_test(NAME position COMMAND test_position)

    add_executable(test_psi_forward tests/test_psi_forward.cpp)
    target_link_libraries(test_psi_forward PRIVATE winnex_xfactor_c)
    # Manual run (requires a model dir).
endif()
