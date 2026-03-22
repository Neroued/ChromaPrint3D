cmake_minimum_required(VERSION 3.25)

project(ChromaPrint3D VERSION 1.2.9 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional: use ccache to speed up recompilation (local and CI)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C++ compiler launcher")
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

# Build options
option(CHROMAPRINT3D_BUILD_APPS "Build application executables" ON)
option(CHROMAPRINT3D_BUILD_TESTS "Build unit tests" OFF)
option(CHROMAPRINT3D_3MF_ZLIB_REQUIRED "Require zlib for 3MF compression backend" OFF)
option(CHROMAPRINT3D_3MF_FORCE_STORE "Force 3MF writer to use ZIP store-only mode" OFF)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs)
message(STATUS "Found OpenCV ${OpenCV_VERSION}: ${OpenCV_INSTALL_PATH}")

set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "Disable spdlog examples" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Disable spdlog tests" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "Disable spdlog benchmarks" FORCE)
add_subdirectory(3dparty/spdlog)

set(CLIPPER2_UTILS OFF CACHE BOOL "Disable Clipper2 utilities" FORCE)
set(CLIPPER2_EXAMPLES OFF CACHE BOOL "Disable Clipper2 examples" FORCE)
set(CLIPPER2_TESTS OFF CACHE BOOL "Disable Clipper2 tests" FORCE)
set(CLIPPER2_USINGZ "OFF" CACHE STRING "Disable Clipper2Z" FORCE)
add_subdirectory(3dparty/clipper2/CPP)
if(TARGET Clipper2)
    if(MSVC)
        target_compile_options(Clipper2 PRIVATE /W0)
    else()
        target_compile_options(Clipper2 PRIVATE -w)
    endif()
endif()

option(CHROMAPRINT3D_BUILD_INFER "Build inference engine module" ON)
if(CHROMAPRINT3D_BUILD_INFER)
    add_subdirectory(infer)
endif()

add_subdirectory(core)

if(CHROMAPRINT3D_BUILD_APPS)
    add_subdirectory(apps)
    add_subdirectory(web/backend)
endif()

if(CHROMAPRINT3D_BUILD_TESTS)
    enable_testing()
    add_subdirectory(core/tests)
    if(CHROMAPRINT3D_BUILD_APPS)
        add_subdirectory(web/backend/tests)
    endif()
    if(CHROMAPRINT3D_BUILD_INFER)
        add_subdirectory(infer/tests)
    endif()
endif()

