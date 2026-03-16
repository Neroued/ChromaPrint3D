# ── ChromaPrint3D core library ────────────────────────────────────────────────

add_library(chromaprint3d STATIC)
add_library(ChromaPrint3D::core ALIAS chromaprint3d)

set_target_properties(chromaprint3d PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME chromaprint3d)

target_compile_features(chromaprint3d PUBLIC cxx_std_20)
target_compile_definitions(chromaprint3d PUBLIC CHROMAPRINT3D_STATIC)

# Generate version.h from template using PROJECT_VERSION
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/chromaprint3d/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/chromaprint3d/version.h
    @ONLY)

# Public headers (build and install paths)
target_include_directories(chromaprint3d PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Internal detail headers (not installed)
target_include_directories(chromaprint3d PRIVATE src)

# ── Third-party include paths (build-only) ───────────────────────────────────

target_include_directories(chromaprint3d SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/3dparty>)
target_include_directories(chromaprint3d SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/3dparty/clipper2/CPP/Clipper2Lib/include>)

# ── Sources ──────────────────────────────────────────────────────────────────

target_sources(chromaprint3d PRIVATE
    src/common/common.cpp
    src/common/image_io.cpp
    src/imgproc/raster_proc.cpp
    src/imgproc/vector_proc.cpp
    src/imgproc/bezier.cpp
    src/imgproc/occlusion.cpp
    src/imgproc/gradient.cpp
    src/imgproc/vector_preview.cpp
    src/imgproc/shape_width_analyzer.cpp
    src/color_db/color_db.cpp
    src/match/match.cpp
    src/match/slic_segmenter.cpp
    src/match/dither.cpp
    src/match/print_profile.cpp
    src/match/recipe_map.cpp
    src/match/model_package.cpp
    src/match/recipe_convert.cpp
    src/match/candidate_select.cpp
    src/match/vector_match.cpp
    src/calib/calib.cpp
    src/calib/loc.cpp
    src/geo/geo.cpp
    src/geo/export_3mf.cpp
    src/geo/three_mf_writer.cpp
    src/geo/three_mf_extensions.cpp
    src/geo/three_mf_opc.cpp
    src/geo/three_mf_zip.cpp
    src/geo/bambu_metadata.cpp
    src/vecgeo/vector_mesh_builder.cpp
    src/vecgeo/triangulate.cpp
    src/pipeline/pipeline.cpp
    src/pipeline/vector_pipeline.cpp
    src/encoding/encoding.cpp
    src/logging/logging.cpp
    src/matting/matting.cpp
    src/matting/matting_postprocess.cpp
    src/detail/icc_utils.cpp)

# ── Optional: infer module for DL matting ─────────────────────────────────────

if(TARGET ChromaPrint3D::infer)
    target_sources(chromaprint3d PRIVATE src/matting/dl_matting.cpp)
    target_link_libraries(chromaprint3d PUBLIC ChromaPrint3D::infer)
    target_compile_definitions(chromaprint3d PUBLIC CHROMAPRINT3D_HAS_INFER=1)
endif()

# ── Dependencies ─────────────────────────────────────────────────────────────

target_link_libraries(chromaprint3d PUBLIC opencv_core opencv_imgproc opencv_imgcodecs)
target_link_libraries(chromaprint3d PUBLIC Clipper2)
target_link_libraries(chromaprint3d PUBLIC spdlog::spdlog_header_only)

find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    target_link_libraries(chromaprint3d PUBLIC ZLIB::ZLIB)
    target_compile_definitions(chromaprint3d PUBLIC CHROMAPRINT3D_HAS_ZLIB=1)
    message(STATUS "Found zlib: ${ZLIB_VERSION_STRING}")
elseif(CHROMAPRINT3D_3MF_ZLIB_REQUIRED)
    message(FATAL_ERROR "CHROMAPRINT3D_3MF_ZLIB_REQUIRED=ON but zlib was not found")
else()
    message(WARNING "zlib not found — 3MF export falls back to ZIP store-only mode")
endif()

if(CHROMAPRINT3D_3MF_FORCE_STORE)
    target_compile_definitions(chromaprint3d PUBLIC CHROMAPRINT3D_3MF_FORCE_STORE=1)
    message(STATUS "3MF ZIP compression is disabled by CHROMAPRINT3D_3MF_FORCE_STORE")
endif()

find_library(LCMS2_LIBRARY lcms2)
find_path(LCMS2_INCLUDE_DIR lcms2.h)
if(LCMS2_LIBRARY AND LCMS2_INCLUDE_DIR)
    find_package(JPEG QUIET)
    if(NOT JPEG_FOUND)
        message(WARNING "lcms2 found but JPEG not found — ICC color management disabled")
    else()
    target_include_directories(chromaprint3d PRIVATE ${LCMS2_INCLUDE_DIR})
    target_link_libraries(chromaprint3d PRIVATE ${LCMS2_LIBRARY} JPEG::JPEG)
    target_compile_definitions(chromaprint3d PRIVATE CHROMAPRINT3D_HAS_LCMS2=1)
    message(STATUS "Found lcms2: ${LCMS2_LIBRARY}, jpeg: ${JPEG_LIBRARY}")
    endif()
else()
    message(WARNING "lcms2 not found — ICC color management disabled")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(chromaprint3d PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── Compiler flags (per-build-type, no hardcoded -O3) ────────────────────────

target_compile_options(chromaprint3d PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:RelWithDebInfo>>:/O2>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:RelWithDebInfo>>:/Zi>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:RelWithDebInfo>>:-O2>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<CONFIG:RelWithDebInfo>>:-g>)

# ── Install rules ────────────────────────────────────────────────────────────

include(GNUInstallDirs)

install(TARGETS chromaprint3d
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/chromaprint3d
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/chromaprint3d/version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/chromaprint3d)
