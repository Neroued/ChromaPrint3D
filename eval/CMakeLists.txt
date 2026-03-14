add_library(chromaprint3d_eval STATIC
    src/svg_geometry.cpp
    src/svg_rasterizer.cpp
    src/pixel_metrics.cpp
    src/edge_metrics.cpp
    src/path_metrics.cpp
    src/benchmark.cpp
    src/vectorize_eval.cpp)
add_library(ChromaPrint3D::eval ALIAS chromaprint3d_eval)

target_include_directories(chromaprint3d_eval
    PUBLIC  include
    PRIVATE src)

# NanoSVG headers and OpenCV propagate via ChromaPrint3D::core PUBLIC deps.
target_link_libraries(chromaprint3d_eval
    PUBLIC  ChromaPrint3D::core
    PRIVATE spdlog::spdlog)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(chromaprint3d_eval PRIVATE OpenMP::OpenMP_CXX)
endif()
