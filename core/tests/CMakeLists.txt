include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(googletest)

set(TEST_SOURCES
    test_vec3.cpp
    test_color.cpp
    test_kdtree.cpp
    test_color_db.cpp
    test_recipe_convert.cpp
    test_raster_match.cpp
    test_vector_match.cpp
    test_layer_previews.cpp
    test_voxel.cpp
    test_mesh.cpp
    test_vector_mesh.cpp
    test_3mf_writer.cpp
    test_matting.cpp
    test_slicer_preset.cpp
    test_memory_utils.cpp)

add_executable(chromaprint3d_tests ${TEST_SOURCES})
target_link_libraries(chromaprint3d_tests PRIVATE ChromaPrint3D::core GTest::gtest_main)

find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    target_link_libraries(chromaprint3d_tests PRIVATE ZLIB::ZLIB)
    target_compile_definitions(chromaprint3d_tests PRIVATE CHROMAPRINT3D_TEST_HAS_ZLIB=1)
endif()

# Internal detail headers needed for recipe_convert test
target_include_directories(chromaprint3d_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/core/src)

include(GoogleTest)
gtest_discover_tests(chromaprint3d_tests)
