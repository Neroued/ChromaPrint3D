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
    test_voxel.cpp
    test_mesh.cpp)

add_executable(chromaprint3d_tests ${TEST_SOURCES})
target_link_libraries(chromaprint3d_tests PRIVATE ChromaPrint3D::core GTest::gtest_main)

# Internal detail headers needed for recipe_convert test
target_include_directories(chromaprint3d_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/core/src)

include(GoogleTest)
gtest_discover_tests(chromaprint3d_tests)
