set(TEST_SOURCES
    test_server_facade.cpp
    test_task_runtime.cpp)

add_executable(chromaprint3d_backend_tests ${TEST_SOURCES})

target_link_libraries(chromaprint3d_backend_tests PRIVATE
    chromaprint3d_backend
    GTest::gtest_main)

target_compile_definitions(chromaprint3d_backend_tests PRIVATE
    CHROMAPRINT3D_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

include(GoogleTest)
gtest_discover_tests(chromaprint3d_backend_tests)
