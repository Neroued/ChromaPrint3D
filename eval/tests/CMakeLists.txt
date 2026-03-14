add_executable(eval_tests test_vectorize_eval.cpp)
target_link_libraries(eval_tests PRIVATE ChromaPrint3D::eval GTest::gtest_main)
target_include_directories(eval_tests PRIVATE ${CMAKE_SOURCE_DIR}/eval/src)

include(GoogleTest)
gtest_discover_tests(eval_tests)
