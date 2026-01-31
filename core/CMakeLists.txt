add_library(core STATIC)

target_compile_features(core PUBLIC cxx_std_20)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

target_include_directories(core PUBLIC include)

target_sources(core PRIVATE
    src/imgproc/imgproc.cpp)

target_include_directories(core SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(core PUBLIC ${OpenCV_LIBS})

