cmake_minimum_required(VERSION 3.25)

project(ChromaPrint3D VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIB3MF_TESTS OFF CACHE BOOL "Disable lib3mf tests" FORCE)
add_subdirectory(3dparty/lib3mf)
if(TARGET lib3mf)
    if(MSVC)
        target_compile_options(lib3mf PRIVATE /W0)
    else()
        target_compile_options(lib3mf PRIVATE -w)
    endif()
endif()

add_subdirectory(core)

