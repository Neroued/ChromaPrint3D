set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Threads REQUIRED)
find_package(Drogon CONFIG QUIET)

if(NOT Drogon_FOUND AND NOT TARGET Drogon::Drogon AND NOT TARGET drogon)
    include(FetchContent)
    FetchContent_Declare(
        jsoncpp
        GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
        GIT_TAG 1.9.6
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(jsoncpp)
    set(JSONCPP_INCLUDE_DIRS "${jsoncpp_SOURCE_DIR}/include" CACHE STRING "jsoncpp include path" FORCE)
    set(JSONCPP_LIBRARIES jsoncpp_static CACHE STRING "jsoncpp library target" FORCE)

    # Use a stable tag to keep reproducible builds.
    FetchContent_Declare(
        drogon
        GIT_REPOSITORY https://github.com/drogonframework/drogon.git
        GIT_TAG v1.9.8
        GIT_SHALLOW TRUE
        GIT_SUBMODULES_RECURSE TRUE)
    set(BUILD_CTL OFF CACHE BOOL "Disable drogon ctl tool" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Disable drogon examples" FORCE)
    set(BUILD_ORM OFF CACHE BOOL "Disable drogon orm helper" FORCE)
    set(BUILD_YAML_CONFIG OFF CACHE BOOL "Disable drogon yaml support" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable drogon/trantor tests" FORCE)
    set(DROGON_BUILD_TESTING OFF CACHE BOOL "Disable drogon tests" FORCE)
    set(TRANTOR_BUILD_TESTING OFF CACHE BOOL "Disable trantor tests" FORCE)
    FetchContent_MakeAvailable(drogon)

    # Treat Drogon/Trantor as third-party: keep their warnings quiet.
    if(TARGET drogon)
        if(MSVC)
            target_compile_options(drogon PRIVATE /W0)
        else()
            target_compile_options(drogon PRIVATE -w)
        endif()
    endif()
    if(TARGET trantor)
        if(MSVC)
            target_compile_options(trantor PRIVATE /W0)
        else()
            target_compile_options(trantor PRIVATE -w)
        endif()
    endif()
endif()

set(CHROMAPRINT3D_BACKEND_LIBRARY_SOURCES
    src/config/server_config.cpp
    src/infrastructure/session_store.cpp
    src/infrastructure/task_runtime.cpp
    src/infrastructure/spillable_artifact.cpp
    src/infrastructure/data_repository.cpp
    src/infrastructure/board_runtime_cache.cpp
    src/application/server_facade.cpp
    src/presentation/backend_runtime.cpp
    src/presentation/cors_advice.cpp
    src/presentation/api_v1_controller.cpp)

if(TARGET Drogon::Drogon)
    set(CHROMAPRINT3D_DROGON_TARGET Drogon::Drogon)
elseif(TARGET drogon)
    set(CHROMAPRINT3D_DROGON_TARGET drogon)
else()
    message(FATAL_ERROR "Drogon target not found after discovery/fetch")
endif()

add_library(chromaprint3d_backend STATIC ${CHROMAPRINT3D_BACKEND_LIBRARY_SOURCES})

target_include_directories(chromaprint3d_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(chromaprint3d_backend PUBLIC
    ChromaPrint3D::core
    Threads::Threads
    ${CHROMAPRINT3D_DROGON_TARGET})

add_executable(chromaprint3d_server server_main.cpp)

target_link_libraries(chromaprint3d_server PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,chromaprint3d_backend>)
