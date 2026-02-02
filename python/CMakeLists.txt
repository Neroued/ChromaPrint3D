set(PYBIND_SOURCES
    bindings/module.cpp
    bindings/bind_common.cpp
    bindings/bind_color_db.cpp
    bindings/bind_imgproc.cpp
    bindings/bind_match.cpp
    bindings/bind_geo.cpp
    bindings/bind_calib.cpp
)

pybind11_add_module(ChromaPrint3D ${PYBIND_SOURCES})

target_compile_features(ChromaPrint3D PRIVATE cxx_std_20)
target_link_libraries(ChromaPrint3D PRIVATE core)
target_include_directories(ChromaPrint3D PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings
)

if (SKBUILD)
    install(TARGETS ChromaPrint3D
        LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}
    )
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/ChromaPrint3D.pyi
        DESTINATION ${SKBUILD_PLATLIB_DIR}
    )
endif()
