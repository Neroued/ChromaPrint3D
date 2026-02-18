#pragma once

/// \file export.h
/// \brief Visibility/export macros for shared library builds.

#ifdef CHROMAPRINT3D_STATIC
    #define CHROMAPRINT3D_API
#elif defined(CHROMAPRINT3D_BUILDING)
    #if defined(_MSC_VER)
        #define CHROMAPRINT3D_API __declspec(dllexport)
    #elif defined(__GNUC__) || defined(__clang__)
        #define CHROMAPRINT3D_API __attribute__((visibility("default")))
    #else
        #define CHROMAPRINT3D_API
    #endif
#else
    #if defined(_MSC_VER)
        #define CHROMAPRINT3D_API __declspec(dllimport)
    #else
        #define CHROMAPRINT3D_API
    #endif
#endif
