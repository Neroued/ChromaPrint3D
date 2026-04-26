/// \file core/src/color/conversions.cpp
/// \brief Out-of-line conversion helpers — sRGB byte ↔ linear LUT primitives.
///
/// The 256-entry float LUT is the single source of truth for sRGB byte
/// decoding across all hot paths (`SrgbU8::ToRgb`, `BgrToLab`, etc.). The
/// LUT is initialized at static-storage time; thread-safe construction is
/// guaranteed by C++ static-init rules (the local-static `kLut` array is
/// only written during the `Init()` call which the compiler runs once).

#include "chromaprint3d/color/conversions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace ChromaPrint3D {

namespace {

/// 256-entry sRGB-byte → linear-float LUT. Initialised once at first use.
struct SrgbToLinearLutInit {
    std::array<float, 256> table{};

    SrgbToLinearLutInit() {
        for (int i = 0; i < 256; ++i) { table[i] = SrgbToLinear(static_cast<float>(i) / 255.0f); }
    }
};

const SrgbToLinearLutInit& Lut() {
    static const SrgbToLinearLutInit kLut;
    return kLut;
}

} // namespace

float SrgbToLinearByte(std::uint8_t v) { return Lut().table[v]; }

std::uint8_t LinearToSrgbByte(float linear) {
    if (!(linear > 0.0f)) return 0; // catches NaN / negative
    if (linear >= 1.0f) return 255;
    const float srgb = LinearToSrgb(linear);
    int rounded      = static_cast<int>(std::round(srgb * 255.0f));
    rounded          = std::max(0, std::min(255, rounded));
    return static_cast<std::uint8_t>(rounded);
}

} // namespace ChromaPrint3D
