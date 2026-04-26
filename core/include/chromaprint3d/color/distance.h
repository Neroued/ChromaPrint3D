#pragma once

/// \file color/distance.h
/// \brief Color-distance metrics: ΔE76 (Lab), Euclidean RGB squared,
///        SrgbU8 byte-distance squared, BBS-parity HSV distance, and
///        Lab a*b* polar hue-angle difference.
///
/// `DeltaE2000` is intentionally **not** in this PR — see plan §3.4.

#include "chromaprint3d/color/conversions.h"
#include "chromaprint3d/color/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ChromaPrint3D {

/// CIE Delta-E 76 (Euclidean L*a*b* distance).
inline float DeltaE76(const Lab& a, const Lab& b) { return Lab::DeltaE76(a, b); }

/// Squared Euclidean distance in linear sRGB.
inline float RgbDistanceSq(const Rgb& a, const Rgb& b) {
    const float dr = a.r() - b.r();
    const float dg = a.g() - b.g();
    const float db = a.b() - b.b();
    return dr * dr + dg * dg + db * db;
}

/// Integer squared distance over gamma-encoded sRGB bytes (no allocations,
/// no floats — used by Bambu metadata color slot matching).
inline int SrgbU8DistanceSq(const SrgbU8& a, const SrgbU8& b) {
    const int dr = static_cast<int>(a.r) - static_cast<int>(b.r);
    const int dg = static_cast<int>(a.g) - static_cast<int>(b.g);
    const int db = static_cast<int>(a.b) - static_cast<int>(b.b);
    return dr * dr + dg * dg + db * db;
}

/// BambuStudio-equivalent HSV distance: polar (cos·s·v, sin·s·v), capped.
/// Mirrors `calc_flush_vol_rgb`'s helper byte-for-byte.
inline float HsvDistanceBbs(const Hsv& a, const Hsv& b) {
    constexpr float kPi = 3.14159265358979323846f;
    const float h1r     = a.h() / 180.0f * kPi;
    const float h2r     = b.h() / 180.0f * kPi;
    const float dx      = std::cos(h1r) * a.s() * a.v() - std::cos(h2r) * b.s() * b.v();
    const float dy      = std::sin(h1r) * a.s() * a.v() - std::sin(h2r) * b.s() * b.v();
    return std::min(1.2f, std::sqrt(dx * dx + dy * dy));
}

/// Lab a*b* polar hue-angle difference, in degrees.
///
/// **NOT** the same as Hsv's hue: this is `atan2(b, a)` in the Lab a*b*
/// chromaticity plane. Used by the recipe-alternatives ranking heuristic.
inline float LabHueAngleDeg(const Lab& x, const Lab& y) {
    constexpr float kPi  = 3.14159265358979323846f;
    constexpr float k180 = 180.0f / kPi;
    float h1             = std::atan2(x.b(), x.a()) * k180;
    float h2             = std::atan2(y.b(), y.a()) * k180;
    float diff           = std::fabs(h1 - h2);
    if (diff > 180.0f) diff = 360.0f - diff;
    return diff;
}

} // namespace ChromaPrint3D
