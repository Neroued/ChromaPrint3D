#pragma once

/// \file color/conversions.h
/// \brief Scalar color-space conversion primitives — sRGB↔linear, RGB↔XYZ↔Lab,
///        HSV↔RGB. Project-D65 math is the **only** authoritative path.
///
/// Constants live in `core/src/color/conversions.cpp` (256-entry sRGB→linear
/// LUT, 4096-entry LabF LUT). Image-level entries that operate on `cv::Mat`
/// live in `color/image.h`.

#include "chromaprint3d/color/types.h"

#include <cmath>
#include <cstdint>

namespace ChromaPrint3D {

// ── sRGB <-> linear (scalar) ────────────────────────────────────────────────

/// sRGB gamma → linear (per-channel float [0,1]).
inline float SrgbToLinear(float c) {
    if (c <= 0.04045f) return c / 12.92f;
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

/// Linear → sRGB gamma (per-channel float [0,1]).
inline float LinearToSrgb(float c) {
    if (c <= 0.0031308f) return 12.92f * c;
    return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

/// sRGB byte (0..255) → linear float using shared 256-entry LUT.
/// Defined in `core/src/color/conversions.cpp`.
float SrgbToLinearByte(std::uint8_t v);

/// Linear float → sRGB byte (round + clamp). Inverse of `SrgbToLinearByte`.
std::uint8_t LinearToSrgbByte(float linear);

// ── Project-D65 RGB ↔ XYZ matrices (scalar) ─────────────────────────────────

/// Linear RGB → XYZ under D65 (project math).
inline Vec3f RGBToXYZ(const Vec3f& rgb) {
    float r = rgb.x, g = rgb.y, b = rgb.z;
    return {0.4124564f * r + 0.3575761f * g + 0.1804375f * b,
            0.2126729f * r + 0.7151522f * g + 0.0721750f * b,
            0.0193339f * r + 0.1191920f * g + 0.9503041f * b};
}

/// XYZ → linear RGB under D65 (inverse matrix).
inline Vec3f XYZToRGB(const Vec3f& xyz) {
    return {3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
            -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
            0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z};
}

// ── LabF / LabInvF (project math: (6/29)^3 threshold, true cbrt) ────────────

inline float LabF(float t) {
    constexpr float delta  = 6.0f / 29.0f;
    constexpr float delta3 = delta * delta * delta;
    if (t > delta3) return std::cbrt(t);
    return t / (3.0f * delta * delta) + 4.0f / 29.0f;
}

inline float LabInvF(float t) {
    constexpr float delta = 6.0f / 29.0f;
    if (t > delta) return t * t * t;
    return 3.0f * delta * delta * (t - 4.0f / 29.0f);
}

/// XYZ → CIE L*a*b* under D65 (project math).
inline Vec3f XYZToLab(const Vec3f& xyz) {
    constexpr float Xn = 0.95047f, Yn = 1.00000f, Zn = 1.08883f;
    float fx = LabF(xyz.x / Xn);
    float fy = LabF(xyz.y / Yn);
    float fz = LabF(xyz.z / Zn);
    return {116.0f * fy - 16.0f, 500.0f * (fx - fy), 200.0f * (fy - fz)};
}

/// CIE L*a*b* → XYZ under D65 (project math).
inline Vec3f LabToXYZ(const Vec3f& lab) {
    constexpr float Xn = 0.95047f, Yn = 1.00000f, Zn = 1.08883f;
    float fy = (lab.x + 16.0f) / 116.0f;
    float fx = fy + lab.y / 500.0f;
    float fz = fy - lab.z / 200.0f;
    return {Xn * LabInvF(fx), Yn * LabInvF(fy), Zn * LabInvF(fz)};
}

// ── Inline `Rgb` / `Lab` / `SrgbU8` conversion bodies ───────────────────────

inline Lab Rgb::ToLab() const {
    Vec3f lab = XYZToLab(RGBToXYZ(AsVec3f()));
    return Lab(lab.x, lab.y, lab.z);
}

inline Rgb Rgb::FromLab(const Lab& lab) {
    Vec3f rgb = XYZToRGB(LabToXYZ(lab.AsVec3f()));
    return Rgb(rgb.x, rgb.y, rgb.z);
}

inline Rgb Lab::ToRgb() const {
    Vec3f rgb = XYZToRGB(LabToXYZ(AsVec3f()));
    return Rgb(rgb.x, rgb.y, rgb.z);
}

inline Lab Lab::FromRgb(const Rgb& rgb) { return rgb.ToLab(); }

/// SrgbU8 → linear `Rgb` via 256-entry LUT (defined in conversions.cpp).
inline Rgb SrgbU8::ToRgb() const {
    return Rgb(SrgbToLinearByte(r), SrgbToLinearByte(g), SrgbToLinearByte(b));
}

/// Round-trip: linear `Rgb` → gamma-encoded byte. Clamps to [0,1] first.
inline SrgbU8 SrgbU8::FromRgb(const Rgb& rgb) {
    Rgb c = Rgb::Clamp01(rgb);
    return SrgbU8{LinearToSrgbByte(c.r()), LinearToSrgbByte(c.g()), LinearToSrgbByte(c.b())};
}

/// `Rgb::ToSrgbU8` convenience wrapper.
inline SrgbU8 RgbToSrgbU8(const Rgb& rgb) { return SrgbU8::FromRgb(rgb); }

// ── HSV (linear-input form) — used by `Hsv::FromRgb` / `Hsv::ToRgb` ─────────

namespace detail {

/// Shared HSV core: takes three components in [0,1] using the **same**
/// algebraic formula. Caller decides whether inputs are linear or gamma.
inline Hsv RgbToHsvAlg(float r, float g, float b) {
    const float Cmax  = std::max({r, g, b});
    const float Cmin  = std::min({r, g, b});
    const float delta = Cmax - Cmin;

    float h = 0.0f;
    if (std::abs(delta) < 1e-3f) {
        h = 0.0f;
    } else if (Cmax == r) {
        h = 60.0f * std::fmod((g - b) / delta, 6.0f);
    } else if (Cmax == g) {
        h = 60.0f * ((b - r) / delta + 2.0f);
    } else {
        h = 60.0f * ((r - g) / delta + 4.0f);
    }
    if (h < 0.0f) h += 360.0f;

    float s = (std::abs(Cmax) < 1e-3f) ? 0.0f : (delta / Cmax);
    float v = Cmax;
    return Hsv(h, s, v);
}

inline void HsvToRgbAlg(float h, float s, float v, float& r, float& g, float& b) {
    h = std::fmod(h, 360.0f);
    if (h < 0.0f) h += 360.0f;
    const float c = v * s;
    const float x = c * (1.0f - std::abs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
    const float m = v - c;
    float r0 = 0, g0 = 0, b0 = 0;
    if (h < 60.0f) {
        r0 = c;
        g0 = x;
    } else if (h < 120.0f) {
        r0 = x;
        g0 = c;
    } else if (h < 180.0f) {
        g0 = c;
        b0 = x;
    } else if (h < 240.0f) {
        g0 = x;
        b0 = c;
    } else if (h < 300.0f) {
        r0 = x;
        b0 = c;
    } else {
        r0 = c;
        b0 = x;
    }
    r = r0 + m;
    g = g0 + m;
    b = b0 + m;
}

} // namespace detail

inline Hsv Hsv::FromRgb(const Rgb& rgb) { return detail::RgbToHsvAlg(rgb.r(), rgb.g(), rgb.b()); }

inline Hsv Hsv::FromSrgbU8(const SrgbU8& c) {
    return detail::RgbToHsvAlg(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f);
}

inline Rgb Hsv::ToRgb() const {
    float r = 0, g = 0, b = 0;
    detail::HsvToRgbAlg(h_, s_, v_, r, g, b);
    return Rgb(r, g, b);
}

// ── ToHex helpers ───────────────────────────────────────────────────────────

inline std::string Rgb::ToHex() const { return SrgbU8::ToHex(SrgbU8::FromRgb(*this)); }

inline std::string Lab::ToHex() const { return SrgbU8::ToHex(SrgbU8::FromRgb(ToRgb())); }

} // namespace ChromaPrint3D
