#pragma once

/// \file color/types.h
/// \brief First-class color types: Rgb / Lab / SrgbU8 / Hsv.
///
/// Design contract (color unification refactor):
///   - `Rgb`  : linear sRGB, components in [0,1] float.
///   - `Lab`  : CIE L*a*b* under project-D65 math (`(6/29)^3` threshold,
///              D65 white point, true cbrt). NOT compatible with OpenCV
///              `cv::cvtColor(BGR2Lab)` outputs at sub-ΔE level — see
///              `docs/development.md` "Lab math" section.
///   - `SrgbU8`: gamma-encoded sRGB byte triple (no alpha). Used as the
///              hex / palette / 3MF metadata transport type.
///   - `Hsv`  : HSV with two semantic entries: `FromRgb` (linear, science
///              convention) and `FromSrgbU8` (gamma-encoded, BambuStudio
///              `calc_flush_vol_rgb` byte parity).
///
/// `Rgb` and `Lab` are **strong types** — they intentionally do NOT inherit
/// from `Vec3f`. Use the explicit `AsVec3f()` / `FromVec3f()` helpers when
/// crossing the boundary into geometry / OpenCV adapter code.

#include "chromaprint3d/vec3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace ChromaPrint3D {

struct Lab; // forward
struct Rgb; // forward

// ── Rgb (linear sRGB float[0,1]) ────────────────────────────────────────────

/// Linear sRGB color. Components in [0,1].
struct Rgb {
    float r_ = 0.0f;
    float g_ = 0.0f;
    float b_ = 0.0f;

    constexpr Rgb() = default;

    /// Construct from explicit linear components.
    constexpr Rgb(float r, float g, float b) : r_(r), g_(g), b_(b) {}

    constexpr float r() const { return r_; }

    constexpr float g() const { return g_; }

    constexpr float b() const { return b_; }

    float& r() { return r_; }

    float& g() { return g_; }

    float& b() { return b_; }

    /// Explicit boundary helpers for crossing into Vec3f-shaped APIs.
    constexpr Vec3f AsVec3f() const { return Vec3f(r_, g_, b_); }

    static constexpr Rgb FromVec3f(const Vec3f& v) { return Rgb(v.x, v.y, v.z); }

    /// Subscript: 0 = r, 1 = g, 2 = b. Required by kdt::HasSubscript so
    /// `kdt::KDTree<Entry, 3, RgbProj, ...>` can index components without
    /// going through `AsVec3f`.
    constexpr float operator[](int i) const { return i == 0 ? r_ : (i == 1 ? g_ : b_); }

    float& operator[](int i) { return i == 0 ? r_ : (i == 1 ? g_ : b_); }

    Rgb operator+(const Rgb& o) const { return {r_ + o.r_, g_ + o.g_, b_ + o.b_}; }

    Rgb operator-(const Rgb& o) const { return {r_ - o.r_, g_ - o.g_, b_ - o.b_}; }

    Rgb operator*(float s) const { return {r_ * s, g_ * s, b_ * s}; }

    Rgb operator/(float s) const { return {r_ / s, g_ / s, b_ / s}; }

    Rgb& operator+=(const Rgb& o) {
        r_ += o.r_;
        g_ += o.g_;
        b_ += o.b_;
        return *this;
    }

    Rgb& operator-=(const Rgb& o) {
        r_ -= o.r_;
        g_ -= o.g_;
        b_ -= o.b_;
        return *this;
    }

    Rgb& operator*=(float s) {
        r_ *= s;
        g_ *= s;
        b_ *= s;
        return *this;
    }

    Rgb& operator/=(float s) {
        r_ /= s;
        g_ /= s;
        b_ /= s;
        return *this;
    }

    bool operator==(const Rgb&) const = default;

    static Rgb Clamp(const Rgb& v, float lo, float hi) {
        auto c = [lo, hi](float x) { return std::max(lo, std::min(x, hi)); };
        return {c(v.r_), c(v.g_), c(v.b_)};
    }

    static Rgb Clamp01(const Rgb& v) { return Clamp(v, 0.0f, 1.0f); }

    static Rgb Lerp(const Rgb& a, const Rgb& b, float t) { return a + (b - a) * t; }

    static float Distance(const Rgb& a, const Rgb& b) {
        const float dr = a.r_ - b.r_, dg = a.g_ - b.g_, db = a.b_ - b.b_;
        return std::sqrt(dr * dr + dg * dg + db * db);
    }

    /// Convert this linear sRGB color to project-D65 Lab.
    Lab ToLab() const;

    /// Convert from project-D65 Lab.
    static Rgb FromLab(const Lab& lab);

    /// Hex shortcut: `#RRGGBB` (uppercase) via SrgbU8 round-trip.
    std::string ToHex() const;
};

inline Rgb operator*(float s, const Rgb& v) { return v * s; }

// ── Lab (CIE L*a*b* under project-D65 math) ─────────────────────────────────

/// Project-D65 L*a*b*. L* in [0,100], a*/b* nominally in [-128,127].
struct Lab {
    float l_ = 0.0f;
    float a_ = 0.0f;
    float b_ = 0.0f;

    constexpr Lab() = default;

    constexpr Lab(float l, float a, float b) : l_(l), a_(a), b_(b) {}

    constexpr float l() const { return l_; }

    constexpr float a() const { return a_; }

    constexpr float b() const { return b_; }

    float& l() { return l_; }

    float& a() { return a_; }

    float& b() { return b_; }

    constexpr Vec3f AsVec3f() const { return Vec3f(l_, a_, b_); }

    static constexpr Lab FromVec3f(const Vec3f& v) { return Lab(v.x, v.y, v.z); }

    /// Subscript: 0 = L*, 1 = a*, 2 = b*. Required by kdt::HasSubscript.
    constexpr float operator[](int i) const { return i == 0 ? l_ : (i == 1 ? a_ : b_); }

    float& operator[](int i) { return i == 0 ? l_ : (i == 1 ? a_ : b_); }

    Lab operator+(const Lab& o) const { return {l_ + o.l_, a_ + o.a_, b_ + o.b_}; }

    Lab operator-(const Lab& o) const { return {l_ - o.l_, a_ - o.a_, b_ - o.b_}; }

    Lab operator*(float s) const { return {l_ * s, a_ * s, b_ * s}; }

    Lab operator/(float s) const { return {l_ / s, a_ / s, b_ / s}; }

    Lab& operator+=(const Lab& o) {
        l_ += o.l_;
        a_ += o.a_;
        b_ += o.b_;
        return *this;
    }

    Lab& operator-=(const Lab& o) {
        l_ -= o.l_;
        a_ -= o.a_;
        b_ -= o.b_;
        return *this;
    }

    Lab& operator*=(float s) {
        l_ *= s;
        a_ *= s;
        b_ *= s;
        return *this;
    }

    Lab& operator/=(float s) {
        l_ /= s;
        a_ /= s;
        b_ /= s;
        return *this;
    }

    bool operator==(const Lab&) const = default;

    static Lab Lerp(const Lab& a, const Lab& b, float t) { return a + (b - a) * t; }

    static float Distance(const Lab& a, const Lab& b) {
        const float dl = a.l_ - b.l_, da = a.a_ - b.a_, db = a.b_ - b.b_;
        return std::sqrt(dl * dl + da * da + db * db);
    }

    /// CIE Delta-E 76 (Euclidean Lab distance).
    static float DeltaE76(const Lab& x, const Lab& y) { return Distance(x, y); }

    /// Convert this project-D65 Lab to linear sRGB.
    Rgb ToRgb() const;

    /// Convert from linear sRGB.
    static Lab FromRgb(const Rgb& rgb);

    /// Hex shortcut: project-Lab → linear-RGB → SrgbU8 → `#RRGGBB`.
    std::string ToHex() const;
};

inline Lab operator*(float s, const Lab& v) { return v * s; }

// ── SrgbU8 (gamma-encoded sRGB byte triple, no alpha) ───────────────────────

/// Gamma-encoded sRGB byte color used for hex / palette / 3MF transport.
struct SrgbU8 {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;

    constexpr SrgbU8() = default;

    constexpr SrgbU8(std::uint8_t r_, std::uint8_t g_, std::uint8_t b_) : r(r_), g(g_), b(b_) {}

    bool operator==(const SrgbU8&) const = default;

    /// Convert to linear `Rgb` via per-channel sRGB inverse-gamma LUT.
    Rgb ToRgb() const;

    /// Round-trip from linear `Rgb` (clamps then encodes via gamma).
    static SrgbU8 FromRgb(const Rgb& rgb);

    /// Hex parsing — strict input contract:
    ///   - Accepts `#RGB` / `#RRGGBB` / `0xRRGGBB` / `RGB` / `RRGGBB` (RGB only).
    ///   - **Rejects** any 8-payload form (`#RRGGBBAA` etc.) — RGBA is handled
    ///     by `neroued_3mf::Color::FromHex`. Returns `std::nullopt`.
    ///   - Rejects non-hex characters, empty, lengths != 3 / != 6.
    static std::optional<SrgbU8> FromHex(std::string_view hex);

    /// Format as `#RRGGBB` (uppercase).
    static std::string ToHex(const SrgbU8& c);

    std::string ToHex() const { return ToHex(*this); }
};

// ── Hsv (two semantic entries: FromRgb=linear / FromSrgbU8=gamma) ───────────

/// HSV color. h ∈ [0,360), s/v ∈ [0,1].
///
/// IMPORTANT: `FromRgb` and `FromSrgbU8` use the **same algebraic formula**
/// — only the input convention differs.
///   * `FromRgb(linear in [0,1])`  : science-convention HSV.
///   * `FromSrgbU8(gamma in [0,255])`: byte-level parity with BambuStudio
///     `calc_flush_vol_rgb` (used by flush volume calculator).
/// Do NOT use `FromRgb(linear)` to reproduce BBS behaviour, or vice versa.
struct Hsv {
    float h_ = 0.0f;
    float s_ = 0.0f;
    float v_ = 0.0f;

    constexpr Hsv() = default;

    constexpr Hsv(float h, float s, float v) : h_(h), s_(s), v_(v) {}

    constexpr float h() const { return h_; }

    constexpr float s() const { return s_; }

    constexpr float v() const { return v_; }

    bool operator==(const Hsv&) const = default;

    /// Construct from linear sRGB color (science-convention HSV).
    static Hsv FromRgb(const Rgb& rgb);

    /// Construct from gamma-encoded sRGB bytes (BambuStudio parity).
    static Hsv FromSrgbU8(const SrgbU8& c);

    /// Convert to linear sRGB (uses linear semantics).
    Rgb ToRgb() const;
};

} // namespace ChromaPrint3D

// ── std::hash specialisations ───────────────────────────────────────────────

namespace std {

template <>
struct hash<ChromaPrint3D::Rgb> {
    size_t operator()(const ChromaPrint3D::Rgb& c) const noexcept {
        size_t h = std::hash<float>{}(c.r());
        h ^= std::hash<float>{}(c.g()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(c.b()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

template <>
struct hash<ChromaPrint3D::SrgbU8> {
    size_t operator()(const ChromaPrint3D::SrgbU8& c) const noexcept {
        return (static_cast<size_t>(c.r) << 16) | (static_cast<size_t>(c.g) << 8) |
               static_cast<size_t>(c.b);
    }
};

} // namespace std
