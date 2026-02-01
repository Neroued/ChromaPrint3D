#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ChromaPrint3D {

struct Vec3f {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    constexpr Vec3f() = default;

    constexpr Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3f operator+(const Vec3f& other) const {
        return Vec3f(x + other.x, y + other.y, z + other.z);
    }

    Vec3f operator-(const Vec3f& other) const {
        return Vec3f(x - other.x, y - other.y, z - other.z);
    }

    Vec3f operator*(float s) const { return Vec3f(x * s, y * s, z * s); }

    Vec3f operator/(float s) const { return Vec3f(x / s, y / s, z / s); }

    Vec3f& operator+=(const Vec3f& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vec3f& operator-=(const Vec3f& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    Vec3f& operator*=(float s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    Vec3f& operator/=(float s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    float Dot(const Vec3f& other) const { return x * other.x + y * other.y + z * other.z; }

    float LengthSquared() const { return Dot(*this); }

    float Length() const { return std::sqrt(LengthSquared()); }

    Vec3f Normalized() const {
        float len = Length();
        return len > 0.0f ? (*this / len) : Vec3f();
    }

    bool IsFinite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }

    bool NearlyEqual(const Vec3f& other, float eps = 1e-5f) const {
        return std::fabs(x - other.x) <= eps && std::fabs(y - other.y) <= eps &&
               std::fabs(z - other.z) <= eps;
    }

    static Vec3f Lerp(const Vec3f& a, const Vec3f& b, float t) { return a + (b - a) * t; }

    static Vec3f Clamp(const Vec3f& v, float lo, float hi) {
        return Vec3f(ClampFloat(v.x, lo, hi), ClampFloat(v.y, lo, hi), ClampFloat(v.z, lo, hi));
    }

    static Vec3f Clamp01(const Vec3f& v) { return Clamp(v, 0.0f, 1.0f); }

    static float Distance(const Vec3f& a, const Vec3f& b) { return (a - b).Length(); }

private:
    static float ClampFloat(float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); }
};

inline Vec3f operator*(float s, const Vec3f& v) { return v * s; }

inline float SrgbToLinear(float c) {
    if (c <= 0.04045f) { return c / 12.92f; }
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

inline float LinearToSrgb(float c) {
    if (c <= 0.0031308f) { return 12.92f * c; }
    return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

inline Vec3f RGBToXYZ(const Vec3f& rgb) {
    float r = rgb.x;
    float g = rgb.y;
    float b = rgb.z;
    return Vec3f(0.4124564f * r + 0.3575761f * g + 0.1804375f * b,
                 0.2126729f * r + 0.7151522f * g + 0.0721750f * b,
                 0.0193339f * r + 0.1191920f * g + 0.9503041f * b);
}

inline Vec3f XYZToRGB(const Vec3f& xyz) {
    float r = 3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z;
    float g = -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z;
    float b = 0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z;
    return Vec3f(r, g, b);
}

inline float LabF(float t) {
    const float delta  = 6.0f / 29.0f;
    const float delta3 = delta * delta * delta;
    if (t > delta3) { return std::cbrt(t); }
    return t / (3.0f * delta * delta) + 4.0f / 29.0f;
}

inline float LabInvF(float t) {
    const float delta = 6.0f / 29.0f;
    if (t > delta) { return t * t * t; }
    return 3.0f * delta * delta * (t - 4.0f / 29.0f);
}

inline Vec3f XYZToLab(const Vec3f& xyz) {
    const float Xn = 0.95047f;
    const float Yn = 1.00000f;
    const float Zn = 1.08883f;
    float fx       = LabF(xyz.x / Xn);
    float fy       = LabF(xyz.y / Yn);
    float fz       = LabF(xyz.z / Zn);
    return Vec3f(116.0f * fy - 16.0f, 500.0f * (fx - fy), 200.0f * (fy - fz));
}

inline Vec3f LabToXYZ(const Vec3f& lab) {
    const float Xn = 0.95047f;
    const float Yn = 1.00000f;
    const float Zn = 1.08883f;
    float fy       = (lab.x + 16.0f) / 116.0f;
    float fx       = fy + lab.y / 500.0f;
    float fz       = fy - lab.z / 200.0f;
    return Vec3f(Xn * LabInvF(fx), Yn * LabInvF(fy), Zn * LabInvF(fz));
}

struct Lab;

struct Rgb : public Vec3f {
    constexpr Rgb() = default;

    constexpr Rgb(float r_, float g_, float b_) : Vec3f(r_, g_, b_) {}

    explicit constexpr Rgb(const Vec3f& v) : Vec3f(v) {}

    float& r() { return x; }

    float& g() { return y; }

    float& b() { return z; }

    const float& r() const { return x; }

    const float& g() const { return y; }

    const float& b() const { return z; }

    static constexpr Rgb FromRgb(float r, float g, float b) { return Rgb(r, g, b); }

    static Rgb FromRgb255(uint8_t r, uint8_t g, uint8_t b) {
        return Rgb(SrgbToLinear(r / 255.0f), SrgbToLinear(g / 255.0f),
                   SrgbToLinear(b / 255.0f));
    }

    void ToRgb255(uint8_t& r8, uint8_t& g8, uint8_t& b8) const {
        Rgb c = Clamp01(*this);
        Rgb s(LinearToSrgb(c.x), LinearToSrgb(c.y), LinearToSrgb(c.z));
        r8 = static_cast<uint8_t>(std::round(s.x * 255.0f));
        g8 = static_cast<uint8_t>(std::round(s.y * 255.0f));
        b8 = static_cast<uint8_t>(std::round(s.z * 255.0f));
    }

    Lab ToLab() const;
    static Rgb FromLab(const Lab& lab);

    Rgb operator+(const Rgb& other) const { return Rgb(x + other.x, y + other.y, z + other.z); }

    Rgb operator-(const Rgb& other) const { return Rgb(x - other.x, y - other.y, z - other.z); }

    Rgb operator*(float s) const { return Rgb(x * s, y * s, z * s); }

    Rgb operator/(float s) const { return Rgb(x / s, y / s, z / s); }

    Rgb& operator+=(const Rgb& other) {
        Vec3f::operator+=(other);
        return *this;
    }

    Rgb& operator-=(const Rgb& other) {
        Vec3f::operator-=(other);
        return *this;
    }

    Rgb& operator*=(float s) {
        Vec3f::operator*=(s);
        return *this;
    }

    Rgb& operator/=(float s) {
        Vec3f::operator/=(s);
        return *this;
    }

    static Rgb Lerp(const Rgb& a, const Rgb& b, float t) { return a + (b - a) * t; }

    static Rgb Clamp(const Rgb& v, float lo, float hi) {
        Vec3f clamped = Vec3f::Clamp(v, lo, hi);
        return Rgb(clamped.x, clamped.y, clamped.z);
    }

    static Rgb Clamp01(const Rgb& v) { return Clamp(v, 0.0f, 1.0f); }

    static float Distance(const Rgb& a, const Rgb& b) { return Vec3f::Distance(a, b); }
};

struct Lab : public Vec3f {
    constexpr Lab() = default;

    constexpr Lab(float l_, float a_, float b_) : Vec3f(l_, a_, b_) {}

    explicit constexpr Lab(const Vec3f& v) : Vec3f(v) {}

    float& l() { return x; }

    float& a() { return y; }

    float& b() { return z; }

    const float& l() const { return x; }

    const float& a() const { return y; }

    const float& b() const { return z; }

    static constexpr Lab FromLab(float l, float a, float b) { return Lab(l, a, b); }

    Rgb ToRgb() const;
    static Lab FromRgb(const Rgb& rgb);

    Lab operator+(const Lab& other) const { return Lab(x + other.x, y + other.y, z + other.z); }

    Lab operator-(const Lab& other) const { return Lab(x - other.x, y - other.y, z - other.z); }

    Lab operator*(float s) const { return Lab(x * s, y * s, z * s); }

    Lab operator/(float s) const { return Lab(x / s, y / s, z / s); }

    Lab& operator+=(const Lab& other) {
        Vec3f::operator+=(other);
        return *this;
    }

    Lab& operator-=(const Lab& other) {
        Vec3f::operator-=(other);
        return *this;
    }

    Lab& operator*=(float s) {
        Vec3f::operator*=(s);
        return *this;
    }

    Lab& operator/=(float s) {
        Vec3f::operator/=(s);
        return *this;
    }

    static Lab Lerp(const Lab& a, const Lab& b, float t) { return a + (b - a) * t; }

    static float Distance(const Lab& a, const Lab& b) { return Vec3f::Distance(a, b); }

    static float DeltaE76(const Lab& lab1, const Lab& lab2) { return Distance(lab1, lab2); }
};

inline Rgb operator*(float s, const Rgb& v) { return v * s; }

inline Lab operator*(float s, const Lab& v) { return v * s; }

inline Lab Rgb::ToLab() const {
    Vec3f lab = XYZToLab(RGBToXYZ(*this));
    return Lab(lab.x, lab.y, lab.z);
}

inline Rgb Rgb::FromLab(const Lab& lab) {
    Vec3f rgb = XYZToRGB(LabToXYZ(lab));
    return Rgb(rgb.x, rgb.y, rgb.z);
}

inline Rgb Lab::ToRgb() const {
    Vec3f rgb = XYZToRGB(LabToXYZ(*this));
    return Rgb(rgb.x, rgb.y, rgb.z);
}

inline Lab Lab::FromRgb(const Rgb& rgb) { return rgb.ToLab(); }

} // namespace ChromaPrint3D