#pragma once

/// \file vec3.h
/// \brief 3-component integer and float vector types.

#include <algorithm>
#include <cmath>

namespace ChromaPrint3D {

/// 3-component integer vector.
struct Vec3i {
    int x = 0; ///< X component.
    int y = 0; ///< Y component.
    int z = 0; ///< Z component.

    constexpr Vec3i() = default;
    /// Constructs a vector with the given components.
    /// \param x_ X component
    /// \param y_ Y component
    /// \param z_ Z component
    constexpr Vec3i(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}

    /// Vector addition.
    Vec3i operator+(const Vec3i& o) const { return {x + o.x, y + o.y, z + o.z}; }
    /// Vector subtraction.
    Vec3i operator-(const Vec3i& o) const { return {x - o.x, y - o.y, z - o.z}; }
    /// Scalar multiplication.
    Vec3i operator*(int s) const { return {x * s, y * s, z * s}; }
    /// Scalar division.
    Vec3i operator/(int s) const { return {x / s, y / s, z / s}; }

    /// In-place vector addition.
    Vec3i& operator+=(const Vec3i& o) { x += o.x; y += o.y; z += o.z; return *this; }
    /// In-place vector subtraction.
    Vec3i& operator-=(const Vec3i& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    /// In-place scalar multiplication.
    Vec3i& operator*=(int s) { x *= s; y *= s; z *= s; return *this; }
    /// In-place scalar division.
    Vec3i& operator/=(int s) { x /= s; y /= s; z /= s; return *this; }

    /// Component access by index (0=x, 1=y, 2=z).
    int& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    /// Component access by index (0=x, 1=y, 2=z).
    const int& operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }

    /// Computes the dot product with another vector.
    /// \param o The other vector
    /// \return Dot product result
    int Dot(const Vec3i& o) const { return x * o.x + y * o.y + z * o.z; }
    /// Computes the squared length of the vector.
    /// \return Squared length (x² + y² + z²)
    int LengthSquared() const { return Dot(*this); }
};

inline Vec3i operator*(int s, const Vec3i& v) { return v * s; }

/// 3-component float vector.
struct Vec3f {
    float x = 0.0f; ///< X component.
    float y = 0.0f; ///< Y component.
    float z = 0.0f; ///< Z component.

    constexpr Vec3f() = default;
    /// Constructs a vector with the given components.
    /// \param x_ X component
    /// \param y_ Y component
    /// \param z_ Z component
    constexpr Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    /// Vector addition.
    Vec3f operator+(const Vec3f& o) const { return {x + o.x, y + o.y, z + o.z}; }
    /// Vector subtraction.
    Vec3f operator-(const Vec3f& o) const { return {x - o.x, y - o.y, z - o.z}; }
    /// Scalar multiplication.
    Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }
    /// Scalar division.
    Vec3f operator/(float s) const { return {x / s, y / s, z / s}; }

    /// In-place vector addition.
    Vec3f& operator+=(const Vec3f& o) { x += o.x; y += o.y; z += o.z; return *this; }
    /// In-place vector subtraction.
    Vec3f& operator-=(const Vec3f& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    /// In-place scalar multiplication.
    Vec3f& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    /// In-place scalar division.
    Vec3f& operator/=(float s) { x /= s; y /= s; z /= s; return *this; }

    /// Component access by index (0=x, 1=y, 2=z).
    float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    /// Component access by index (0=x, 1=y, 2=z).
    const float& operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }

    /// Computes the dot product with another vector.
    /// \param o The other vector
    /// \return Dot product result
    float Dot(const Vec3f& o) const { return x * o.x + y * o.y + z * o.z; }
    /// Computes the squared length of the vector.
    /// \return Squared length (x² + y² + z²)
    float LengthSquared() const { return Dot(*this); }
    /// Computes the length (magnitude) of the vector.
    /// \return Length √(x² + y² + z²)
    float Length() const { return std::sqrt(LengthSquared()); }

    /// Returns a normalized copy of this vector (unit length).
    /// \return Normalized vector, or zero vector if length is zero
    Vec3f Normalized() const {
        float len = Length();
        return len > 0.0f ? (*this / len) : Vec3f();
    }

    /// Checks if all components are finite (not NaN or infinity).
    /// \return True if all components are finite
    bool IsFinite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }

    /// Checks if this vector is approximately equal to another within tolerance.
    /// \param o The other vector to compare
    /// \param eps Tolerance threshold (default: 1e-5)
    /// \return True if all components differ by at most eps
    bool NearlyEqual(const Vec3f& o, float eps = 1e-5f) const {
        return std::fabs(x - o.x) <= eps && std::fabs(y - o.y) <= eps &&
               std::fabs(z - o.z) <= eps;
    }

    /// Linear interpolation between two vectors.
    /// \param a Start vector
    /// \param b End vector
    /// \param t Interpolation factor [0,1]
    /// \return Interpolated vector
    static Vec3f Lerp(const Vec3f& a, const Vec3f& b, float t) { return a + (b - a) * t; }

    /// Clamps each component to the specified range.
    /// \param v Vector to clamp
    /// \param lo Lower bound
    /// \param hi Upper bound
    /// \return Clamped vector
    static Vec3f Clamp(const Vec3f& v, float lo, float hi) {
        auto c = [](float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); };
        return {c(v.x, lo, hi), c(v.y, lo, hi), c(v.z, lo, hi)};
    }

    /// Clamps each component to [0, 1].
    /// \param v Vector to clamp
    /// \return Clamped vector
    static Vec3f Clamp01(const Vec3f& v) { return Clamp(v, 0.0f, 1.0f); }

    /// Computes the Euclidean distance between two vectors.
    /// \param a First vector
    /// \param b Second vector
    /// \return Distance between a and b
    static float Distance(const Vec3f& a, const Vec3f& b) { return (a - b).Length(); }
};

inline Vec3f operator*(float s, const Vec3f& v) { return v * s; }

} // namespace ChromaPrint3D
