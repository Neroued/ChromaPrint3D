#pragma once

/// \file vec3.h
/// \brief 3-component vector types: generic `Vec3<T>` (with `Vec3i`/`Vec3u`
/// aliases) for integer coordinates / mesh indices, and a dedicated `Vec3f`
/// for floating-point geometry.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ChromaPrint3D {

/// 3-component generic vector. Used with integer instantiations for voxel
/// coordinates and mesh indices; see the `Vec3i`/`Vec3u` aliases below.
template <typename T>
struct Vec3 {
    T x = 0; ///< X component.
    T y = 0; ///< Y component.
    T z = 0; ///< Z component.

    constexpr Vec3() = default;

    /// Constructs a vector with the given components.
    /// \param x_ X component
    /// \param y_ Y component
    /// \param z_ Z component
    constexpr Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    /// Vector addition.
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }

    /// Vector subtraction.
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }

    /// Scalar multiplication.
    Vec3 operator*(T s) const { return {x * s, y * s, z * s}; }

    /// Scalar division.
    Vec3 operator/(T s) const { return {x / s, y / s, z / s}; }

    /// In-place vector addition.
    Vec3& operator+=(const Vec3& o) {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }

    /// In-place vector subtraction.
    Vec3& operator-=(const Vec3& o) {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return *this;
    }

    /// In-place scalar multiplication.
    Vec3& operator*=(T s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    /// In-place scalar division.
    Vec3& operator/=(T s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    /// Component access by index (0=x, 1=y, 2=z).
    T& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }

    /// Component access by index (0=x, 1=y, 2=z).
    const T& operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }

    /// Computes the dot product with another vector.
    /// \param o The other vector
    /// \return Dot product result
    T Dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }

    /// Computes the squared length of the vector.
    /// \return Squared length (x² + y² + z²)
    T LengthSquared() const { return Dot(*this); }

    /// Componentwise equality (auto-generated `!=` via C++20).
    bool operator==(const Vec3&) const = default;
};

template <typename S, typename T>
inline Vec3<T> operator*(S s, const Vec3<T>& v) {
    return v * static_cast<T>(s);
}

/// Signed 32-bit integer vector. Used for voxel coordinates and signed offsets.
using Vec3i = Vec3<int32_t>;

/// Unsigned 32-bit integer vector. Used for mesh triangle indices; matches
/// `neroued_3mf::IndexTriangle` layout for zero-copy 3MF export.
using Vec3u = Vec3<uint32_t>;

static_assert(sizeof(Vec3u) == 12, "Vec3u must be 12 bytes for layout compatibility with IndexTriangle");

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
    Vec3f& operator+=(const Vec3f& o) {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }

    /// In-place vector subtraction.
    Vec3f& operator-=(const Vec3f& o) {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return *this;
    }

    /// In-place scalar multiplication.
    Vec3f& operator*=(float s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    /// In-place scalar division.
    Vec3f& operator/=(float s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

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
        return std::fabs(x - o.x) <= eps && std::fabs(y - o.y) <= eps && std::fabs(z - o.z) <= eps;
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
