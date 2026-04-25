#pragma once

/// \file export_bridge.h
/// \brief Zero-copy type bridge between ChromaPrint3D and neroued_3mf types.
///
/// Verifies layout compatibility at compile time and provides MakeMeshView()
/// to construct a neroued_3mf::MeshView directly from ChromaPrint3D::Mesh
/// without copying vertex or index data.

#include "chromaprint3d/mesh.h"
#include "chromaprint3d/vec3.h"

#include <neroued/3mf/types.h>

#include <cstddef>
#include <span>
#include <type_traits>

namespace ChromaPrint3D {
namespace detail {

// ── ChromaPrint3D::Vec3f ↔ neroued_3mf::Vec3f ────────────────────────────

static_assert(sizeof(ChromaPrint3D::Vec3f) == sizeof(neroued_3mf::Vec3f),
              "Vec3f types must have the same size");
static_assert(alignof(ChromaPrint3D::Vec3f) == alignof(neroued_3mf::Vec3f),
              "Vec3f types must have the same alignment");
static_assert(offsetof(ChromaPrint3D::Vec3f, x) == offsetof(neroued_3mf::Vec3f, x),
              "Vec3f::x offset mismatch");
static_assert(offsetof(ChromaPrint3D::Vec3f, y) == offsetof(neroued_3mf::Vec3f, y),
              "Vec3f::y offset mismatch");
static_assert(offsetof(ChromaPrint3D::Vec3f, z) == offsetof(neroued_3mf::Vec3f, z),
              "Vec3f::z offset mismatch");
static_assert(std::is_trivially_copyable_v<ChromaPrint3D::Vec3f>);
static_assert(std::is_trivially_copyable_v<neroued_3mf::Vec3f>);

// ── ChromaPrint3D::Vec3u ↔ neroued_3mf::IndexTriangle ────────────────────

static_assert(sizeof(Vec3u) == sizeof(neroued_3mf::IndexTriangle),
              "Vec3u and IndexTriangle must have the same size");
static_assert(alignof(Vec3u) == alignof(neroued_3mf::IndexTriangle),
              "Vec3u and IndexTriangle must have the same alignment");
static_assert(offsetof(Vec3u, x) == offsetof(neroued_3mf::IndexTriangle, v1),
              "Vec3u::x ↔ IndexTriangle::v1 offset mismatch");
static_assert(offsetof(Vec3u, y) == offsetof(neroued_3mf::IndexTriangle, v2),
              "Vec3u::y ↔ IndexTriangle::v2 offset mismatch");
static_assert(offsetof(Vec3u, z) == offsetof(neroued_3mf::IndexTriangle, v3),
              "Vec3u::z ↔ IndexTriangle::v3 offset mismatch");
static_assert(std::is_trivially_copyable_v<Vec3u>);
static_assert(std::is_trivially_copyable_v<neroued_3mf::IndexTriangle>);

/// Construct a zero-copy MeshView from a ChromaPrint3D Mesh.
/// The caller must keep \p mesh alive until the MeshView is consumed
/// (i.e., until WriteToBuffer / WriteToFile returns).
inline neroued_3mf::MeshView MakeMeshView(const Mesh& mesh) {
    return {
        .vertices = std::span<const neroued_3mf::Vec3f>(
            reinterpret_cast<const neroued_3mf::Vec3f*>(mesh.vertices.data()),
            mesh.vertices.size()),
        .triangles = std::span<const neroued_3mf::IndexTriangle>(
            reinterpret_cast<const neroued_3mf::IndexTriangle*>(mesh.indices.data()),
            mesh.indices.size()),
    };
}

} // namespace detail
} // namespace ChromaPrint3D
