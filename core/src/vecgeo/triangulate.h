#pragma once

/// \file triangulate.h
/// \brief Triangulate merged Clipper2 regions for extrusion.
///
/// The vertex ordering in TriangulatedRegion is critical:
/// vertices are stored in the exact order of polygon_groups traversal
/// (group → ring → vertex). ExtrudeSlab relies on this for wall index
/// computation.

#include "chromaprint3d/vector_image.h"
#include "chromaprint3d/vec3.h"

#include <clipper2/clipper.h>

#include <vector>

namespace ChromaPrint3D::detail {

/// Triangulated 2D region produced from merged Clipper2 paths.
struct TriangulatedRegion {
    /// Polygon groups for earcut: each group = [outer, hole1, hole2, ...].
    std::vector<std::vector<Contour>> polygon_groups;

    /// Flat vertex array, ordered by group→ring→vertex.
    std::vector<Vec2f> vertices;

    /// Triangle indices into `vertices`.
    std::vector<Vec3u> triangles;

    bool Empty() const { return polygon_groups.empty(); }
};

/// Classify Clipper2 Union result into outer/hole rings, group them, and
/// triangulate via earcut.
TriangulatedRegion TriangulateMergedPaths(const Clipper2Lib::Paths64& paths);

} // namespace ChromaPrint3D::detail
