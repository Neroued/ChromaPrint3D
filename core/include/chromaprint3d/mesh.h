#pragma once

/// \file mesh.h
/// \brief Triangle mesh generation from voxel grids.

#include "vec3.h"

#include <cstdint>
#include <vector>

namespace ChromaPrint3D {

struct VoxelGrid; // forward declaration

/// Parameters for Mesh::Build().
struct BuildMeshConfig {
    float layer_height_mm = 0.08f;
    float pixel_mm        = 1.0f;
};

/// A simple indexed triangle mesh.
struct Mesh {
    std::vector<Vec3f> vertices;
    std::vector<Vec3i> indices;

    /// Generate a mesh from a VoxelGrid using greedy meshing.
    static Mesh Build(const VoxelGrid& voxel_grid, const BuildMeshConfig& cfg = BuildMeshConfig{});
};

} // namespace ChromaPrint3D
