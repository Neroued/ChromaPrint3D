#pragma once

/// \file mesh.h
/// \brief Triangle mesh generation from voxel grids.

#include "vec3.h"

#include <cstdint>
#include <vector>

namespace ChromaPrint3D {

struct VoxelGrid; // forward declaration

/// Per-interface z-offset applied during vertex generation in Mesh::Build().
/// Used to introduce a micro-gap between base and color meshes.
struct InterfaceZOffset {
    int z_index     = -1;   ///< Voxel z-index of the interface; <0 means inactive.
    float offset_mm = 0.0f; ///< Signed offset applied to vertices at this z-index.
};

/// Parameters for Mesh::Build().
struct BuildMeshConfig {
    float layer_height_mm   = 0.08f;
    float pixel_mm          = 1.0f;
    float base_color_gap_mm = 0.0f;
    bool double_sided       = false;
    InterfaceZOffset
        interface_offsets[2]; ///< At most 2 interfaces (single-sided: 1, double-sided: 2).
};

/// A simple indexed triangle mesh.
struct Mesh {
    std::vector<Vec3f> vertices;
    std::vector<Vec3u> indices;

    /// Generate a mesh from a VoxelGrid using greedy meshing.
    static Mesh Build(const VoxelGrid& voxel_grid, const BuildMeshConfig& cfg = BuildMeshConfig{});
};

} // namespace ChromaPrint3D
