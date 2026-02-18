#include <gtest/gtest.h>
#include "chromaprint3d/voxel.h"
#include "chromaprint3d/mesh.h"

using namespace ChromaPrint3D;

TEST(Mesh, BuildFromSingleVoxel) {
    VoxelGrid grid;
    grid.width      = 1;
    grid.height     = 1;
    grid.num_layers = 1;
    grid.ooc.assign(1, 0);
    grid.Set(0, 0, 0, true);

    BuildMeshConfig cfg;
    cfg.layer_height_mm = 1.0f;
    cfg.pixel_mm        = 1.0f;

    Mesh mesh = Mesh::Build(grid, cfg);
    EXPECT_FALSE(mesh.vertices.empty());
    EXPECT_FALSE(mesh.indices.empty());
    EXPECT_EQ(mesh.indices.size() % 1, 0u);
}

TEST(Mesh, EmptyGridProducesEmptyMesh) {
    VoxelGrid grid;
    grid.width      = 2;
    grid.height     = 2;
    grid.num_layers = 2;
    grid.ooc.assign(static_cast<size_t>(2 * 2 * 2), 0);

    Mesh mesh = Mesh::Build(grid);
    EXPECT_TRUE(mesh.vertices.empty());
    EXPECT_TRUE(mesh.indices.empty());
}

TEST(Mesh, TriangleIndicesAreValid) {
    VoxelGrid grid;
    grid.width      = 2;
    grid.height     = 2;
    grid.num_layers = 1;
    grid.ooc.assign(static_cast<size_t>(2 * 2 * 1), 0);
    grid.Set(0, 0, 0, true);
    grid.Set(1, 0, 0, true);

    Mesh mesh = Mesh::Build(grid);
    int max_vertex = static_cast<int>(mesh.vertices.size());
    for (const Vec3i& tri : mesh.indices) {
        EXPECT_GE(tri.x, 0);
        EXPECT_LT(tri.x, max_vertex);
        EXPECT_GE(tri.y, 0);
        EXPECT_LT(tri.y, max_vertex);
        EXPECT_GE(tri.z, 0);
        EXPECT_LT(tri.z, max_vertex);
    }
}

TEST(Mesh, VerticesAreFinite) {
    VoxelGrid grid;
    grid.width      = 3;
    grid.height     = 3;
    grid.num_layers = 2;
    grid.ooc.assign(static_cast<size_t>(3 * 3 * 2), 0);
    grid.Set(1, 1, 0, true);
    grid.Set(1, 1, 1, true);

    BuildMeshConfig cfg;
    cfg.layer_height_mm = 0.08f;
    cfg.pixel_mm        = 0.42f;

    Mesh mesh = Mesh::Build(grid, cfg);
    for (const Vec3f& v : mesh.vertices) {
        EXPECT_TRUE(v.IsFinite());
    }
}
