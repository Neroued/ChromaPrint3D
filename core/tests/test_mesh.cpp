#include <gtest/gtest.h>
#include "chromaprint3d/voxel.h"
#include "chromaprint3d/mesh.h"

#include <algorithm>
#include <cmath>
#include <limits>

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

    Mesh mesh           = Mesh::Build(grid);
    uint32_t max_vertex = static_cast<uint32_t>(mesh.vertices.size());
    for (const Vec3u& tri : mesh.indices) {
        EXPECT_LT(tri.x, max_vertex);
        EXPECT_LT(tri.y, max_vertex);
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
    for (const Vec3f& v : mesh.vertices) { EXPECT_TRUE(v.IsFinite()); }
}

namespace {

std::pair<float, float> CollectZRange(const Mesh& mesh) {
    float lo = std::numeric_limits<float>::infinity();
    float hi = -std::numeric_limits<float>::infinity();
    for (const Vec3f& v : mesh.vertices) {
        lo = std::min(lo, v.z);
        hi = std::max(hi, v.z);
    }
    return {lo, hi};
}

VoxelGrid MakeFilledGrid(int w, int h, int layers) {
    VoxelGrid g;
    g.width      = w;
    g.height     = h;
    g.num_layers = layers;
    g.ooc.assign(static_cast<size_t>(w * h * layers), 1);
    return g;
}

} // namespace

TEST(Mesh, InterfaceOffsetShiftsZAtExactIndex) {
    VoxelGrid grid;
    grid.width      = 2;
    grid.height     = 2;
    grid.num_layers = 4;
    grid.ooc.assign(static_cast<size_t>(2 * 2 * 4), 0);
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z) grid.Set(x, y, z, true);

    BuildMeshConfig cfg;
    cfg.layer_height_mm      = 1.0f;
    cfg.pixel_mm             = 1.0f;
    cfg.interface_offsets[0] = {2, -0.1f};

    Mesh mesh = Mesh::Build(grid, cfg);
    ASSERT_FALSE(mesh.vertices.empty());

    bool found_shifted = false;
    for (const Vec3f& v : mesh.vertices) {
        EXPECT_TRUE(v.IsFinite());
        if (std::fabs(v.z - 1.9f) < 1e-6f) { found_shifted = true; }
        EXPECT_GT(std::fabs(v.z - 2.0f), 1e-7f)
            << "No vertex should remain at the unshifted interface z=2.0";
    }
    EXPECT_TRUE(found_shifted) << "Expected a vertex at shifted z=1.9";
}

TEST(Mesh, InterfaceOffsetZeroGapIsNoOp) {
    VoxelGrid grid = MakeFilledGrid(2, 2, 3);

    BuildMeshConfig cfg_plain;
    cfg_plain.layer_height_mm = 1.0f;
    cfg_plain.pixel_mm        = 1.0f;

    BuildMeshConfig cfg_gap0   = cfg_plain;
    cfg_gap0.base_color_gap_mm = 0.0f;

    Mesh m1 = Mesh::Build(grid, cfg_plain);
    Mesh m2 = Mesh::Build(grid, cfg_gap0);
    EXPECT_EQ(m1.vertices.size(), m2.vertices.size());
    EXPECT_EQ(m1.indices.size(), m2.indices.size());
}

TEST(Mesh, TwoGridsWithGapHaveCorrectSeparation) {
    constexpr int kBaseLayers  = 2;
    constexpr int kColorLayers = 2;
    constexpr int kTotal       = kBaseLayers + kColorLayers;
    constexpr float kLh        = 1.0f;
    constexpr float kGap       = 0.1f;
    constexpr float kHalfGap   = kGap * 0.5f;

    VoxelGrid base_grid = MakeFilledGrid(2, 2, kTotal);
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = kBaseLayers; z < kTotal; ++z) base_grid.Set(x, y, z, false);

    VoxelGrid color_grid = MakeFilledGrid(2, 2, kTotal);
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < kBaseLayers; ++z) color_grid.Set(x, y, z, false);

    BuildMeshConfig base_cfg;
    base_cfg.layer_height_mm      = kLh;
    base_cfg.pixel_mm             = 1.0f;
    base_cfg.interface_offsets[0] = {kBaseLayers, -kHalfGap};

    BuildMeshConfig color_cfg;
    color_cfg.layer_height_mm      = kLh;
    color_cfg.pixel_mm             = 1.0f;
    color_cfg.interface_offsets[0] = {kBaseLayers, +kHalfGap};

    Mesh base_mesh  = Mesh::Build(base_grid, base_cfg);
    Mesh color_mesh = Mesh::Build(color_grid, color_cfg);

    auto [base_min, base_max]   = CollectZRange(base_mesh);
    auto [color_min, color_max] = CollectZRange(color_mesh);

    EXPECT_NEAR(base_max, static_cast<float>(kBaseLayers) * kLh - kHalfGap, 1e-6f);
    EXPECT_NEAR(color_min, static_cast<float>(kBaseLayers) * kLh + kHalfGap, 1e-6f);
    EXPECT_NEAR(color_min - base_max, kGap, 1e-6f);
}
