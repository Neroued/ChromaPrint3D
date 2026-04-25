#include <gtest/gtest.h>

#include "chromaprint3d/vector_mesh.h"
#include "chromaprint3d/vector_recipe_map.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ChromaPrint3D;

namespace {

Contour MakeRect(float x0, float y0, float x1, float y1) {
    return Contour{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
}

VectorRecipeMap BuildSingleChannelRecipeMap(int shape_count) {
    VectorRecipeMap map;
    map.color_layers = 1;
    map.num_channels = 1;
    map.layer_order  = LayerOrder::Top2Bottom;
    map.entries.reserve(static_cast<size_t>(shape_count));
    for (int i = 0; i < shape_count; ++i) {
        VectorRecipeMap::ShapeEntry entry;
        entry.shape_idx = i;
        entry.recipe    = {0};
        map.entries.push_back(std::move(entry));
    }
    return map;
}

struct EdgeKey {
    uint32_t a = 0;
    uint32_t b = 0;

    bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& e) const {
        size_t h = std::hash<uint32_t>{}(e.a);
        h ^= std::hash<uint32_t>{}(e.b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct FaceKey {
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;

    bool operator==(const FaceKey& o) const { return a == o.a && b == o.b && c == o.c; }
};

struct FaceKeyHash {
    size_t operator()(const FaceKey& f) const {
        size_t h = std::hash<uint32_t>{}(f.a);
        h ^= std::hash<uint32_t>{}(f.b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(f.c) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

EdgeKey MakeEdgeKey(uint32_t a, uint32_t b) {
    if (b < a) std::swap(a, b);
    return {a, b};
}

FaceKey MakeFaceKey(uint32_t i0, uint32_t i1, uint32_t i2) {
    std::array<uint32_t, 3> ids{i0, i1, i2};
    std::sort(ids.begin(), ids.end());
    return {ids[0], ids[1], ids[2]};
}

bool IsDegenerateByArea(const Mesh& mesh, const Vec3u& tri) {
    const Vec3f& a = mesh.vertices[static_cast<size_t>(tri.x)];
    const Vec3f& b = mesh.vertices[static_cast<size_t>(tri.y)];
    const Vec3f& c = mesh.vertices[static_cast<size_t>(tri.z)];
    Vec3f ab       = b - a;
    Vec3f ac       = c - a;
    float cx       = ab.y * ac.z - ab.z * ac.y;
    float cy       = ab.z * ac.x - ab.x * ac.z;
    float cz       = ab.x * ac.y - ab.y * ac.x;
    return (cx * cx + cy * cy + cz * cz) <= 1e-12f;
}

struct MeshTopologyMetrics {
    size_t non_manifold_edges   = 0;
    size_t duplicate_faces      = 0;
    size_t degenerate_triangles = 0;
};

MeshTopologyMetrics AnalyzeMesh(const Mesh& mesh) {
    MeshTopologyMetrics m;
    std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_use;
    std::unordered_set<FaceKey, FaceKeyHash> faces;
    edge_use.reserve(mesh.indices.size() * 3);
    faces.reserve(mesh.indices.size() * 2);

    const uint32_t max_idx = static_cast<uint32_t>(mesh.vertices.size());
    for (const Vec3u& tri : mesh.indices) {
        if (tri.x >= max_idx || tri.y >= max_idx || tri.z >= max_idx ||
            tri.x == tri.y || tri.y == tri.z || tri.x == tri.z ||
            IsDegenerateByArea(mesh, tri)) {
            ++m.degenerate_triangles;
            continue;
        }

        FaceKey fk = MakeFaceKey(tri.x, tri.y, tri.z);
        if (!faces.insert(fk).second) { ++m.duplicate_faces; }

        ++edge_use[MakeEdgeKey(tri.x, tri.y)];
        ++edge_use[MakeEdgeKey(tri.y, tri.z)];
        ++edge_use[MakeEdgeKey(tri.z, tri.x)];
    }

    for (const auto& [_, count] : edge_use) {
        if (count != 2) ++m.non_manifold_edges;
    }
    return m;
}

std::pair<float, float> MeshZRange(const Mesh& mesh) {
    if (mesh.vertices.empty()) { return {0.0f, 0.0f}; }
    float min_z = std::numeric_limits<float>::infinity();
    float max_z = -std::numeric_limits<float>::infinity();
    for (const Vec3f& v : mesh.vertices) {
        min_z = std::min(min_z, v.z);
        max_z = std::max(max_z, v.z);
    }
    return {min_z, max_z};
}

} // namespace

TEST(VectorMesh, SingleRectangleIsWatertight) {
    VectorShape shape;
    shape.contours.push_back(MakeRect(0.0f, 0.0f, 10.0f, 10.0f));

    std::vector<VectorShape> shapes{shape};
    VectorRecipeMap map = BuildSingleChannelRecipeMap(1);

    VectorMeshConfig cfg;
    cfg.layer_height_mm = 0.2f;

    std::vector<Mesh> meshes = BuildVectorMeshes(shapes, map, cfg);
    ASSERT_EQ(meshes.size(), 1u);
    ASSERT_FALSE(meshes[0].indices.empty());

    MeshTopologyMetrics metrics = AnalyzeMesh(meshes[0]);
    EXPECT_EQ(metrics.non_manifold_edges, 0u);
    EXPECT_EQ(metrics.duplicate_faces, 0u);
    EXPECT_EQ(metrics.degenerate_triangles, 0u);
}

TEST(VectorMesh, AdjacentRectanglesDoNotGenerateInternalWalls) {
    VectorShape left;
    left.contours.push_back(MakeRect(0.0f, 0.0f, 10.0f, 10.0f));
    VectorShape right;
    right.contours.push_back(MakeRect(10.0f, 0.0f, 20.0f, 10.0f));

    std::vector<VectorShape> shapes{left, right};
    VectorRecipeMap map = BuildSingleChannelRecipeMap(2);

    VectorMeshConfig cfg;
    cfg.layer_height_mm = 0.2f;

    std::vector<Mesh> meshes = BuildVectorMeshes(shapes, map, cfg);
    ASSERT_EQ(meshes.size(), 1u);
    ASSERT_FALSE(meshes[0].indices.empty());

    MeshTopologyMetrics metrics = AnalyzeMesh(meshes[0]);
    EXPECT_EQ(metrics.non_manifold_edges, 0u);
    EXPECT_EQ(metrics.duplicate_faces, 0u);
    EXPECT_EQ(metrics.degenerate_triangles, 0u);
    EXPECT_LT(meshes[0].indices.size(), 24u);
}

TEST(VectorMesh, DoubleSidedUsesMirroredColorLayersWithBaseInMiddle) {
    VectorShape shape;
    shape.contours.push_back(MakeRect(0.0f, 0.0f, 10.0f, 10.0f));
    std::vector<VectorShape> shapes{shape};

    VectorRecipeMap map = BuildSingleChannelRecipeMap(1);

    VectorMeshConfig cfg;
    cfg.layer_height_mm = 0.2f;
    cfg.base_layers     = 2;
    cfg.double_sided    = true;

    std::vector<Mesh> meshes = BuildVectorMeshes(shapes, map, cfg);
    ASSERT_EQ(meshes.size(), 2u);
    ASSERT_FALSE(meshes[0].indices.empty());
    ASSERT_FALSE(meshes[1].indices.empty());

    constexpr float kTol                  = 1e-6f;
    const auto [color_min_z, color_max_z] = MeshZRange(meshes[0]);
    const auto [base_min_z, base_max_z]   = MeshZRange(meshes[1]);

    EXPECT_NEAR(color_min_z, 0.0f, kTol);
    EXPECT_NEAR(color_max_z, 0.8f, kTol);
    EXPECT_NEAR(base_min_z, 0.2f, kTol);
    EXPECT_NEAR(base_max_z, 0.6f, kTol);
}

TEST(VectorMesh, SingleSidedGapSeparatesBaseAndColor) {
    VectorShape shape;
    shape.contours.push_back(MakeRect(0.0f, 0.0f, 10.0f, 10.0f));
    std::vector<VectorShape> shapes{shape};
    VectorRecipeMap map = BuildSingleChannelRecipeMap(1);

    constexpr float kLh  = 0.2f;
    constexpr float kGap = 0.04f;

    VectorMeshConfig cfg;
    cfg.layer_height_mm   = kLh;
    cfg.base_layers       = 2;
    cfg.base_color_gap_mm = kGap;

    std::vector<Mesh> meshes = BuildVectorMeshes(shapes, map, cfg);
    ASSERT_EQ(meshes.size(), 2u);

    constexpr float kTol                  = 1e-6f;
    const auto [color_min_z, color_max_z] = MeshZRange(meshes[0]);
    const auto [base_min_z, base_max_z]   = MeshZRange(meshes[1]);

    EXPECT_NEAR(base_min_z, 0.0f, kTol);
    EXPECT_NEAR(base_max_z, 2 * kLh - kGap * 0.5f, kTol);
    EXPECT_NEAR(color_min_z, 2 * kLh + kGap * 0.5f, kTol);
    EXPECT_NEAR(color_min_z - base_max_z, kGap, kTol);
}

TEST(VectorMesh, DoubleSidedGapSeparatesBothInterfaces) {
    VectorShape shape;
    shape.contours.push_back(MakeRect(0.0f, 0.0f, 10.0f, 10.0f));
    std::vector<VectorShape> shapes{shape};
    VectorRecipeMap map = BuildSingleChannelRecipeMap(1);

    constexpr float kLh  = 0.2f;
    constexpr float kGap = 0.04f;
    constexpr float kHG  = kGap * 0.5f;

    VectorMeshConfig cfg;
    cfg.layer_height_mm   = kLh;
    cfg.base_layers       = 2;
    cfg.double_sided      = true;
    cfg.base_color_gap_mm = kGap;

    std::vector<Mesh> meshes = BuildVectorMeshes(shapes, map, cfg);
    ASSERT_EQ(meshes.size(), 2u);

    constexpr float kTol                  = 1e-6f;
    const auto [color_min_z, color_max_z] = MeshZRange(meshes[0]);
    const auto [base_min_z, base_max_z]   = MeshZRange(meshes[1]);

    const float base_start_z = 1 * kLh;
    EXPECT_NEAR(base_min_z, base_start_z + kHG, kTol);
    EXPECT_NEAR(base_max_z, (1 + 2) * kLh - kHG, kTol);

    EXPECT_NEAR(color_min_z, 0.0f, kTol);
    EXPECT_NEAR(color_max_z, 0.8f, kTol);

    float bottom_gap = base_min_z - (base_start_z - kHG);
    EXPECT_NEAR(bottom_gap, kGap, kTol);
    float top_gap = ((1 + 2) * kLh + kHG) - base_max_z;
    EXPECT_NEAR(top_gap, kGap, kTol);
}

TEST(VectorMesh, ZeroGapIsNoOp) {
    VectorShape shape;
    shape.contours.push_back(MakeRect(0.0f, 0.0f, 10.0f, 10.0f));
    std::vector<VectorShape> shapes{shape};
    VectorRecipeMap map = BuildSingleChannelRecipeMap(1);

    VectorMeshConfig cfg_nogap;
    cfg_nogap.layer_height_mm = 0.2f;
    cfg_nogap.base_layers     = 2;

    VectorMeshConfig cfg_zero  = cfg_nogap;
    cfg_zero.base_color_gap_mm = 0.0f;

    std::vector<Mesh> m1 = BuildVectorMeshes(shapes, map, cfg_nogap);
    std::vector<Mesh> m2 = BuildVectorMeshes(shapes, map, cfg_zero);
    ASSERT_EQ(m1.size(), m2.size());
    for (size_t i = 0; i < m1.size(); ++i) {
        EXPECT_EQ(m1[i].vertices.size(), m2[i].vertices.size());
        EXPECT_EQ(m1[i].indices.size(), m2[i].indices.size());
    }
}
