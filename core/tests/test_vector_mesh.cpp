#include <gtest/gtest.h>

#include "chromaprint3d/vector_mesh.h"
#include "chromaprint3d/vector_recipe_map.h"
#include "vecgeo/triangulate.h"

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

struct MeshTopologyMetrics {
    size_t open_edges           = 0;
    size_t non_manifold_edges   = 0;
    size_t duplicate_faces      = 0;
    size_t degenerate_triangles = 0;
};

// Mirrors BambuStudio's its_edge_diagnostics(): purely index-based edge
// counting with no positional welding. Only faces with repeated or
// out-of-range indices are excluded; zero-area faces with distinct indices
// still contribute edges, exactly as the slicer sees them.
MeshTopologyMetrics AnalyzeMesh(const Mesh& mesh) {
    MeshTopologyMetrics m;
    std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_use;
    std::unordered_set<FaceKey, FaceKeyHash> faces;
    edge_use.reserve(mesh.indices.size() * 3);
    faces.reserve(mesh.indices.size() * 2);

    const uint32_t max_idx = static_cast<uint32_t>(mesh.vertices.size());
    for (const Vec3u& tri : mesh.indices) {
        if (tri.x >= max_idx || tri.y >= max_idx || tri.z >= max_idx || tri.x == tri.y ||
            tri.y == tri.z || tri.x == tri.z) {
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
        if (count == 1) {
            ++m.open_edges;
        } else if (count > 2) {
            ++m.non_manifold_edges;
        }
    }
    return m;
}

// Verify the TriangulatedRegion contract required by ExtrudeSlab for a
// watertight extrusion: vertices laid out group→ring→vertex, every ring edge
// used exactly once by the cap triangulation (the wall quad supplies the
// second use), and every non-ring edge used exactly twice (interior).
void ExpectRegionExtrudable(const ChromaPrint3D::detail::TriangulatedRegion& region) {
    size_t total_ring_pts = 0;
    std::unordered_set<EdgeKey, EdgeKeyHash> ring_edges;
    for (const auto& group : region.polygon_groups) {
        for (const auto& ring : group) {
            const size_t n = ring.size();
            ASSERT_GE(n, 3u);
            for (size_t i = 0; i < n; ++i) {
                size_t j = (i + 1) % n;
                ring_edges.insert(MakeEdgeKey(static_cast<uint32_t>(total_ring_pts + i),
                                              static_cast<uint32_t>(total_ring_pts + j)));
            }
            total_ring_pts += n;
        }
    }
    ASSERT_EQ(region.vertices.size(), total_ring_pts);

    std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_use;
    for (const Vec3u& tri : region.triangles) {
        ++edge_use[MakeEdgeKey(tri.x, tri.y)];
        ++edge_use[MakeEdgeKey(tri.y, tri.z)];
        ++edge_use[MakeEdgeKey(tri.z, tri.x)];
    }

    for (const auto& [edge, count] : edge_use) {
        if (ring_edges.count(edge) > 0) {
            EXPECT_EQ(count, 1) << "ring edge (" << edge.a << "," << edge.b
                                << ") must be used exactly once by the cap";
        } else {
            EXPECT_EQ(count, 2) << "interior edge (" << edge.a << "," << edge.b
                                << ") must be used exactly twice";
        }
    }
    for (const EdgeKey& edge : ring_edges) {
        EXPECT_EQ(edge_use.count(edge), 1u)
            << "ring edge (" << edge.a << "," << edge.b << ") missing from cap triangulation";
    }
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
    EXPECT_EQ(metrics.open_edges, 0u);
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
    EXPECT_EQ(metrics.open_edges, 0u);
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

// ---------------------------------------------------------------------------
// Triangulation regression tests (dirty ring inputs)
// ---------------------------------------------------------------------------

TEST(VectorTriangulate, DirtyRingWithDuplicateAndCollinearPointsStaysExtrudable) {
    // 10mm square (Clipper scale 1e5) with a collinear midpoint on the bottom
    // edge and a duplicated corner. earcut filters such points internally; the
    // triangulator must strip them from the rings too, or cap boundary edges
    // desynchronize from the side walls and become open edges.
    Clipper2Lib::Paths64 paths;
    paths.push_back(Clipper2Lib::Path64{
        {0, 0}, {500000, 0}, {1000000, 0}, {1000000, 1000000}, {1000000, 1000000}, {0, 1000000}});

    detail::TriangulatedRegion region = detail::TriangulateMergedPaths(paths);
    ASSERT_EQ(region.polygon_groups.size(), 1u);
    ASSERT_EQ(region.polygon_groups[0].size(), 1u);
    EXPECT_EQ(region.polygon_groups[0][0].size(), 4u); // cleaned to a plain square
    ExpectRegionExtrudable(region);
}

TEST(VectorTriangulate, ZeroAreaRingIsDroppedWithoutOrphanGroup) {
    // A fully collinear ring collapses during cleaning / yields no earcut
    // triangles. It must not leave a polygon group or vertices behind: an
    // uncapped group would make ExtrudeSlab emit a side-wall tube whose two
    // rims are entirely open edges.
    Clipper2Lib::Paths64 paths;
    paths.push_back(Clipper2Lib::Path64{{0, 0}, {1000000, 0}, {2000000, 0}});

    detail::TriangulatedRegion region = detail::TriangulateMergedPaths(paths);
    EXPECT_TRUE(region.polygon_groups.empty());
    EXPECT_TRUE(region.vertices.empty());
    EXPECT_TRUE(region.triangles.empty());
}

TEST(VectorTriangulate, Float32QuantizationCollapseIsCleaned) {
    // At x = 200mm the float32 ulp (~1.5e-5mm) exceeds the Clipper grid step
    // (1e-5mm): the two distinct int64 points 20000001 and 20000002 round to
    // the same float, creating a duplicate vertex after conversion. This is
    // the real-world source of dirty rings on large canvases.
    Clipper2Lib::Paths64 paths;
    paths.push_back(Clipper2Lib::Path64{
        {0, 0}, {20000001, 0}, {20000002, 0}, {20000002, 1000000}, {0, 1000000}});

    detail::TriangulatedRegion region = detail::TriangulateMergedPaths(paths);
    ASSERT_EQ(region.polygon_groups.size(), 1u);
    EXPECT_EQ(region.polygon_groups[0][0].size(), 4u); // duplicate collapsed
    ExpectRegionExtrudable(region);
}

TEST(VectorTriangulate, SquareWithDirtyHoleStaysExtrudable) {
    Clipper2Lib::Paths64 paths;
    // Outer 20mm square with a duplicated corner.
    paths.push_back(Clipper2Lib::Path64{
        {0, 0}, {2000000, 0}, {2000000, 2000000}, {2000000, 2000000}, {0, 2000000}});
    // 10mm hole with a collinear midpoint on its right edge.
    paths.push_back(Clipper2Lib::Path64{{500000, 500000},
                                        {1500000, 500000},
                                        {1500000, 1000000},
                                        {1500000, 1500000},
                                        {500000, 1500000}});

    detail::TriangulatedRegion region = detail::TriangulateMergedPaths(paths);
    ASSERT_EQ(region.polygon_groups.size(), 1u);
    ASSERT_EQ(region.polygon_groups[0].size(), 2u); // outer + hole
    EXPECT_EQ(region.polygon_groups[0][0].size(), 4u);
    EXPECT_EQ(region.polygon_groups[0][1].size(), 4u);
    ExpectRegionExtrudable(region);
}

TEST(VectorMesh, ContourWithDuplicateAndCollinearPointsIsWatertight) {
    // End-to-end guard: dirty input contours must still produce a mesh with
    // zero open / non-manifold edges under index-based diagnostics.
    VectorShape shape;
    shape.contours.push_back(Contour{
        {0.0f, 0.0f}, {5.0f, 0.0f}, {10.0f, 0.0f}, {10.0f, 10.0f}, {10.0f, 10.0f}, {0.0f, 10.0f}});

    std::vector<VectorShape> shapes{shape};
    VectorRecipeMap map = BuildSingleChannelRecipeMap(1);

    VectorMeshConfig cfg;
    cfg.layer_height_mm = 0.2f;

    std::vector<Mesh> meshes = BuildVectorMeshes(shapes, map, cfg);
    ASSERT_EQ(meshes.size(), 1u);
    ASSERT_FALSE(meshes[0].indices.empty());

    MeshTopologyMetrics metrics = AnalyzeMesh(meshes[0]);
    EXPECT_EQ(metrics.open_edges, 0u);
    EXPECT_EQ(metrics.non_manifold_edges, 0u);
    EXPECT_EQ(metrics.duplicate_faces, 0u);
    EXPECT_EQ(metrics.degenerate_triangles, 0u);
}

TEST(VectorMesh, HairlineSliverShapeIsWatertightOrDropped) {
    // A 20mm × 1µm sliver survives Clipper simplification but produces
    // near-zero-area cap triangles. They must be kept: dropping them (the old
    // float degenerate-area filter) punched holes in the caps while the side
    // walls still referenced those boundary edges.
    VectorShape shape;
    shape.contours.push_back(MakeRect(0.0f, 0.0f, 20.0f, 0.001f));

    std::vector<VectorShape> shapes{shape};
    VectorRecipeMap map = BuildSingleChannelRecipeMap(1);

    VectorMeshConfig cfg;
    cfg.layer_height_mm = 0.2f;

    std::vector<Mesh> meshes = BuildVectorMeshes(shapes, map, cfg);
    ASSERT_EQ(meshes.size(), 1u);

    MeshTopologyMetrics metrics = AnalyzeMesh(meshes[0]);
    EXPECT_EQ(metrics.open_edges, 0u);
    EXPECT_EQ(metrics.non_manifold_edges, 0u);
    EXPECT_EQ(metrics.duplicate_faces, 0u);
}
