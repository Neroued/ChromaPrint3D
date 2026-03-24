#include "triangulate.h"

#include <earcut/earcut.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace mapbox::util {

template <>
struct nth<0, ChromaPrint3D::Vec2f> {
    static auto get(const ChromaPrint3D::Vec2f& p) { return p.x; }
};

template <>
struct nth<1, ChromaPrint3D::Vec2f> {
    static auto get(const ChromaPrint3D::Vec2f& p) { return p.y; }
};

} // namespace mapbox::util

namespace ChromaPrint3D::detail {

namespace {

constexpr double kClipperScale = 100000.0;

Contour Path64ToContour(const Clipper2Lib::Path64& path) {
    Contour c;
    c.reserve(path.size());
    for (const auto& pt : path) {
        c.push_back(Vec2f(static_cast<float>(pt.x / kClipperScale),
                          static_cast<float>(pt.y / kClipperScale)));
    }
    return c;
}

struct RingInfo {
    Clipper2Lib::Path64 path;
    double abs_area = 0.0;
    bool is_outer   = true;
    int owner_outer = -1;
};

constexpr float kDegenerateAreaThreshold = 1e-10f;

bool IsDegenerateTriangle2D(const Vec2f& a, const Vec2f& b, const Vec2f& c) {
    float cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    return std::abs(cross) < kDegenerateAreaThreshold;
}

} // namespace

TriangulatedRegion TriangulateMergedPaths(const Clipper2Lib::Paths64& paths) {
    TriangulatedRegion result;
    if (paths.empty()) return result;

    std::vector<RingInfo> rings;
    rings.reserve(paths.size());
    for (const auto& path : paths) {
        if (path.size() < 3) continue;
        RingInfo info;
        info.path     = path;
        info.abs_area = std::abs(Clipper2Lib::Area(info.path));
        rings.push_back(std::move(info));
    }
    if (rings.empty()) return result;

    // Classify outer/hole by containment depth (even = outer, odd = hole).
    for (size_t i = 0; i < rings.size(); ++i) {
        int depth = 0;
        for (size_t j = 0; j < rings.size(); ++j) {
            if (i == j) continue;
            if (Clipper2Lib::Path2ContainsPath1(rings[i].path, rings[j].path)) ++depth;
        }
        rings[i].is_outer = (depth % 2 == 0);
    }

    // Normalize winding: outer → positive area, hole → negative area.
    for (auto& ring : rings) {
        bool is_positive = Clipper2Lib::Area(ring.path) > 0;
        if ((ring.is_outer && !is_positive) || (!ring.is_outer && is_positive)) {
            std::reverse(ring.path.begin(), ring.path.end());
        }
        ring.abs_area = std::abs(Clipper2Lib::Area(ring.path));
    }

    std::vector<int> outer_ids;
    outer_ids.reserve(rings.size());
    for (size_t i = 0; i < rings.size(); ++i) {
        if (rings[i].is_outer) outer_ids.push_back(static_cast<int>(i));
    }

    // Assign each hole to the smallest containing outer ring.
    for (size_t i = 0; i < rings.size(); ++i) {
        if (rings[i].is_outer) continue;
        double best_area = std::numeric_limits<double>::infinity();
        int best_outer   = -1;
        for (int oid : outer_ids) {
            if (!Clipper2Lib::Path2ContainsPath1(rings[i].path,
                                                 rings[static_cast<size_t>(oid)].path)) {
                continue;
            }
            double outer_area = rings[static_cast<size_t>(oid)].abs_area;
            if (outer_area < best_area) {
                best_area  = outer_area;
                best_outer = oid;
            }
        }
        if (best_outer < 0) {
            // Orphan hole: promote to standalone outer ring.
            if (Clipper2Lib::Area(rings[i].path) < 0) {
                std::reverse(rings[i].path.begin(), rings[i].path.end());
            }
            rings[i].is_outer = true;
            outer_ids.push_back(static_cast<int>(i));
        } else {
            rings[i].owner_outer = best_outer;
        }
    }

    // Build polygon_groups: each group = [outer, hole1, hole2, ...]
    std::vector<int> outer_to_group(rings.size(), -1);
    for (int oid : outer_ids) {
        Contour outer_contour = Path64ToContour(rings[static_cast<size_t>(oid)].path);
        if (outer_contour.size() < 3) continue;
        outer_to_group[static_cast<size_t>(oid)] = static_cast<int>(result.polygon_groups.size());
        result.polygon_groups.push_back({std::move(outer_contour)});
    }
    for (size_t i = 0; i < rings.size(); ++i) {
        if (rings[i].is_outer || rings[i].owner_outer < 0) continue;
        int group_idx = outer_to_group[static_cast<size_t>(rings[i].owner_outer)];
        if (group_idx < 0) continue;
        Contour hole_contour = Path64ToContour(rings[i].path);
        if (hole_contour.size() < 3) continue;
        result.polygon_groups[static_cast<size_t>(group_idx)].push_back(std::move(hole_contour));
    }

    // Triangulate each group with earcut.
    for (const auto& group : result.polygon_groups) {
        if (group.empty()) continue;

        std::vector<std::vector<Vec2f>> polygon;
        polygon.reserve(group.size());
        size_t total_pts = 0;
        for (const auto& ring : group) {
            total_pts += ring.size();
            polygon.push_back(ring);
        }
        if (polygon.empty()) continue;

        std::vector<uint32_t> indices = mapbox::earcut<uint32_t>(polygon);

        size_t base = result.vertices.size();
        result.vertices.reserve(base + total_pts);
        for (const auto& ring : polygon) {
            for (const Vec2f& p : ring) result.vertices.push_back(p);
        }

        if (indices.empty()) continue;

        result.triangles.reserve(result.triangles.size() + indices.size() / 3);
        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
            uint32_t i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
            const auto& v0 = result.vertices[base + i0];
            const auto& v1 = result.vertices[base + i1];
            const auto& v2 = result.vertices[base + i2];
            if (IsDegenerateTriangle2D(v0, v1, v2)) continue;
            result.triangles.emplace_back(static_cast<int>(base + i0), static_cast<int>(base + i1),
                                          static_cast<int>(base + i2));
        }
    }

    return result;
}

} // namespace ChromaPrint3D::detail
