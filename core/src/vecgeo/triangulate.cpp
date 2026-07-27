#include "triangulate.h"

#include "chromaprint3d/error.h"

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

uint32_t CheckedU32Index(std::size_t value) {
    if (value > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        throw InputError("Triangulated vertex index exceeds uint32_t range");
    }
    return static_cast<uint32_t>(value);
}

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

// Double-precision cross product of (p→q) × (q→r). Matches the arithmetic
// used by earcut's internal `area()` check exactly: float inputs widen to
// double, both products are exact (24-bit mantissas), so a zero result here
// is zero inside earcut too.
double CrossD(const Vec2f& p, const Vec2f& q, const Vec2f& r) {
    return (static_cast<double>(q.y) - static_cast<double>(p.y)) *
               (static_cast<double>(r.x) - static_cast<double>(q.x)) -
           (static_cast<double>(q.x) - static_cast<double>(p.x)) *
               (static_cast<double>(r.y) - static_cast<double>(q.y));
}

// Remove consecutive duplicate points and exactly-collinear points, using the
// same predicate as earcut's `filterPoints` (duplicate successor, or
// double-precision cross == 0). earcut silently drops such points from its
// triangulation, so any point left in the ring but unused by earcut would
// desynchronize cap boundary edges from the side walls ExtrudeSlab builds
// from this very ring — producing open edges. Cleaning up-front keeps both
// consumers in lockstep. Rings reduced below 3 points are emptied.
void CleanRing(Contour& ring) {
    bool changed = true;
    while (changed && ring.size() >= 3) {
        changed = false;
        for (size_t i = 0; i < ring.size() && ring.size() >= 3;) {
            const size_t n    = ring.size();
            const Vec2f& prev = ring[(i + n - 1) % n];
            const Vec2f& cur  = ring[i];
            const Vec2f& next = ring[(i + 1) % n];
            bool duplicate    = (cur.x == next.x && cur.y == next.y);
            if (duplicate || CrossD(prev, cur, next) == 0.0) {
                ring.erase(ring.begin() + static_cast<std::ptrdiff_t>(i));
                changed = true;
            } else {
                ++i;
            }
        }
    }
    if (ring.size() < 3) { ring.clear(); }
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

    // Build candidate groups: each group = [outer, hole1, hole2, ...].
    // Rings are cleaned right after float conversion so triangulation and
    // wall extrusion later operate on the identical point set.
    std::vector<std::vector<Contour>> groups;
    std::vector<int> outer_to_group(rings.size(), -1);
    for (int oid : outer_ids) {
        Contour outer_contour = Path64ToContour(rings[static_cast<size_t>(oid)].path);
        CleanRing(outer_contour);
        if (outer_contour.size() < 3) continue;
        outer_to_group[static_cast<size_t>(oid)] = static_cast<int>(groups.size());
        groups.push_back({std::move(outer_contour)});
    }
    for (size_t i = 0; i < rings.size(); ++i) {
        if (rings[i].is_outer || rings[i].owner_outer < 0) continue;
        int group_idx = outer_to_group[static_cast<size_t>(rings[i].owner_outer)];
        if (group_idx < 0) continue;
        Contour hole_contour = Path64ToContour(rings[i].path);
        CleanRing(hole_contour);
        if (hole_contour.size() < 3) continue;
        groups[static_cast<size_t>(group_idx)].push_back(std::move(hole_contour));
    }

    // Triangulate each group with earcut. A group is committed to the result
    // only when it yields triangles: committing a group without caps would
    // make ExtrudeSlab emit side walls around an open tube (open edges).
    // Every earcut triangle is kept — zero-area triangles with distinct
    // indices are topologically required for watertightness, and slicers'
    // index-based diagnostics do not flag them.
    for (auto& group : groups) {
        if (group.empty()) continue;

        size_t total_pts = 0;
        for (const auto& ring : group) { total_pts += ring.size(); }

        std::vector<uint32_t> indices = mapbox::earcut<uint32_t>(group);
        if (indices.empty()) continue;

        size_t base = result.vertices.size();
        result.vertices.reserve(base + total_pts);
        for (const auto& ring : group) {
            for (const Vec2f& p : ring) result.vertices.push_back(p);
        }

        result.triangles.reserve(result.triangles.size() + indices.size() / 3);
        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
            result.triangles.emplace_back(CheckedU32Index(base + indices[i]),
                                          CheckedU32Index(base + indices[i + 1]),
                                          CheckedU32Index(base + indices[i + 2]));
        }
        result.polygon_groups.push_back(std::move(group));
    }

    return result;
}

} // namespace ChromaPrint3D::detail
