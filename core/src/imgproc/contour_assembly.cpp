#include "imgproc/contour_assembly.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace ChromaPrint3D::detail {

namespace {

double PolylineSignedArea(const std::vector<Vec2f>& pts) {
    if (pts.size() < 3) return 0.0;
    double acc = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        const Vec2f& a = pts[i];
        const Vec2f& b = pts[(i + 1) % pts.size()];
        acc += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
    }
    return 0.5 * acc;
}

BezierContour MakeDegenerateBezierContour(const std::vector<Vec2f>& pts, bool closed) {
    BezierContour bc;
    bc.closed = closed;
    bc.segments.reserve(pts.size());
    size_t n = closed ? pts.size() : pts.size() - 1;
    for (size_t i = 0; i < n; ++i) {
        Vec2f a = pts[i];
        Vec2f b = pts[(i + 1) % pts.size()];
        Vec2f d = b - a;
        if (d.LengthSquared() < 1e-10f) continue;
        bc.segments.push_back({a, a + d * (1.0f / 3.0f), a + d * (2.0f / 3.0f), b});
    }
    return bc;
}

BezierContour PointsToBezierContour(const std::vector<Vec2f>& pts, bool closed,
                                    const CurveFitConfig* fit_cfg) {
    BezierContour bc;
    bc.closed = closed;
    if (pts.size() < 2) return bc;

    if (fit_cfg && pts.size() >= 3) {
        auto fitted =
            closed ? FitBezierToClosedPolyline(pts, *fit_cfg) : FitBezierToPolyline(pts, *fit_cfg);
        if (!fitted.empty()) {
            // Area-ratio sanity check for closed contours
            bool valid = true;
            if (closed) {
                BezierContour tmp;
                tmp.closed       = true;
                tmp.segments     = fitted;
                double bez_area  = std::abs(BezierContourSignedArea(tmp));
                double poly_area = std::abs(PolylineSignedArea(pts));
                if (poly_area > 1.0 && (bez_area < poly_area * 0.3 || bez_area > poly_area * 3.0)) {
                    valid = false;
                    spdlog::debug("PointsToBezierContour fallback: invalid area ratio, points={}, "
                                  "poly_area={:.3f}, bezier_area={:.3f}",
                                  pts.size(), poly_area, bez_area);
                }
            }
            if (valid) {
                bc.segments = std::move(fitted);
                return bc;
            }
        }
    }

    return MakeDegenerateBezierContour(pts, closed);
}

bool PointInPolygon(const Vec2f& p, const std::vector<Vec2f>& poly) {
    int n       = static_cast<int>(poly.size());
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        float yi = poly[i].y, yj = poly[j].y;
        float xi = poly[i].x, xj = poly[j].x;
        if (((yi > p.y) != (yj > p.y)) && (p.x < (xj - xi) * (p.y - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

struct OrientedEdgeRef {
    int edge_id;
    bool reversed;
};

// For a given label, collect all edge references where this label appears,
// oriented so that the label is on the left side when walking the edge forward.
std::vector<OrientedEdgeRef> CollectEdgesForLabel(const BoundaryGraph& graph, int label) {
    std::vector<OrientedEdgeRef> refs;
    for (int i = 0; i < static_cast<int>(graph.edges.size()); ++i) {
        const auto& e = graph.edges[i];
        if (e.label_left == label) {
            refs.push_back({i, false});
        } else if (e.label_right == label) {
            refs.push_back({i, true});
        }
    }
    return refs;
}

Vec2f EdgeStartPoint(const BoundaryGraph& graph, const OrientedEdgeRef& ref) {
    const auto& e = graph.edges[ref.edge_id];
    return ref.reversed ? e.points.back() : e.points.front();
}

Vec2f EdgeEndPoint(const BoundaryGraph& graph, const OrientedEdgeRef& ref) {
    const auto& e = graph.edges[ref.edge_id];
    return ref.reversed ? e.points.front() : e.points.back();
}

int EdgeStartNode(const BoundaryGraph& graph, const OrientedEdgeRef& ref) {
    const auto& e = graph.edges[ref.edge_id];
    return ref.reversed ? e.node_end : e.node_start;
}

int EdgeEndNode(const BoundaryGraph& graph, const OrientedEdgeRef& ref) {
    const auto& e = graph.edges[ref.edge_id];
    return ref.reversed ? e.node_start : e.node_end;
}

void AppendEdgePoints(const BoundaryGraph& graph, const OrientedEdgeRef& ref,
                      std::vector<Vec2f>& out, bool skip_first) {
    const auto& pts = graph.edges[ref.edge_id].points;
    if (pts.empty()) return;
    if (ref.reversed) {
        int start =
            skip_first ? static_cast<int>(pts.size()) - 2 : static_cast<int>(pts.size()) - 1;
        for (int i = start; i >= 0; --i) { out.push_back(pts[i]); }
    } else {
        size_t start = skip_first ? 1 : 0;
        for (size_t i = start; i < pts.size(); ++i) { out.push_back(pts[i]); }
    }
}

// Chain oriented edges into closed contour loops for a single label.
std::vector<std::vector<Vec2f>> ChainEdgesIntoLoops(const BoundaryGraph& graph,
                                                    const std::vector<OrientedEdgeRef>& refs) {
    std::vector<std::vector<Vec2f>> loops;
    if (refs.empty()) return loops;

    // Build adjacency: for each end-node, which edges depart from it?
    std::unordered_map<int, std::vector<int>> node_to_refs;
    for (int i = 0; i < static_cast<int>(refs.size()); ++i) {
        int sn = EdgeStartNode(graph, refs[i]);
        node_to_refs[sn].push_back(i);
    }

    std::vector<bool> used(refs.size(), false);

    for (int seed = 0; seed < static_cast<int>(refs.size()); ++seed) {
        if (used[seed]) continue;

        std::vector<Vec2f> loop;
        int cur = seed;
        bool ok = true;

        while (true) {
            if (used[cur]) {
                ok = (cur == seed && !loop.empty());
                break;
            }
            used[cur] = true;
            AppendEdgePoints(graph, refs[cur], loop, !loop.empty());

            int end_node = EdgeEndNode(graph, refs[cur]);
            auto it      = node_to_refs.find(end_node);
            if (it == node_to_refs.end()) {
                ok = false;
                break;
            }

            int next = -1;
            for (int ri : it->second) {
                if (!used[ri]) {
                    next = ri;
                    break;
                }
            }
            if (next < 0) {
                // Check if we've closed the loop back to seed
                if (EdgeStartNode(graph, refs[seed]) == end_node && !loop.empty()) { ok = true; }
                break;
            }
            cur = next;
        }

        if (!ok || loop.size() < 3) continue;

        // Remove duplicate closing point if present
        if (loop.size() > 1 && (loop.front() - loop.back()).LengthSquared() < 1e-6f) {
            loop.pop_back();
        }
        if (loop.size() >= 3) { loops.push_back(std::move(loop)); }
    }
    return loops;
}

// Remove near-collinear interior points to clean up residual zigzag
// from the crack grid after subpixel refinement.
void DecimateNearCollinear(std::vector<Vec2f>& pts, float epsilon) {
    constexpr int kMinPoints = 6;
    constexpr int kMaxPasses = 3;
    const float eps_sq       = epsilon * epsilon;

    for (int pass = 0; pass < kMaxPasses; ++pass) {
        int n = static_cast<int>(pts.size());
        if (n <= kMinPoints) break;

        std::vector<bool> remove(static_cast<size_t>(n), false);
        bool prev_removed = false;
        int count         = 0;

        for (int i = 0; i < n; ++i) {
            if (prev_removed) {
                prev_removed = false;
                continue;
            }
            int prev        = ((i - 1) % n + n) % n;
            int next        = (i + 1) % n;
            Vec2f ab        = pts[static_cast<size_t>(next)] - pts[static_cast<size_t>(prev)];
            float ab_len_sq = ab.LengthSquared();
            if (ab_len_sq < 1e-12f) continue;
            Vec2f ap      = pts[static_cast<size_t>(i)] - pts[static_cast<size_t>(prev)];
            float cross   = ap.x * ab.y - ap.y * ab.x;
            float dist_sq = (cross * cross) / ab_len_sq;

            if (dist_sq < eps_sq && (n - count) > kMinPoints) {
                remove[static_cast<size_t>(i)] = true;
                prev_removed                   = true;
                ++count;
            }
        }

        if (count == 0) break;

        std::vector<Vec2f> result;
        result.reserve(static_cast<size_t>(n - count));
        for (int i = 0; i < n; ++i) {
            if (!remove[static_cast<size_t>(i)]) result.push_back(pts[static_cast<size_t>(i)]);
        }
        pts = std::move(result);
    }
}

float LocalCurvature(const std::vector<Vec2f>& pts, int i, int n) {
    int im1    = ((i - 1) % n + n) % n;
    int ip1    = (i + 1) % n;
    Vec2f v1   = pts[i] - pts[im1];
    Vec2f v2   = pts[ip1] - pts[i];
    float len1 = v1.Length();
    float len2 = v2.Length();
    if (len1 < 1e-6f || len2 < 1e-6f) return 0.0f;
    float cross = std::abs(v1.x * v2.y - v1.y * v2.x);
    return cross / (len1 * len2);
}

void SmoothOpenChain(std::vector<Vec2f>& pts, float max_displacement, int iterations) {
    if (pts.size() < 5) return;
    const int n = static_cast<int>(pts.size());

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Vec2f> prev = pts;
        for (int i = 2; i < n - 2; ++i) {
            Vec2f s = (prev[i - 2] + prev[i - 1] * 4.0f + prev[i] * 6.0f + prev[i + 1] * 4.0f +
                       prev[i + 2]) *
                      (1.0f / 16.0f);
            Vec2f delta = s - prev[i];
            float dist  = delta.Length();
            if (dist > max_displacement) s = prev[i] + delta * (max_displacement / dist);
            pts[i] = s;
        }
    }
}

void SmoothClosedLoop(std::vector<Vec2f>& pts, float max_displacement, int iterations) {
    if (pts.size() < 5) return;
    const int n = static_cast<int>(pts.size());

    constexpr float kHighCurvature = 0.5f;

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Vec2f> prev_pts = pts;
        std::vector<Vec2f> smoothed(n);
        for (int i = 0; i < n; ++i) {
            int im2 = ((i - 2) % n + n) % n;
            int im1 = ((i - 1) % n + n) % n;
            int ip1 = (i + 1) % n;
            int ip2 = (i + 2) % n;
            smoothed[i] =
                (pts[im2] + pts[im1] * 4.0f + pts[i] * 6.0f + pts[ip1] * 4.0f + pts[ip2]) *
                (1.0f / 16.0f);
        }
        for (int i = 0; i < n; ++i) {
            float curv        = LocalCurvature(prev_pts, i, n);
            float attenuation = (curv > kHighCurvature) ? std::max(0.1f, 1.0f - curv) : 1.0f;
            float local_max   = max_displacement * attenuation;

            Vec2f delta = smoothed[i] - prev_pts[i];
            float dist  = delta.Length();
            if (dist > local_max) { smoothed[i] = prev_pts[i] + delta * (local_max / dist); }
        }
        pts = std::move(smoothed);
    }
}

CubicBezier ReverseBezierSegment(const CubicBezier& seg) {
    return {seg.p3, seg.p2, seg.p1, seg.p0};
}

std::vector<CubicBezier> ReverseBezierChain(const std::vector<CubicBezier>& chain) {
    std::vector<CubicBezier> rev;
    rev.reserve(chain.size());
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        rev.push_back(ReverseBezierSegment(*it));
    }
    return rev;
}

struct EdgeRefLoop {
    std::vector<OrientedEdgeRef> refs;
};

std::vector<EdgeRefLoop> ChainEdgeRefsIntoLoops(const BoundaryGraph& graph,
                                                const std::vector<OrientedEdgeRef>& refs) {
    std::vector<EdgeRefLoop> loops;
    if (refs.empty()) return loops;

    std::unordered_map<int, std::vector<int>> node_to_refs;
    for (int i = 0; i < static_cast<int>(refs.size()); ++i) {
        int sn = EdgeStartNode(graph, refs[i]);
        node_to_refs[sn].push_back(i);
    }

    std::vector<bool> used(refs.size(), false);

    for (int seed = 0; seed < static_cast<int>(refs.size()); ++seed) {
        if (used[seed]) continue;

        EdgeRefLoop loop;
        int cur = seed;
        bool ok = true;

        while (true) {
            if (used[cur]) {
                ok = (cur == seed && !loop.refs.empty());
                break;
            }
            used[cur] = true;
            loop.refs.push_back(refs[cur]);

            int end_node = EdgeEndNode(graph, refs[cur]);
            auto it      = node_to_refs.find(end_node);
            if (it == node_to_refs.end()) {
                ok = false;
                break;
            }

            int next = -1;
            for (int ri : it->second) {
                if (!used[ri]) {
                    next = ri;
                    break;
                }
            }
            if (next < 0) {
                if (EdgeStartNode(graph, refs[seed]) == end_node && !loop.refs.empty()) {
                    ok = true;
                }
                break;
            }
            cur = next;
        }

        if (ok && !loop.refs.empty()) { loops.push_back(std::move(loop)); }
    }
    return loops;
}

double BezierChainSignedArea(const std::vector<CubicBezier>& segs) {
    std::vector<Vec2f> pts;
    pts.reserve(segs.size() * 4);
    for (const auto& s : segs) {
        pts.push_back(s.p0);
        pts.push_back(s.p1);
        pts.push_back(s.p2);
    }
    if (!segs.empty()) pts.push_back(segs.back().p3);
    return PolylineSignedArea(pts);
}

} // namespace

ContourSmoothConfig ContourSmoothFromLevel(float smoothness) {
    float s = std::clamp(smoothness, 0.0f, 1.0f);
    ContourSmoothConfig c;
    c.decimate_epsilon        = 0.05f + 0.35f * s;
    c.smooth_max_displacement = 0.2f + 0.6f * s;
    c.smooth_iterations       = std::max(1, static_cast<int>(std::lround(1.0f + 3.0f * s)));
    return c;
}

std::vector<VectorizedShape> AssembleContoursFromGraph(const BoundaryGraph& graph, int num_labels,
                                                       const std::vector<Rgb>& palette,
                                                       float min_contour_area, float min_hole_area,
                                                       const CurveFitConfig* fit_cfg,
                                                       const ContourSmoothConfig& smooth_cfg,
                                                       float merge_tolerance) {
    std::vector<VectorizedShape> shapes;
    if (graph.edges.empty() || num_labels <= 0) {
        spdlog::debug("AssembleContoursFromGraph skipped: edges={}, num_labels={}",
                      graph.edges.size(), num_labels);
        return shapes;
    }
    const auto start = std::chrono::steady_clock::now();
    spdlog::debug("AssembleContoursFromGraph start: edges={}, num_labels={}", graph.edges.size(),
                  num_labels);

    // ── Phase 1: Pre-smooth each BoundaryEdge independently ─────────────────
    const int num_edges = static_cast<int>(graph.edges.size());
    std::vector<std::vector<Vec2f>> edge_smoothed(num_edges);

    for (int eid = 0; eid < num_edges; ++eid) {
        edge_smoothed[eid] = graph.edges[eid].points;
        if (edge_smoothed[eid].size() < 2) continue;
        DecimateNearCollinear(edge_smoothed[eid], smooth_cfg.decimate_epsilon);
        if (edge_smoothed[eid].size() >= 5) {
            SmoothOpenChain(edge_smoothed[eid], smooth_cfg.smooth_max_displacement,
                            std::max(1, smooth_cfg.smooth_iterations));
        }
    }
    spdlog::debug("Edge pre-smoothing done: edges={}", num_edges);

    // ── Phase 1.5: Per-edge Bezier fitting ──────────────────────────────────
    // Fit Bezier curves to each edge independently so that both labels sharing
    // an edge reference the exact same curve segments — guaranteeing watertight
    // boundaries after assembly.
    std::vector<std::vector<CubicBezier>> edge_beziers(num_edges);
    if (fit_cfg) {
        for (int eid = 0; eid < num_edges; ++eid) {
            const auto& pts = edge_smoothed[eid];
            if (pts.size() < 2) continue;

            bool is_self_loop = (graph.edges[eid].node_start == graph.edges[eid].node_end);
            if (is_self_loop && pts.size() >= 3) {
                auto loop_pts = pts;
                if (loop_pts.size() > 1 &&
                    (loop_pts.front() - loop_pts.back()).LengthSquared() < 1e-6f)
                    loop_pts.pop_back();
                if (loop_pts.size() >= 3)
                    edge_beziers[eid] = FitBezierToClosedPolyline(loop_pts, *fit_cfg);
            } else {
                edge_beziers[eid] = FitBezierToPolyline(pts, *fit_cfg);
            }

            if (edge_beziers[eid].empty()) {
                auto fb           = MakeDegenerateBezierContour(pts, is_self_loop);
                edge_beziers[eid] = std::move(fb.segments);
            }
        }
    } else {
        for (int eid = 0; eid < num_edges; ++eid) {
            const auto& pts = edge_smoothed[eid];
            if (pts.size() < 2) continue;
            bool is_self_loop = (graph.edges[eid].node_start == graph.edges[eid].node_end);
            auto fb           = MakeDegenerateBezierContour(pts, is_self_loop);
            edge_beziers[eid] = std::move(fb.segments);
        }
    }
    spdlog::debug("Per-edge Bezier fitting done: edges={}", num_edges);

    // ── Phase 1.5b: Per-edge MergeNearLinearSegments ────────────────────────
    // Done per-edge (not per-contour) so both sides of a shared boundary see
    // the identical merge result, preserving the watertight guarantee.
    if (merge_tolerance > 0.0f) {
        int merged_total = 0;
        for (int eid = 0; eid < num_edges; ++eid) {
            int before = static_cast<int>(edge_beziers[eid].size());
            MergeNearLinearSegments(edge_beziers[eid], merge_tolerance);
            merged_total += before - static_cast<int>(edge_beziers[eid].size());
        }
        if (merged_total > 0)
            spdlog::debug("Per-edge MergeNearLinear: removed {} segments", merged_total);
    }

    // ── Phase 2: Per-label assembly from pre-fitted Bezier segments ─────────
    for (int label = 0; label < num_labels; ++label) {
        auto refs = CollectEdgesForLabel(graph, label);
        if (refs.empty()) continue;

        auto ref_loops = ChainEdgeRefsIntoLoops(graph, refs);
        if (ref_loops.empty()) {
            spdlog::warn("AssembleContoursFromGraph: refs found but no loops, label={}, refs={}",
                         label, refs.size());
            continue;
        }

        struct ClassifiedContour {
            BezierContour contour;
            std::vector<Vec2f> polygon;
            double signed_area;
            double abs_area;
            double original_signed_area;
        };

        std::vector<ClassifiedContour> classified;
        for (auto& rl : ref_loops) {
            BezierContour contour;
            contour.closed = true;
            for (const auto& ref : rl.refs) {
                const auto& segs = edge_beziers[ref.edge_id];
                if (ref.reversed) {
                    for (int k = static_cast<int>(segs.size()) - 1; k >= 0; --k)
                        contour.segments.push_back(
                            {segs[k].p3, segs[k].p2, segs[k].p1, segs[k].p0});
                } else {
                    for (const auto& s : segs) contour.segments.push_back(s);
                }
            }
            if (contour.segments.empty()) continue;
            contour.segments.back().p3 = contour.segments.front().p0;

            double sa = BezierContourSignedArea(contour);

            std::vector<Vec2f> poly;
            poly.reserve(contour.segments.size());
            for (const auto& seg : contour.segments) poly.push_back(seg.p0);

            classified.push_back({std::move(contour), std::move(poly), sa, std::abs(sa), sa});
        }

        if (classified.empty()) continue;

        for (auto& cl : classified) {
            if (cl.signed_area < 0) {
                ReverseBezierContour(cl.contour);
                std::reverse(cl.polygon.begin(), cl.polygon.end());
                cl.signed_area = -cl.signed_area;
            }
        }

        std::sort(classified.begin(), classified.end(),
                  [](const auto& a, const auto& b) { return a.abs_area > b.abs_area; });

        struct ContourInfo {
            int parent   = -1;
            bool is_hole = false;
        };

        std::vector<ContourInfo> info(classified.size());

        for (int i = 0; i < static_cast<int>(classified.size()); ++i) {
            bool is_hole_by_winding = classified[i].original_signed_area > 0;
            if (!is_hole_by_winding) continue;
            info[i].is_hole = true;

            Vec2f centroid{0, 0};
            for (const auto& p : classified[i].polygon) centroid = centroid + p;
            centroid = centroid * (1.0f / static_cast<float>(classified[i].polygon.size()));

            for (int j = 0; j < static_cast<int>(classified.size()); ++j) {
                if (j == i || info[j].is_hole) continue;
                if (PointInPolygon(centroid, classified[j].polygon)) {
                    info[i].parent = j;
                    break;
                }
            }
        }

        std::unordered_map<int, std::vector<int>> outer_to_holes;
        std::vector<int> outers;
        for (int i = 0; i < static_cast<int>(classified.size()); ++i) {
            if (!info[i].is_hole) {
                outers.push_back(i);
            } else if (info[i].parent >= 0) {
                outer_to_holes[info[i].parent].push_back(i);
            }
        }

        int dropped_small_outer = 0;
        int dropped_small_hole  = 0;
        int shape_count_out     = 0;
        for (int oi : outers) {
            if (classified[oi].abs_area < static_cast<double>(min_contour_area)) {
                ++dropped_small_outer;
                continue;
            }

            VectorizedShape shape;
            shape.area = classified[oi].abs_area;
            if (label >= 0 && label < static_cast<int>(palette.size())) {
                shape.color = palette[label];
            }

            if (classified[oi].contour.segments.empty()) continue;
            shape.contours.push_back(std::move(classified[oi].contour));

            auto hit = outer_to_holes.find(oi);
            if (hit != outer_to_holes.end()) {
                for (int hi : hit->second) {
                    if (classified[hi].abs_area < static_cast<double>(min_hole_area)) {
                        ++dropped_small_hole;
                        continue;
                    }
                    ReverseBezierContour(classified[hi].contour);
                    if (!classified[hi].contour.segments.empty()) {
                        shape.contours.push_back(std::move(classified[hi].contour));
                    }
                }
            }

            shapes.push_back(std::move(shape));
            ++shape_count_out;
        }
        spdlog::debug("AssembleContoursFromGraph label={}: refs={}, loops={}, outers={}, "
                      "dropped_outer={}, dropped_hole={}, shapes={}",
                      label, refs.size(), ref_loops.size(), outers.size(), dropped_small_outer,
                      dropped_small_hole, shape_count_out);
    }
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    spdlog::info("AssembleContoursFromGraph done: shapes={}, elapsed_ms={:.2f}", shapes.size(),
                 elapsed_ms);
    return shapes;
}

} // namespace ChromaPrint3D::detail
