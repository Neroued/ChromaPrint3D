#include "imgproc/curve_fitting.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace ChromaPrint3D::detail {

namespace {

constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;

float AngleBetween(Vec2f a, Vec2f b) {
    float dot   = a.Dot(b);
    float cross = a.Cross(b);
    return std::atan2(cross, dot);
}

Vec2f EstimateTangent(const std::vector<Vec2f>& pts, int idx, bool forward) {
    int n = static_cast<int>(pts.size());
    if (n < 2) return {1.0f, 0.0f};
    int window = std::max(3, std::min(n / 4, 10));
    if (forward) {
        int end = std::min(idx + window, n - 1);
        Vec2f d = pts[end] - pts[idx];
        return d.LengthSquared() > 1e-10f ? d.Normalized() : Vec2f{1.0f, 0.0f};
    } else {
        int begin = std::max(idx - window, 0);
        Vec2f d   = pts[begin] - pts[idx];
        return d.LengthSquared() > 1e-10f ? d.Normalized() : Vec2f{-1.0f, 0.0f};
    }
}

std::vector<float> ChordLengthParameterize(const std::vector<Vec2f>& pts) {
    int n = static_cast<int>(pts.size());
    std::vector<float> u(n, 0.0f);
    for (int i = 1; i < n; ++i) { u[i] = u[i - 1] + (pts[i] - pts[i - 1]).Length(); }
    if (u.back() > 1e-8f) {
        float inv = 1.0f / u.back();
        for (int i = 1; i < n; ++i) u[i] *= inv;
    }
    u.back() = 1.0f;
    return u;
}

Vec2f BezierEval(Vec2f p0, Vec2f p1, Vec2f p2, Vec2f p3, float t) {
    float s = 1.0f - t;
    return p0 * (s * s * s) + p1 * (3.0f * s * s * t) + p2 * (3.0f * s * t * t) + p3 * (t * t * t);
}

Vec2f BezierDerivEval(Vec2f p0, Vec2f p1, Vec2f p2, Vec2f p3, float t) {
    float s  = 1.0f - t;
    Vec2f d1 = (p1 - p0) * 3.0f;
    Vec2f d2 = (p2 - p1) * 3.0f;
    Vec2f d3 = (p3 - p2) * 3.0f;
    return d1 * (s * s) + d2 * (2.0f * s * t) + d3 * (t * t);
}

Vec2f BezierDeriv2Eval(Vec2f p0, Vec2f p1, Vec2f p2, Vec2f p3, float t) {
    float s  = 1.0f - t;
    Vec2f d1 = (p2 - p1 * 2.0f + p0) * 6.0f;
    Vec2f d2 = (p3 - p2 * 2.0f + p1) * 6.0f;
    return d1 * s + d2 * t;
}

CubicBezier MakeLinearBezier(Vec2f a, Vec2f b) {
    Vec2f d = b - a;
    return {a, a + d * (1.0f / 3.0f), a + d * (2.0f / 3.0f), b};
}

CubicBezier FitSingleBezier(const std::vector<Vec2f>& pts, const std::vector<float>& u,
                            Vec2f tan_left, Vec2f tan_right) {
    int n = static_cast<int>(pts.size());
    if (n < 2) return MakeLinearBezier(pts.front(), pts.back());
    if (n == 2) return MakeLinearBezier(pts[0], pts[1]);

    Vec2f p0 = pts.front();
    Vec2f p3 = pts.back();

    // Set up the system for alpha1, alpha2
    // B(t) = p0*(1-t)^3 + p1*3*(1-t)^2*t + p2*3*(1-t)*t^2 + p3*t^3
    // where p1 = p0 + alpha1 * tan_left, p2 = p3 + alpha2 * tan_right
    float c00 = 0, c01 = 0, c11 = 0;
    float x0 = 0, x1 = 0;

    for (int i = 0; i < n; ++i) {
        float t  = u[i];
        float s  = 1.0f - t;
        float b1 = 3.0f * s * s * t;
        float b2 = 3.0f * s * t * t;

        Vec2f a1 = tan_left * b1;
        Vec2f a2 = tan_right * b2;

        Vec2f tmp = pts[i] - BezierEval(p0, p0, p3, p3, t);

        c00 += a1.Dot(a1);
        c01 += a1.Dot(a2);
        c11 += a2.Dot(a2);
        x0 += a1.Dot(tmp);
        x1 += a2.Dot(tmp);
    }

    float det = c00 * c11 - c01 * c01;
    float alpha1, alpha2;
    if (std::abs(det) < 1e-12f) {
        float dist = (p3 - p0).Length() / 3.0f;
        alpha1     = dist;
        alpha2     = -dist;
    } else {
        alpha1 = (c11 * x0 - c01 * x1) / det;
        alpha2 = (c00 * x1 - c01 * x0) / det;
    }

    float seg_len = (p3 - p0).Length();
    float eps     = seg_len * 1e-4f;
    if (alpha1 < eps || alpha2 < eps) {
        float dist = seg_len / 3.0f;
        alpha1     = dist;
        alpha2     = dist;
    }

    return {p0, p0 + tan_left * alpha1, p3 + tan_right * alpha2, p3};
}

float ComputeMaxError(const std::vector<Vec2f>& pts, const std::vector<float>& u,
                      const CubicBezier& bez, int& split_idx) {
    float max_err = 0.0f;
    split_idx     = static_cast<int>(pts.size()) / 2;
    for (int i = 1; i < static_cast<int>(pts.size()) - 1; ++i) {
        Vec2f p   = BezierEval(bez.p0, bez.p1, bez.p2, bez.p3, u[i]);
        float err = (p - pts[i]).LengthSquared();
        if (err > max_err) {
            max_err   = err;
            split_idx = i;
        }
    }
    return std::sqrt(max_err);
}

void Reparameterize(const std::vector<Vec2f>& pts, std::vector<float>& u, const CubicBezier& bez) {
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        float t    = u[i];
        Vec2f b    = BezierEval(bez.p0, bez.p1, bez.p2, bez.p3, t);
        Vec2f d1   = BezierDerivEval(bez.p0, bez.p1, bez.p2, bez.p3, t);
        Vec2f d2   = BezierDeriv2Eval(bez.p0, bez.p1, bez.p2, bez.p3, t);
        Vec2f diff = b - pts[i];

        float num   = diff.Dot(d1);
        float denom = d1.Dot(d1) + diff.Dot(d2);
        if (std::abs(denom) > 1e-10f) {
            float nt = t - num / denom;
            u[i]     = std::clamp(nt, 0.0f, 1.0f);
        }
    }
}

void SchneiderFitRecursive(const std::vector<Vec2f>& pts, Vec2f tan_left, Vec2f tan_right,
                           float threshold, int max_depth, int reparam_iters,
                           std::vector<CubicBezier>& out, int depth) {
    int n = static_cast<int>(pts.size());
    if (n < 2) return;
    if (n == 2) {
        out.push_back(MakeLinearBezier(pts[0], pts[1]));
        return;
    }

    if (depth >= max_depth) {
        out.push_back(MakeLinearBezier(pts.front(), pts.back()));
        return;
    }

    auto u   = ChordLengthParameterize(pts);
    auto bez = FitSingleBezier(pts, u, tan_left, tan_right);

    int split_idx;
    float err = ComputeMaxError(pts, u, bez, split_idx);

    if (err <= threshold) {
        out.push_back(bez);
        return;
    }

    // Newton-Raphson reparameterization
    for (int iter = 0; iter < reparam_iters; ++iter) {
        Reparameterize(pts, u, bez);
        bez = FitSingleBezier(pts, u, tan_left, tan_right);
        err = ComputeMaxError(pts, u, bez, split_idx);
        if (err <= threshold) {
            out.push_back(bez);
            return;
        }
    }

    // Split at point of max error and recurse
    split_idx        = std::clamp(split_idx, 1, n - 2);
    Vec2f center_tan = (pts[split_idx + 1] - pts[split_idx - 1]).Normalized();
    if (center_tan.LengthSquared() < 0.5f) center_tan = (pts[split_idx] - pts[0]).Normalized();

    std::vector<Vec2f> left(pts.begin(), pts.begin() + split_idx + 1);
    std::vector<Vec2f> right(pts.begin() + split_idx, pts.end());

    SchneiderFitRecursive(left, tan_left, center_tan * (-1.0f), threshold, max_depth, reparam_iters,
                          out, depth + 1);
    SchneiderFitRecursive(right, center_tan, tan_right, threshold, max_depth, reparam_iters, out,
                          depth + 1);
}

bool IsAngleCornerOpen(const std::vector<Vec2f>& pts, int i, int k, int n, float cos_thresh) {
    int prev_idx = std::max(0, i - k);
    int next_idx = std::min(n - 1, i + k);
    Vec2f v1     = pts[prev_idx] - pts[i];
    Vec2f v2     = pts[next_idx] - pts[i];
    float len1   = v1.Length();
    float len2   = v2.Length();
    if (len1 < 1e-6f || len2 < 1e-6f) return false;
    return v1.Dot(v2) / (len1 * len2) < cos_thresh;
}

bool IsAngleCornerClosed(const std::vector<Vec2f>& pts, int i, int k, int n, float cos_thresh) {
    int prev_idx = ((i - k) % n + n) % n;
    int next_idx = (i + k) % n;
    Vec2f v1     = pts[prev_idx] - pts[i];
    Vec2f v2     = pts[next_idx] - pts[i];
    float len1   = v1.Length();
    float len2   = v2.Length();
    if (len1 < 1e-6f || len2 < 1e-6f) return false;
    return v1.Dot(v2) / (len1 * len2) < cos_thresh;
}

float DiscreteCurvatureOpen(const std::vector<Vec2f>& pts, int i, int n) {
    int im1    = std::max(0, i - 1);
    int ip1    = std::min(n - 1, i + 1);
    Vec2f v1   = pts[i] - pts[im1];
    Vec2f v2   = pts[ip1] - pts[i];
    float len1 = v1.Length();
    float len2 = v2.Length();
    if (len1 < 1e-6f || len2 < 1e-6f) return 0.0f;
    float cross = std::abs(v1.Cross(v2));
    return cross / (len1 * len2);
}

float DiscreteCurvatureClosed(const std::vector<Vec2f>& pts, int i, int n) {
    int im1    = ((i - 1) % n + n) % n;
    int ip1    = (i + 1) % n;
    Vec2f v1   = pts[i] - pts[im1];
    Vec2f v2   = pts[ip1] - pts[i];
    float len1 = v1.Length();
    float len2 = v2.Length();
    if (len1 < 1e-6f || len2 < 1e-6f) return 0.0f;
    float cross = std::abs(v1.Cross(v2));
    return cross / (len1 * len2);
}

Vec2f CornerTangent(const std::vector<Vec2f>& pts, int corner_idx, int n, bool forward,
                    bool closed) {
    constexpr int kSmallWindow = 2;
    if (forward) {
        int next =
            closed ? ((corner_idx + kSmallWindow) % n) : std::min(n - 1, corner_idx + kSmallWindow);
        Vec2f d = pts[next] - pts[corner_idx];
        return d.LengthSquared() > 1e-10f ? d.Normalized() : Vec2f{1.0f, 0.0f};
    } else {
        int prev = closed ? (((corner_idx - kSmallWindow) % n + n) % n)
                          : std::max(0, corner_idx - kSmallWindow);
        Vec2f d  = pts[prev] - pts[corner_idx];
        return d.LengthSquared() > 1e-10f ? d.Normalized() : Vec2f{-1.0f, 0.0f};
    }
}

} // namespace

std::vector<int> DetectCorners(const std::vector<Vec2f>& pts, const CurveFitConfig& cfg) {
    int n = static_cast<int>(pts.size());
    if (n < 3) return {};

    float cos_thresh = std::cos(cfg.corner_angle_threshold_deg * kDegToRad);
    int k_base       = std::max(1, cfg.corner_neighbor_k);

    std::vector<bool> is_corner(n, false);

    // Multi-scale angle detection
    std::vector<int> scales;
    if (cfg.enable_multiscale_corners) {
        scales = {std::max(1, k_base / 2), k_base, std::min(n / 2, k_base * 2)};
    } else {
        scales = {k_base};
    }

    for (int k : scales) {
        for (int i = 1; i < n - 1; ++i) {
            if (is_corner[i]) continue;
            if (IsAngleCornerOpen(pts, i, k, n, cos_thresh)) { is_corner[i] = true; }
        }
    }

    // Curvature-jump detection
    if (cfg.enable_curvature_corners && n >= 5) {
        std::vector<float> curv(n, 0.0f);
        for (int i = 1; i < n - 1; ++i) { curv[i] = DiscreteCurvatureOpen(pts, i, n); }

        for (int i = 2; i < n - 2; ++i) {
            if (is_corner[i]) continue;
            float jump = std::abs(curv[i] - curv[i - 1]) + std::abs(curv[i] - curv[i + 1]);
            if (jump > cfg.curvature_jump_threshold && curv[i] > 0.3f) { is_corner[i] = true; }
        }
    }

    std::vector<int> corners;
    for (int i = 1; i < n - 1; ++i) {
        if (is_corner[i]) corners.push_back(i);
    }
    return corners;
}

std::vector<CubicBezier> FitBezierToPolyline(const std::vector<Vec2f>& pts,
                                             const CurveFitConfig& cfg) {
    std::vector<CubicBezier> result;
    int n = static_cast<int>(pts.size());
    if (n < 2) return result;
    if (n == 2) {
        result.push_back(MakeLinearBezier(pts[0], pts[1]));
        return result;
    }

    auto corners = DetectCorners(pts, cfg);

    std::vector<int> splits;
    splits.push_back(0);
    for (int c : corners) {
        if (c > splits.back()) splits.push_back(c);
    }
    if (splits.back() != n - 1) { splits.push_back(n - 1); }

    std::vector<bool> split_is_corner(splits.size(), false);
    for (size_t si = 1; si + 1 < splits.size(); ++si) { split_is_corner[si] = true; }

    for (int s = 0; s + 1 < static_cast<int>(splits.size()); ++s) {
        int i0 = splits[s];
        int i1 = splits[s + 1];
        if (i1 - i0 < 1) continue;

        std::vector<Vec2f> segment(pts.begin() + i0, pts.begin() + i1 + 1);

        Vec2f tan_left = split_is_corner[s] ? CornerTangent(pts, i0, n, true, false)
                                            : EstimateTangent(segment, 0, true);
        Vec2f tan_right =
            split_is_corner[s + 1]
                ? CornerTangent(pts, i1, n, false, false)
                : EstimateTangent(segment, static_cast<int>(segment.size()) - 1, false);

        SchneiderFitRecursive(segment, tan_left, tan_right, cfg.error_threshold,
                              cfg.max_recursion_depth, cfg.reparameterize_iterations, result, 0);
    }

    return result;
}

std::vector<CubicBezier> FitBezierToClosedPolyline(const std::vector<Vec2f>& pts,
                                                   const CurveFitConfig& cfg) {
    std::vector<CubicBezier> result;
    int n = static_cast<int>(pts.size());
    if (n < 3) return result;

    float cos_thresh = std::cos(cfg.corner_angle_threshold_deg * kDegToRad);
    int k_base       = std::max(1, cfg.corner_neighbor_k);

    std::vector<bool> is_corner(n, false);

    std::vector<int> scales;
    if (cfg.enable_multiscale_corners) {
        scales = {std::max(1, k_base / 2), k_base, std::min(n / 2, k_base * 2)};
    } else {
        scales = {k_base};
    }

    for (int k : scales) {
        for (int i = 0; i < n; ++i) {
            if (is_corner[i]) continue;
            if (IsAngleCornerClosed(pts, i, k, n, cos_thresh)) { is_corner[i] = true; }
        }
    }

    if (cfg.enable_curvature_corners && n >= 5) {
        std::vector<float> curv(n, 0.0f);
        for (int i = 0; i < n; ++i) { curv[i] = DiscreteCurvatureClosed(pts, i, n); }
        for (int i = 0; i < n; ++i) {
            if (is_corner[i]) continue;
            int im1    = ((i - 1) % n + n) % n;
            int ip1    = (i + 1) % n;
            float jump = std::abs(curv[i] - curv[im1]) + std::abs(curv[i] - curv[ip1]);
            if (jump > cfg.curvature_jump_threshold && curv[i] > 0.3f) { is_corner[i] = true; }
        }
    }

    std::vector<int> corners;
    for (int i = 0; i < n; ++i) {
        if (is_corner[i]) corners.push_back(i);
    }

    if (corners.empty()) {
        std::vector<Vec2f> extended(pts.begin(), pts.end());
        extended.push_back(pts[0]);

        Vec2f tan_left  = EstimateTangent(extended, 0, true);
        Vec2f tan_right = EstimateTangent(extended, n, false);

        SchneiderFitRecursive(extended, tan_left, tan_right, cfg.error_threshold,
                              cfg.max_recursion_depth, cfg.reparameterize_iterations, result, 0);
    } else {
        std::sort(corners.begin(), corners.end());
        int nc = static_cast<int>(corners.size());

        for (int ci = 0; ci < nc; ++ci) {
            int seg_start = corners[ci];
            int seg_end   = corners[(ci + 1) % nc];

            std::vector<Vec2f> segment;
            if (seg_end > seg_start) {
                segment.assign(pts.begin() + seg_start, pts.begin() + seg_end + 1);
            } else {
                segment.assign(pts.begin() + seg_start, pts.end());
                segment.insert(segment.end(), pts.begin(), pts.begin() + seg_end + 1);
            }

            if (segment.size() < 2) continue;

            Vec2f tan_left  = CornerTangent(pts, seg_start, n, true, true);
            Vec2f tan_right = CornerTangent(pts, seg_end, n, false, true);

            SchneiderFitRecursive(segment, tan_left, tan_right, cfg.error_threshold,
                                  cfg.max_recursion_depth, cfg.reparameterize_iterations, result,
                                  0);
        }
    }

    if (!result.empty()) { result.back().p3 = result.front().p0; }
    return result;
}

void MergeNearLinearSegments(std::vector<CubicBezier>& segments, float tolerance) {
    if (segments.size() < 2 || tolerance <= 0.0f) return;

    constexpr size_t kMaxRunLength = 6;

    auto is_near_linear = [](const CubicBezier& b) -> bool {
        Vec2f chord     = b.p3 - b.p0;
        float chord_len = chord.Length();
        if (chord_len < 1e-6f) return true;
        float inv2 = 1.0f / (chord_len * chord_len);
        float d1   = std::abs((b.p1 - b.p0).Cross(chord)) * inv2;
        float d2   = std::abs((b.p2 - b.p3).Cross(chord)) * inv2;
        return d1 < 0.03f && d2 < 0.03f;
    };

    auto points_fit_chord = [&](const std::vector<CubicBezier>& segs, size_t from,
                                size_t to) -> bool {
        Vec2f chord     = segs[to - 1].p3 - segs[from].p0;
        float chord_len = chord.Length();
        if (chord_len < 1e-6f) return true;
        for (size_t k = from; k < to; ++k) {
            Vec2f rel  = segs[k].p3 - segs[from].p0;
            float dist = std::abs(rel.Cross(chord)) / chord_len;
            if (dist > tolerance * chord_len) return false;
        }
        return true;
    };

    std::vector<CubicBezier> merged;
    merged.reserve(segments.size());

    size_t i = 0;
    while (i < segments.size()) {
        if (!is_near_linear(segments[i])) {
            merged.push_back(segments[i]);
            ++i;
            continue;
        }

        size_t j = i + 1;
        while (j < segments.size() && (j - i) < kMaxRunLength && is_near_linear(segments[j]) &&
               points_fit_chord(segments, i, j + 1)) {
            ++j;
        }

        if (j - i < 2) {
            merged.push_back(segments[i]);
            ++i;
            continue;
        }

        Vec2f run_start = segments[i].p0;
        Vec2f run_end   = segments[j - 1].p3;
        Vec2f d         = run_end - run_start;
        merged.push_back(
            {run_start, run_start + d * (1.0f / 3.0f), run_start + d * (2.0f / 3.0f), run_end});
        i = j;
    }

    segments = std::move(merged);
}

int FitBezierOnGraph(BoundaryGraph& graph, const CurveFitConfig& cfg) {
    int fallback_count = 0;
    for (auto& edge : graph.edges) {
        if (edge.points.size() < 3) {
            ++fallback_count;
            continue;
        }
        auto fitted         = FitBezierToPolyline(edge.points, cfg);
        bool has_real_curve = false;
        for (const auto& b : fitted) {
            Vec2f chord     = b.p3 - b.p0;
            float chord_len = chord.Length();
            if (chord_len < 1e-6f) continue;
            float d1 = std::abs((b.p1 - b.p0).Cross(chord)) / chord_len;
            float d2 = std::abs((b.p2 - b.p3).Cross(chord)) / chord_len;
            if (d1 > 0.1f || d2 > 0.1f) {
                has_real_curve = true;
                break;
            }
        }
        if (!has_real_curve && edge.points.size() >= 3) { ++fallback_count; }
    }
    return fallback_count;
}

} // namespace ChromaPrint3D::detail
