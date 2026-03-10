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

} // namespace

std::vector<int> DetectCorners(const std::vector<Vec2f>& pts, const CurveFitConfig& cfg) {
    std::vector<int> corners;
    int n = static_cast<int>(pts.size());
    if (n < 3) return corners;

    float cos_thresh = std::cos(cfg.corner_angle_threshold_deg * kDegToRad);
    int k            = std::max(1, cfg.corner_neighbor_k);

    for (int i = 1; i < n - 1; ++i) {
        int prev_idx = std::max(0, i - k);
        int next_idx = std::min(n - 1, i + k);
        Vec2f v1     = (pts[prev_idx] - pts[i]);
        Vec2f v2     = (pts[next_idx] - pts[i]);
        float len1   = v1.Length();
        float len2   = v2.Length();
        if (len1 < 1e-6f || len2 < 1e-6f) continue;
        float cos_angle = v1.Dot(v2) / (len1 * len2);
        if (cos_angle < cos_thresh) { corners.push_back(i); }
    }
    return corners;
}

std::vector<CubicBezier> FitBezierToPolyline(const std::vector<Vec2f>& pts,
                                             const CurveFitConfig& cfg) {
    std::vector<CubicBezier> result;
    if (pts.size() < 2) return result;
    if (pts.size() == 2) {
        result.push_back(MakeLinearBezier(pts[0], pts[1]));
        return result;
    }

    auto corners = DetectCorners(pts, cfg);

    // Build segment boundaries: [0, corner1, corner2, ..., n-1]
    std::vector<int> splits;
    splits.push_back(0);
    for (int c : corners) {
        if (c > splits.back()) splits.push_back(c);
    }
    if (splits.back() != static_cast<int>(pts.size()) - 1) {
        splits.push_back(static_cast<int>(pts.size()) - 1);
    }

    for (int s = 0; s + 1 < static_cast<int>(splits.size()); ++s) {
        int i0 = splits[s];
        int i1 = splits[s + 1];
        if (i1 - i0 < 1) continue;

        std::vector<Vec2f> segment(pts.begin() + i0, pts.begin() + i1 + 1);
        Vec2f tan_left  = EstimateTangent(segment, 0, true);
        Vec2f tan_right = EstimateTangent(segment, static_cast<int>(segment.size()) - 1, false);

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

    // Corner detection with wrap-around for closed polylines
    float cos_thresh = std::cos(cfg.corner_angle_threshold_deg * kDegToRad);
    int k            = std::max(1, cfg.corner_neighbor_k);
    std::vector<int> corners;

    for (int i = 0; i < n; ++i) {
        int prev_idx = ((i - k) % n + n) % n;
        int next_idx = (i + k) % n;
        Vec2f v1     = pts[prev_idx] - pts[i];
        Vec2f v2     = pts[next_idx] - pts[i];
        float len1   = v1.Length();
        float len2   = v2.Length();
        if (len1 < 1e-6f || len2 < 1e-6f) continue;
        float cos_angle = v1.Dot(v2) / (len1 * len2);
        if (cos_angle < cos_thresh) { corners.push_back(i); }
    }

    if (corners.empty()) {
        // No corners: fit the entire loop. Extend pts with closure segment.
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

            Vec2f tan_left  = EstimateTangent(segment, 0, true);
            Vec2f tan_right = EstimateTangent(segment, static_cast<int>(segment.size()) - 1, false);

            SchneiderFitRecursive(segment, tan_left, tan_right, cfg.error_threshold,
                                  cfg.max_recursion_depth, cfg.reparameterize_iterations, result,
                                  0);
        }
    }

    if (!result.empty()) { result.back().p3 = result.front().p0; }
    return result;
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
