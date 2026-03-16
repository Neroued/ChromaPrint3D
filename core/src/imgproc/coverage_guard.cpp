#include "imgproc/coverage_guard.h"

#include "imgproc/bezier.h"
#include "imgproc/potrace_adapter.h"
#include "imgproc/topology_repair.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace ChromaPrint3D::detail {

namespace {

BezierContour RingToBezier(const std::vector<Vec2f>& ring) {
    BezierContour contour;
    contour.closed = true;
    if (ring.size() < 3) return contour;
    contour.segments.reserve(ring.size());

    for (size_t i = 0; i < ring.size(); ++i) {
        const Vec2f& a = ring[i];
        const Vec2f& b = ring[(i + 1) % ring.size()];
        Vec2f d        = b - a;
        if (d.LengthSquared() < 1e-8f) continue;
        contour.segments.push_back({a, a + d * (1.0f / 3.0f), a + d * (2.0f / 3.0f), b});
    }
    return contour;
}

std::vector<cv::Point> FlattenContour(const BezierContour& contour, int width, int height) {
    std::vector<cv::Point> poly;
    if (contour.segments.empty()) return poly;
    std::vector<Vec2f> pts;
    pts.reserve(contour.segments.size() * 8 + 1);
    pts.push_back(contour.segments.front().p0);
    for (const auto& seg : contour.segments) FlattenCubicBezier(seg, 0.45f, pts);

    poly.reserve(pts.size());
    for (const auto& p : pts) {
        int x = std::clamp(static_cast<int>(std::lround(p.x)), 0, width - 1);
        int y = std::clamp(static_cast<int>(std::lround(p.y)), 0, height - 1);
        poly.emplace_back(x, y);
    }
    if (poly.size() > 1 && poly.front() == poly.back()) poly.pop_back();
    return poly;
}

cv::Mat RasterizeCoverage(const std::vector<VectorizedShape>& shapes, int width, int height) {
    cv::Mat coverage(height, width, CV_8UC1, cv::Scalar(0));
    for (const auto& shape : shapes) {
        std::vector<std::vector<cv::Point>> polys;
        for (const auto& contour : shape.contours) {
            auto poly = FlattenContour(contour, width, height);
            if (poly.size() >= 3) polys.push_back(std::move(poly));
        }
        if (!polys.empty()) cv::fillPoly(coverage, polys, cv::Scalar(255));
    }
    return coverage;
}

} // namespace

void ApplyCoverageGuard(std::vector<VectorizedShape>& shapes, const cv::Mat& labels,
                        const std::vector<Rgb>& palette, float min_ratio, float tracing_epsilon,
                        float min_patch_area) {
    if (labels.empty() || labels.type() != CV_32SC1) {
        spdlog::warn("CoverageGuard skipped: invalid labels (empty={} type={})", labels.empty(),
                     labels.empty() ? -1 : labels.type());
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    const int h      = labels.rows;
    const int w      = labels.cols;
    spdlog::debug("CoverageGuard start: labels={}x{}, min_ratio={:.4f}, tracing_eps={:.3f}", w, h,
                  min_ratio, tracing_epsilon);

    cv::Mat source_mask(h, w, CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < h; ++r) {
        const int* row = labels.ptr<int>(r);
        uint8_t* out   = source_mask.ptr<uint8_t>(r);
        for (int c = 0; c < w; ++c) out[c] = (row[c] >= 0) ? 255 : 0;
    }

    cv::Mat coverage = RasterizeCoverage(shapes, w, h);
    cv::Mat covered;
    cv::bitwise_and(source_mask, coverage, covered);

    int source_px  = cv::countNonZero(source_mask);
    int covered_px = cv::countNonZero(covered);
    if (source_px <= 0) {
        spdlog::debug("CoverageGuard skipped: source pixels are zero");
        return;
    }

    float ratio = static_cast<float>(covered_px) / static_cast<float>(source_px);
    if (ratio >= min_ratio) {
        spdlog::debug("CoverageGuard skipped: coverage_ratio={:.4f} >= min_ratio={:.4f}", ratio,
                      min_ratio);
        return;
    }
    spdlog::warn("CoverageGuard triggered: coverage_ratio={:.4f} < min_ratio={:.4f}", ratio,
                 min_ratio);

    cv::Mat missing;
    cv::bitwise_not(coverage, missing);
    cv::bitwise_and(missing, source_mask, missing);

    cv::Mat cc_labels;
    int ncc = cv::connectedComponents(missing, cc_labels, 8, CV_32S);
    if (ncc <= 1) {
        spdlog::debug("CoverageGuard no missing connected components");
        return;
    }

    int eligible_components = 0;
    int patched_components  = 0;
    int patch_shapes_added  = 0;
    int invalid_label_skips = 0;
    for (int cid = 1; cid < ncc; ++cid) {
        cv::Mat comp_mask(h, w, CV_8UC1, cv::Scalar(0));
        std::unordered_map<int, int> label_hist;
        int area = 0;

        for (int r = 0; r < h; ++r) {
            const int* cc_row = cc_labels.ptr<int>(r);
            const int* lb_row = labels.ptr<int>(r);
            uint8_t* out      = comp_mask.ptr<uint8_t>(r);
            for (int c = 0; c < w; ++c) {
                if (cc_row[c] != cid) continue;
                out[c] = 255;
                ++area;
                label_hist[lb_row[c]]++;
            }
        }

        if (area < static_cast<int>(std::max(1.0f, min_patch_area))) continue;
        if (label_hist.empty()) continue;
        ++eligible_components;

        int best_label = -1;
        int best_count = -1;
        for (const auto& kv : label_hist) {
            if (kv.second > best_count) {
                best_count = kv.second;
                best_label = kv.first;
            }
        }
        if (best_label < 0 || best_label >= static_cast<int>(palette.size())) {
            ++invalid_label_skips;
            continue;
        }

        auto traced = TraceMaskWithPotrace(comp_mask, tracing_epsilon * 0.8f);
        auto fixed = RepairTopology(traced, tracing_epsilon * 0.6f, min_patch_area, min_patch_area);
        if (!fixed.empty()) ++patched_components;

        for (auto& g : fixed) {
            VectorizedShape patch;
            patch.color = palette[best_label];
            patch.area  = g.area;
            patch.contours.push_back(RingToBezier(g.outer));
            for (const auto& hole : g.holes) patch.contours.push_back(RingToBezier(hole));
            if (patch.contours.empty()) continue;

            cv::Mat patch_raster(h, w, CV_8UC1, cv::Scalar(0));
            {
                std::vector<std::vector<cv::Point>> polys;
                for (const auto& c : patch.contours) {
                    auto poly = FlattenContour(c, w, h);
                    if (poly.size() >= 3) polys.push_back(std::move(poly));
                }
                if (!polys.empty()) cv::fillPoly(patch_raster, polys, cv::Scalar(255));
            }
            cv::Mat overlap_mask;
            cv::bitwise_and(patch_raster, coverage, overlap_mask);
            int patch_px   = cv::countNonZero(patch_raster);
            int overlap_px = cv::countNonZero(overlap_mask);
            if (patch_px > 0 &&
                static_cast<float>(overlap_px) / static_cast<float>(patch_px) > 0.5f) {
                spdlog::debug("CoverageGuard skip high-overlap patch: patch_px={}, overlap={}",
                              patch_px, overlap_px);
                continue;
            }

            shapes.push_back(std::move(patch));
            ++patch_shapes_added;
        }
    }
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    spdlog::info(
        "CoverageGuard done: source_px={}, covered_px={}, ratio={:.4f}, ncc={}, eligible={}, "
        "patched_components={}, patch_shapes_added={}, invalid_label_skips={}, elapsed_ms={:.2f}",
        source_px, covered_px, ratio, ncc, eligible_components, patched_components,
        patch_shapes_added, invalid_label_skips, elapsed_ms);
}

} // namespace ChromaPrint3D::detail
