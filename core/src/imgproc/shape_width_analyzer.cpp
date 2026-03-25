#include "chromaprint3d/shape_width_analyzer.h"
#include "clipper_scale.h"

#include <clipper2/clipper.h>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ChromaPrint3D {

using detail::kClipperScale;

namespace {
constexpr int kRasterPadding = 2;

struct ShapeBounds {
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    float Width() const { return max_x - min_x; }

    float Height() const { return max_y - min_y; }
};

ShapeBounds ComputeContourBounds(const std::vector<Contour>& contours) {
    ShapeBounds b;
    for (const auto& contour : contours) {
        for (const auto& p : contour) {
            b.min_x = std::min(b.min_x, p.x);
            b.min_y = std::min(b.min_y, p.y);
            b.max_x = std::max(b.max_x, p.x);
            b.max_y = std::max(b.max_y, p.y);
        }
    }
    return b;
}

float ComputeShapeArea(const std::vector<Contour>& contours) {
    double total = 0.0;
    for (const auto& contour : contours) {
        Clipper2Lib::Path64 path;
        path.reserve(contour.size());
        for (const auto& p : contour) {
            path.emplace_back(static_cast<int64_t>(std::llround(p.x * kClipperScale)),
                              static_cast<int64_t>(std::llround(p.y * kClipperScale)));
        }
        total += Clipper2Lib::Area(path);
    }
    return static_cast<float>(std::abs(total) / (kClipperScale * kClipperScale));
}

Rgb EffectiveColor(const VectorShape& shape) {
    if (shape.fill_type == FillType::Solid) return shape.fill_color;
    if (!shape.gradient.stops.empty()) return shape.gradient.stops.front().color;
    return {};
}

/// Rasterize contours into a binary mask. outer contour fills white, holes fill black.
cv::Mat RasterizeShape(const std::vector<Contour>& contours, const ShapeBounds& bounds,
                       float px_per_mm, int& out_width, int& out_height) {
    out_width =
        std::max(1, static_cast<int>(std::ceil(bounds.Width() * px_per_mm)) + 2 * kRasterPadding);
    out_height =
        std::max(1, static_cast<int>(std::ceil(bounds.Height() * px_per_mm)) + 2 * kRasterPadding);

    if (out_width > 2000 || out_height > 2000) {
        spdlog::warn("RasterizeShape: clamping {}x{} -> {}x{} (may reduce accuracy)", out_width,
                     out_height, std::min(out_width, 2000), std::min(out_height, 2000));
        out_width  = std::min(out_width, 2000);
        out_height = std::min(out_height, 2000);
    }

    cv::Mat mask(out_height, out_width, CV_8UC1, cv::Scalar(0));

    float effective_px_per_mm_x =
        static_cast<float>(out_width - 2 * kRasterPadding) / std::max(bounds.Width(), 1e-6f);
    float effective_px_per_mm_y =
        static_cast<float>(out_height - 2 * kRasterPadding) / std::max(bounds.Height(), 1e-6f);

    auto toPx = [&](const Vec2f& p) -> cv::Point {
        int px = static_cast<int>(std::round((p.x - bounds.min_x) * effective_px_per_mm_x)) +
                 kRasterPadding;
        int py = static_cast<int>(std::round((p.y - bounds.min_y) * effective_px_per_mm_y)) +
                 kRasterPadding;
        return {std::clamp(px, 0, out_width - 1), std::clamp(py, 0, out_height - 1)};
    };

    if (contours.empty()) return mask;

    // contours[0] = outer, rest = holes.  fillPoly with all contours uses even-odd
    // to handle holes correctly.
    std::vector<std::vector<cv::Point>> polys;
    polys.reserve(contours.size());
    for (const auto& contour : contours) {
        std::vector<cv::Point> poly;
        poly.reserve(contour.size());
        for (const auto& p : contour) poly.push_back(toPx(p));
        if (poly.size() >= 3) polys.push_back(std::move(poly));
    }

    if (!polys.empty()) cv::fillPoly(mask, polys, cv::Scalar(255));
    return mask;
}

struct RidgeWidthStats {
    float min_width_mm    = 0.0f;
    float median_width_mm = 0.0f;
};

/// Fast width estimation via distance transform + morphological ridge detection.
/// O(n*m) single-pass instead of iterative Zhang-Suen skeleton.
RidgeWidthStats ComputeWidthFromDistanceRidge(const cv::Mat& mask, float px_per_mm) {
    cv::Mat dist;
    cv::distanceTransform(mask, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);

    // Ridge = local maxima of distance transform (medial axis approximation).
    cv::Mat dilated;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::dilate(dist, dilated, kernel);

    // Also compute eroded mask to skip ridge points within 1px of boundary.
    cv::Mat eroded_mask;
    cv::erode(mask, eroded_mask, kernel);

    constexpr float kMinDistPx = 0.5f;

    std::vector<float> widths;
    for (int r = 1; r < mask.rows - 1; ++r) {
        for (int c = 1; c < mask.cols - 1; ++c) {
            float d = dist.at<float>(r, c);
            if (d < kMinDistPx) continue;
            if (eroded_mask.at<uint8_t>(r, c) == 0) continue;
            if (d < dilated.at<float>(r, c)) continue;
            widths.push_back(2.0f * d / px_per_mm);
        }
    }

    if (widths.empty()) {
        double max_val = 0.0;
        cv::minMaxLoc(dist, nullptr, &max_val);
        float w = 2.0f * static_cast<float>(max_val) / px_per_mm;
        return {w, w};
    }

    std::sort(widths.begin(), widths.end());

    // P5 percentile instead of absolute minimum to ignore ridge artifacts.
    size_t p5_idx  = std::min(widths.size() - 1, widths.size() / 20);
    float p5_w     = widths[p5_idx];
    float median_w = widths[widths.size() / 2];
    return {p5_w, median_w};
}

} // namespace

WidthAnalysisResult AnalyzeShapeWidths(const VectorProcResult& vpr, float min_area_mm2,
                                       float raster_px_per_mm) {
    WidthAnalysisResult result;
    result.image_width_mm         = vpr.width_mm;
    result.image_height_mm        = vpr.height_mm;
    result.total_shapes           = static_cast<int>(vpr.shapes.size());
    result.min_area_threshold_mm2 = min_area_mm2;
    result.filtered_count         = 0;

    raster_px_per_mm = std::clamp(raster_px_per_mm, 5.0f, 100.0f);

    for (int i = 0; i < static_cast<int>(vpr.shapes.size()); ++i) {
        const VectorShape& shape = vpr.shapes[static_cast<size_t>(i)];
        if (shape.contours.empty()) {
            ++result.filtered_count;
            continue;
        }

        float area = ComputeShapeArea(shape.contours);
        if (area < min_area_mm2) {
            ++result.filtered_count;
            continue;
        }

        ShapeBounds bounds = ComputeContourBounds(shape.contours);
        if (bounds.Width() < 1e-4f || bounds.Height() < 1e-4f) {
            ++result.filtered_count;
            continue;
        }

        int rw = 0, rh = 0;
        cv::Mat mask = RasterizeShape(shape.contours, bounds, raster_px_per_mm, rw, rh);

        constexpr int kMinRasterDim = 3;
        if (rw < kMinRasterDim || rh < kMinRasterDim || cv::countNonZero(mask) == 0) {
            ++result.filtered_count;
            continue;
        }

        float effective_px_per_mm = std::min(
            static_cast<float>(rw - 2 * kRasterPadding) / std::max(bounds.Width(), 1e-6f),
            static_cast<float>(rh - 2 * kRasterPadding) / std::max(bounds.Height(), 1e-6f));

        auto stats = ComputeWidthFromDistanceRidge(mask, effective_px_per_mm);

        ShapeWidthInfo info;
        info.index           = i;
        info.color           = EffectiveColor(shape);
        info.area_mm2        = area;
        info.min_width_mm    = stats.min_width_mm;
        info.median_width_mm = stats.median_width_mm;
        result.shapes.push_back(info);
    }

    spdlog::info("WidthAnalysis: {} shapes analyzed, {} filtered (area < {:.2f} mm²)",
                 result.shapes.size(), result.filtered_count, min_area_mm2);
    return result;
}

} // namespace ChromaPrint3D
