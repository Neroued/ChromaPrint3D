#include "svg_rasterizer.h"

#include "svg_geometry.h"

#include <nanosvg/nanosvg.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace ChromaPrint3D::eval {

namespace {

cv::Scalar NsvgColorToBgr(unsigned int c) {
    // NanoSVG: bits 0-7 = R, 8-15 = G, 16-23 = B; OpenCV Scalar order is B, G, R
    return {static_cast<double>((c >> 16) & 0xFF), static_cast<double>((c >> 8) & 0xFF),
            static_cast<double>(c & 0xFF)};
}

void FlattenCubicBezier(std::vector<cv::Point>& pts, float x0, float y0, float x1, float y1,
                        float x2, float y2, float x3, float y3, int depth = 0) {
    if (depth > 10) {
        pts.emplace_back(
            cv::Point(static_cast<int>(std::round(x3)), static_cast<int>(std::round(y3))));
        return;
    }
    float dx = x3 - x0, dy = y3 - y0;
    float d2  = std::abs((x1 - x3) * dy - (y1 - y3) * dx);
    float d3  = std::abs((x2 - x3) * dy - (y2 - y3) * dx);
    float tol = 0.25f;
    if ((d2 + d3) * (d2 + d3) < tol * (dx * dx + dy * dy)) {
        pts.emplace_back(
            cv::Point(static_cast<int>(std::round(x3)), static_cast<int>(std::round(y3))));
        return;
    }
    float x01 = (x0 + x1) * 0.5f, y01 = (y0 + y1) * 0.5f;
    float x12 = (x1 + x2) * 0.5f, y12 = (y1 + y2) * 0.5f;
    float x23 = (x2 + x3) * 0.5f, y23 = (y2 + y3) * 0.5f;
    float xa = (x01 + x12) * 0.5f, ya = (y01 + y12) * 0.5f;
    float xb = (x12 + x23) * 0.5f, yb = (y12 + y23) * 0.5f;
    float xm = (xa + xb) * 0.5f, ym = (ya + yb) * 0.5f;
    FlattenCubicBezier(pts, x0, y0, x01, y01, xa, ya, xm, ym, depth + 1);
    FlattenCubicBezier(pts, xm, ym, xb, yb, x23, y23, x3, y3, depth + 1);
}

struct FlatPath {
    std::vector<cv::Point> pts;
    bool closed = false;
};

std::vector<FlatPath> FlattenShape(NSVGshape* shape, float sx, float sy) {
    std::vector<FlatPath> paths;
    for (NSVGpath* path = shape->paths; path; path = path->next) {
        if (path->npts < 4) continue;
        std::vector<cv::Point> pts;
        float* p = path->pts;
        pts.emplace_back(cv::Point(static_cast<int>(std::round(p[0] * sx)),
                                   static_cast<int>(std::round(p[1] * sy))));
        for (int i = 0; i < path->npts - 1; i += 3) {
            float* seg = &path->pts[i * 2];
            FlattenCubicBezier(pts, seg[0] * sx, seg[1] * sy, seg[2] * sx, seg[3] * sy, seg[4] * sx,
                               seg[5] * sy, seg[6] * sx, seg[7] * sy);
        }
        if (pts.size() >= 2) paths.push_back({std::move(pts), path->closed != 0});
    }
    return paths;
}

} // namespace

RasterizedSvg RasterizeSvg(const std::string& svg_content, int width, int height) {
    RasterizedSvg result;
    result.bgr         = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    result.coverage    = cv::Mat::zeros(height, width, CV_8UC1);
    result.shape_count = cv::Mat::zeros(height, width, CV_16UC1);

    std::vector<char> buf(svg_content.begin(), svg_content.end());
    buf.push_back('\0');

    NSVGimage* img = nsvgParse(buf.data(), "px", 96.0f);
    if (!img) return result;
    auto guard = std::unique_ptr<NSVGimage, decltype(&nsvgDelete)>(img, nsvgDelete);

    float sx = (img->width > 0) ? static_cast<float>(width) / img->width : 1.0f;
    float sy = (img->height > 0) ? static_cast<float>(height) / img->height : 1.0f;

    for (NSVGshape* shape = img->shapes; shape; shape = shape->next) {
        if (!(shape->flags & NSVG_FLAGS_VISIBLE)) continue;
        if (shape->fill.type == NSVG_PAINT_NONE && shape->stroke.type == NSVG_PAINT_NONE) continue;

        bool has_fill   = shape->fill.type == NSVG_PAINT_COLOR;
        bool has_stroke = shape->stroke.type == NSVG_PAINT_COLOR;

        auto paths = FlattenShape(shape, sx, sy);
        if (paths.empty()) continue;

        cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);

        // Fill rendering — hole-aware via geometric containment
        if (has_fill) {
            std::vector<std::vector<cv::Point>> fill_contours;
            for (auto& fp : paths) {
                if (fp.pts.size() >= 3) fill_contours.push_back(fp.pts);
            }
            if (!fill_contours.empty()) {
                cv::Scalar fill_color = NsvgColorToBgr(shape->fill.color);
                cv::Mat fill_mask     = FillShapeWithHoles(fill_contours, width, height);
                result.bgr.setTo(fill_color, fill_mask);
                mask.setTo(255, fill_mask);
            }
        }

        // Stroke rendering
        if (has_stroke && shape->strokeWidth > 0) {
            cv::Scalar stroke_color = NsvgColorToBgr(shape->stroke.color);
            int thickness =
                std::max(1, static_cast<int>(std::round(shape->strokeWidth * std::max(sx, sy))));
            for (auto& fp : paths) {
                cv::polylines(result.bgr, fp.pts, fp.closed, stroke_color, thickness, cv::LINE_AA);
                cv::polylines(mask, fp.pts, fp.closed, cv::Scalar(255), thickness, cv::LINE_8);
            }
        }

        result.coverage.setTo(255, mask);
        cv::Mat inc;
        mask.convertTo(inc, CV_16UC1, 1.0 / 255.0);
        result.shape_count += inc;
    }

    return result;
}

} // namespace ChromaPrint3D::eval
