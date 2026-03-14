#include "svg_geometry.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ChromaPrint3D::eval {

double PolylineSignedArea(const std::vector<cv::Point>& pts) {
    if (pts.size() < 3) return 0;
    double area = 0;
    size_t n    = pts.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area += static_cast<double>(pts[i].x) * pts[j].y;
        area -= static_cast<double>(pts[j].x) * pts[i].y;
    }
    return area * 0.5;
}

double PolylinePerimeter(const std::vector<cv::Point>& pts) {
    if (pts.size() < 2) return 0;
    double len = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        size_t j  = (i + 1) % pts.size();
        double dx = pts[j].x - pts[i].x;
        double dy = pts[j].y - pts[i].y;
        len += std::sqrt(dx * dx + dy * dy);
    }
    return len;
}

bool PointInPolyline(const std::vector<cv::Point>& poly, cv::Point pt) {
    bool inside = false;
    size_t n    = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        if (((poly[i].y > pt.y) != (poly[j].y > pt.y)) &&
            (pt.x < (poly[j].x - poly[i].x) * (pt.y - poly[i].y) /
                            static_cast<double>(poly[j].y - poly[i].y) +
                        poly[i].x)) {
            inside = !inside;
        }
    }
    return inside;
}

cv::Mat FillShapeWithHoles(const std::vector<std::vector<cv::Point>>& contours, int width,
                           int height) {
    if (contours.empty() || width <= 0 || height <= 0)
        return cv::Mat::zeros(height, width, CV_8UC1);

    struct ContourInfo {
        size_t index;
        double abs_area;
        cv::Rect bbox;
        bool is_hole = false;
    };

    std::vector<ContourInfo> infos;
    infos.reserve(contours.size());
    for (size_t i = 0; i < contours.size(); ++i) {
        if (contours[i].size() < 3) continue;
        ContourInfo ci;
        ci.index    = i;
        ci.abs_area = std::abs(PolylineSignedArea(contours[i]));
        ci.bbox     = cv::boundingRect(contours[i]);
        infos.push_back(ci);
    }

    std::sort(infos.begin(), infos.end(),
              [](const ContourInfo& a, const ContourInfo& b) { return a.abs_area > b.abs_area; });

    for (size_t i = 0; i < infos.size(); ++i) {
        if (i == 0) {
            infos[i].is_hole = false;
            continue;
        }
        cv::Point center(infos[i].bbox.x + infos[i].bbox.width / 2,
                         infos[i].bbox.y + infos[i].bbox.height / 2);
        bool found = false;
        for (size_t j = 0; j < i; ++j) {
            if (!(infos[i].bbox.x >= infos[j].bbox.x && infos[i].bbox.y >= infos[j].bbox.y &&
                  infos[i].bbox.br().x <= infos[j].bbox.br().x &&
                  infos[i].bbox.br().y <= infos[j].bbox.br().y))
                continue;
            if (PointInPolyline(contours[infos[j].index], center)) {
                infos[i].is_hole = !infos[j].is_hole;
                found            = true;
                break;
            }
        }
        if (!found) infos[i].is_hole = false;
    }

    // Fill outers with 255, erase holes with 0 — processed largest-first
    // so nested structures (bullseye, etc.) resolve correctly.
    cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
    for (auto& ci : infos) {
        std::vector<std::vector<cv::Point>> single = {contours[ci.index]};
        cv::fillPoly(result, single, ci.is_hole ? cv::Scalar(0) : cv::Scalar(255));
    }
    return result;
}

} // namespace ChromaPrint3D::eval
