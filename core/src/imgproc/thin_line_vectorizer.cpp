#include "imgproc/thin_line_vectorizer.h"

#include "imgproc/curve_fitting.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

namespace ChromaPrint3D::detail {

cv::Mat DetectThinRegion(const cv::Mat& mask, float max_radius) {
    if (mask.empty()) return {};
    cv::Mat dist;
    cv::distanceTransform(mask, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    cv::Mat thin;
    cv::compare(dist, max_radius, thin, cv::CMP_LE);
    cv::bitwise_and(thin, mask, thin);
    return thin;
}

namespace {

constexpr int kDr8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
constexpr int kDc8[8] = {0, 1, 1, 1, 0, -1, -1, -1};

int CountNeighbors(const cv::Mat& skel, int r, int c) {
    int count = 0;
    for (int k = 0; k < 8; ++k) {
        int nr = r + kDr8[k];
        int nc = c + kDc8[k];
        if (nr >= 0 && nr < skel.rows && nc >= 0 && nc < skel.cols &&
            skel.at<uint8_t>(nr, nc) != 0) {
            ++count;
        }
    }
    return count;
}

std::vector<Vec2f> TracePath(const cv::Mat& skel, cv::Mat& visited, int sr, int sc) {
    std::vector<Vec2f> path;
    path.push_back({static_cast<float>(sc), static_cast<float>(sr)});
    visited.at<uint8_t>(sr, sc) = 1;

    int cr = sr, cc = sc;
    while (true) {
        bool found = false;
        for (int k = 0; k < 8; ++k) {
            int nr = cr + kDr8[k];
            int nc = cc + kDc8[k];
            if (nr < 0 || nr >= skel.rows || nc < 0 || nc >= skel.cols) continue;
            if (skel.at<uint8_t>(nr, nc) == 0 || visited.at<uint8_t>(nr, nc) != 0) continue;
            visited.at<uint8_t>(nr, nc) = 1;
            path.push_back({static_cast<float>(nc), static_cast<float>(nr)});
            cr    = nr;
            cc    = nc;
            found = true;
            break;
        }
        if (!found) break;
    }
    return path;
}

} // namespace

std::vector<VectorizedShape> ExtractStrokePaths(const cv::Mat& skeleton, const cv::Mat& dist_map,
                                                const Rgb& color, float min_length) {
    std::vector<VectorizedShape> shapes;
    if (skeleton.empty()) return shapes;

    cv::Mat visited = cv::Mat::zeros(skeleton.size(), CV_8UC1);

    // Find endpoints (1 neighbor) and start tracing from them
    std::vector<cv::Point> endpoints;
    for (int r = 0; r < skeleton.rows; ++r) {
        for (int c = 0; c < skeleton.cols; ++c) {
            if (skeleton.at<uint8_t>(r, c) == 0) continue;
            if (CountNeighbors(skeleton, r, c) == 1) { endpoints.push_back({c, r}); }
        }
    }

    // Trace from endpoints first
    for (const auto& ep : endpoints) {
        if (visited.at<uint8_t>(ep.y, ep.x) != 0) continue;
        auto path = TracePath(skeleton, visited, ep.y, ep.x);
        if (static_cast<float>(path.size()) < min_length) continue;

        // Estimate stroke width from distance transform
        float total_width = 0.0f;
        int width_samples = 0;
        for (const auto& p : path) {
            int pr = static_cast<int>(p.y);
            int pc = static_cast<int>(p.x);
            if (pr >= 0 && pr < dist_map.rows && pc >= 0 && pc < dist_map.cols) {
                total_width += dist_map.at<float>(pr, pc) * 2.0f;
                ++width_samples;
            }
        }
        float avg_width =
            width_samples > 0 ? total_width / static_cast<float>(width_samples) : 1.0f;
        avg_width = std::max(0.5f, avg_width);

        CurveFitConfig fit_cfg;
        fit_cfg.error_threshold = 1.0f;
        auto beziers            = FitBezierToPolyline(path, fit_cfg);
        if (beziers.empty()) continue;

        BezierContour contour;
        contour.closed   = false;
        contour.segments = std::move(beziers);

        VectorizedShape shape;
        shape.color        = color;
        shape.is_stroke    = true;
        shape.stroke_width = avg_width;
        shape.area         = 0.0;
        shape.contours.push_back(std::move(contour));
        shapes.push_back(std::move(shape));
    }

    // Trace remaining (closed loops in skeleton)
    for (int r = 0; r < skeleton.rows; ++r) {
        for (int c = 0; c < skeleton.cols; ++c) {
            if (skeleton.at<uint8_t>(r, c) == 0 || visited.at<uint8_t>(r, c) != 0) continue;
            auto path = TracePath(skeleton, visited, r, c);
            if (static_cast<float>(path.size()) < min_length) continue;

            float total_width = 0.0f;
            int width_samples = 0;
            for (const auto& p : path) {
                int pr = static_cast<int>(p.y);
                int pc = static_cast<int>(p.x);
                if (pr >= 0 && pr < dist_map.rows && pc >= 0 && pc < dist_map.cols) {
                    total_width += dist_map.at<float>(pr, pc) * 2.0f;
                    ++width_samples;
                }
            }
            float avg_width =
                width_samples > 0 ? total_width / static_cast<float>(width_samples) : 1.0f;
            avg_width = std::max(0.5f, avg_width);

            CurveFitConfig fit_cfg;
            fit_cfg.error_threshold = 1.0f;
            auto beziers            = FitBezierToPolyline(path, fit_cfg);
            if (beziers.empty()) continue;

            BezierContour contour;
            contour.closed   = false;
            contour.segments = std::move(beziers);

            VectorizedShape shape;
            shape.color        = color;
            shape.is_stroke    = true;
            shape.stroke_width = avg_width;
            shape.area         = 0.0;
            shape.contours.push_back(std::move(contour));
            shapes.push_back(std::move(shape));
        }
    }

    return shapes;
}

} // namespace ChromaPrint3D::detail
