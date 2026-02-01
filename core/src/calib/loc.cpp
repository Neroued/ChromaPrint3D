#include "calib.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define DEBUG_DIR "/tmp/calib_debug"

#ifdef DEBUG_DIR
#    include <filesystem>
#endif

namespace ChromaPrint3D {
namespace {

constexpr float kPi = 3.14159265358979323846f;

#ifdef DEBUG_DIR
static std::string DebugDir() { return std::string(DEBUG_DIR); }

static void SaveDebugImage(const std::string& name, const cv::Mat& img) {
    try {
        std::filesystem::create_directories(DebugDir());
        if (img.empty()) { return; }

        cv::Mat out = img;
        if (out.depth() != CV_8U) {
            double min_v = 0.0, max_v = 0.0;
            cv::minMaxLoc(out, &min_v, &max_v);
            double scale = (max_v - min_v) > 1e-6 ? (255.0 / (max_v - min_v)) : 1.0;
            out.convertTo(out, CV_8U, scale, -min_v * scale);
        }

        if (out.channels() == 4) {
            cv::Mat bgr;
            cv::cvtColor(out, bgr, cv::COLOR_BGRA2BGR);
            out = bgr;
        }
        std::filesystem::path path = std::filesystem::path(DebugDir()) / name;
        cv::imwrite(path.string(), out);
    } catch (...) {
        // Debug output should not fail the main pipeline.
    }
}
#endif

static cv::Mat EnsureBgr(const cv::Mat& src) {
    if (src.empty()) { return cv::Mat(); }
    if (src.channels() == 3) { return src; }
    if (src.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    if (src.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    throw std::runtime_error("Unsupported image channel count");
}

static double MedianGray(const cv::Mat& gray) {
    if (gray.empty() || gray.type() != CV_8UC1) { return 0.0; }
    int hist[256] = {0};
    for (int y = 0; y < gray.rows; ++y) {
        const uint8_t* row = gray.ptr<uint8_t>(y);
        for (int x = 0; x < gray.cols; ++x) { hist[row[x]]++; }
    }
    int total = gray.rows * gray.cols;
    int mid   = total / 2;
    int acc   = 0;
    for (int i = 0; i < 256; ++i) {
        acc += hist[i];
        if (acc >= mid) { return static_cast<double>(i); }
    }
    return 0.0;
}

static cv::Mat AutoCanny(const cv::Mat& gray) {
    const double v = MedianGray(gray);
    double lower   = std::max(0.0, (1.0 - 0.33) * v);
    double upper   = std::min(255.0, (1.0 + 0.33) * v);
    if (upper < 10.0) {
        lower = 30.0;
        upper = 90.0;
    }
    cv::Mat edges;
    cv::Canny(gray, edges, lower, upper);
    return edges;
}

static std::vector<cv::Point2f> OrderCorners(const std::vector<cv::Point2f>& pts) {
    if (pts.size() != 4) { throw std::runtime_error("OrderCorners expects 4 points"); }
    cv::Point2f tl, tr, br, bl;
    double min_sum  = std::numeric_limits<double>::infinity();
    double max_sum  = -std::numeric_limits<double>::infinity();
    double min_diff = std::numeric_limits<double>::infinity();
    double max_diff = -std::numeric_limits<double>::infinity();
    for (const auto& p : pts) {
        double sum  = p.x + p.y;
        double diff = p.x - p.y;
        if (sum < min_sum) { min_sum = sum, tl = p; }
        if (sum > max_sum) { max_sum = sum, br = p; }
        if (diff < min_diff) { min_diff = diff, tr = p; }
        if (diff > max_diff) { max_diff = diff, bl = p; }
    }
    return {tl, tr, br, bl};
}

static std::vector<cv::Point2f> CoarseLocateBoard(const cv::Mat& bgr, double& scale_out) {
    const int width  = bgr.cols;
    const int height = bgr.rows;
    if (width <= 0 || height <= 0) { throw std::runtime_error("Input image is empty"); }

    const int max_dim = std::max(width, height);
    const int target  = 1200;
    scale_out         = (max_dim > target) ? (static_cast<double>(target) / max_dim) : 1.0;

    cv::Mat small;
    if (scale_out < 1.0) {
        cv::resize(bgr, small, cv::Size(), scale_out, scale_out, cv::INTER_AREA);
    } else {
        small = bgr;
    }

    cv::Mat gray;
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0.0);

    cv::Mat edges  = AutoCanny(gray);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

#ifdef DEBUG_DIR
    SaveDebugImage("01_coarse_small.png", small);
    SaveDebugImage("02_coarse_gray.png", gray);
    SaveDebugImage("03_coarse_edges.png", edges);
#endif

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double img_area = static_cast<double>(small.cols) * static_cast<double>(small.rows);
    double best_area      = 0.0;
    cv::RotatedRect best_rect;

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < img_area * 0.1) { continue; }
        cv::RotatedRect rect = cv::minAreaRect(contour);
        float w              = rect.size.width;
        float h              = rect.size.height;
        if (w <= 1.0f || h <= 1.0f) { continue; }
        float ratio = std::max(w, h) / std::min(w, h);
        if (ratio > 1.4f) { continue; }
        double rect_area = static_cast<double>(w) * static_cast<double>(h);
        if (rect_area > best_area) {
            best_area = rect_area;
            best_rect = rect;
        }
    }

    if (best_area <= 0.0) { throw std::runtime_error("Failed to coarse-locate board"); }

    cv::Point2f pts[4];
    best_rect.points(pts);
    std::vector<cv::Point2f> corners(pts, pts + 4);

#ifdef DEBUG_DIR
    cv::Mat contour_vis = small.clone();
    for (const auto& c : contours) {
        cv::drawContours(contour_vis, std::vector{c}, -1, {0, 255, 0}, 1);
    }
    for (int i = 0; i < 4; ++i) {
        cv::line(contour_vis, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
    }
    SaveDebugImage("04_coarse_rect.png", contour_vis);
#endif

    for (auto& p : corners) {
        p.x = static_cast<float>(p.x / scale_out);
        p.y = static_cast<float>(p.y / scale_out);
    }
    return OrderCorners(corners);
}

static cv::Point2f ProjectPoint(const cv::Mat& H, const cv::Point2f& p) {
    std::vector<cv::Point2f> src{p};
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, H);
    return dst.front();
}

static float EstimateRadiusPx(const cv::Mat& H, const cv::Point2f& center, float radius) {
    const cv::Point2f p0 = ProjectPoint(H, center);
    const cv::Point2f p1 = ProjectPoint(H, cv::Point2f(center.x + radius, center.y));
    const cv::Point2f d  = p1 - p0;
    return std::sqrt(d.x * d.x + d.y * d.y);
}

struct CircleResult {
    bool ok = false;
    cv::Point2f center;
    float radius = 0.0f;
};

static CircleResult DetectCircleInRoi(const cv::Mat& gray, float expected_radius,
                                      const std::string& debug_name) {
    CircleResult best;
    if (gray.empty()) { return best; }

    cv::Mat blurred;
    cv::medianBlur(gray, blurred, 5);

    cv::Mat binary;
    cv::threshold(blurred, binary, 0.0, 255.0, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    float best_score = -1.0f;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 10.0) { continue; }
        cv::Point2f c;
        float r = 0.0f;
        cv::minEnclosingCircle(contour, c, r);
        if (r <= 0.0f) { continue; }
        if (expected_radius > 0.0f) {
            if (r < expected_radius * 0.5f || r > expected_radius * 1.5f) { continue; }
        }
        float circularity    = static_cast<float>(area / (kPi * r * r));
        float radius_penalty = 0.0f;
        if (expected_radius > 0.0f) {
            radius_penalty = std::abs(r - expected_radius) / expected_radius;
        }
        float score = circularity - radius_penalty;
        if (score > best_score) {
            best_score  = score;
            best.ok     = true;
            best.center = c;
            best.radius = r;
        }
    }

    const bool found_by_contour = best.ok;
    std::vector<cv::Vec3f> circles;
    cv::GaussianBlur(blurred, blurred, cv::Size(9, 9), 2.0, 2.0);
    const int min_radius = std::max(1, static_cast<int>(std::floor(expected_radius * 0.5f)));
    const int max_radius =
        std::max(min_radius + 1, static_cast<int>(std::ceil(expected_radius * 1.5f)));
    cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, 1.2, expected_radius * 1.5, 100.0, 20.0,
                     min_radius, max_radius);
    if (!found_by_contour && !circles.empty()) {
        float best_diff = std::numeric_limits<float>::infinity();
        for (const auto& c : circles) {
            float r    = c[2];
            float diff = (expected_radius > 0.0f) ? std::abs(r - expected_radius) : 0.0f;
            if (diff < best_diff) {
                best_diff   = diff;
                best.ok     = true;
                best.center = cv::Point2f(c[0], c[1]);
                best.radius = r;
            }
        }
    }

#ifdef DEBUG_DIR
    if (!debug_name.empty()) {
        SaveDebugImage(debug_name + "_gray.png", gray);
        SaveDebugImage(debug_name + "_binary.png", binary);

        cv::Mat overlay;
        cv::cvtColor(gray, overlay, cv::COLOR_GRAY2BGR);
        if (!contours.empty()) {
            cv::drawContours(overlay, contours, -1, cv::Scalar(0, 255, 0), 1);
        }
        if (best.ok) {
            cv::circle(overlay, best.center,
                       std::max(2, static_cast<int>(std::lround(best.radius))),
                       cv::Scalar(0, 0, 255), 2);
        }
        SaveDebugImage(debug_name + "_overlay.png", overlay);
    }
#endif
    return best;
}

static std::vector<cv::Point2f> OrderMainByTag(const std::vector<cv::Point2f>& centers,
                                               const cv::Point2f& tag_center) {
    if (centers.size() != 4) { throw std::runtime_error("Need 4 main holes"); }
    cv::Point2f center(0.0f, 0.0f);
    for (const auto& p : centers) { center += p; }
    center.x /= 4.0f;
    center.y /= 4.0f;

    struct AnglePoint {
        float angle = 0.0f;
        cv::Point2f point;
    };

    std::vector<AnglePoint> ordered;
    ordered.reserve(4);
    for (const auto& p : centers) {
        float angle = std::atan2(p.y - center.y, p.x - center.x);
        ordered.push_back({angle, p});
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const AnglePoint& a, const AnglePoint& b) { return a.angle < b.angle; });

    int tl_idx      = 0;
    float best_dist = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 4; ++i) {
        cv::Point2f d = ordered[i].point - tag_center;
        float dist    = d.x * d.x + d.y * d.y;
        if (dist < best_dist) {
            best_dist = dist;
            tl_idx    = i;
        }
    }

    std::vector<cv::Point2f> result(4);
    for (int i = 0; i < 4; ++i) { result[i] = ordered[(tl_idx + i) % 4].point; }
    return result;
}

static std::vector<cv::Point2f> RotatedVectors(const cv::Point2f& v) {
    return {
        v,
        cv::Point2f(-v.y, v.x),
        cv::Point2f(-v.x, -v.y),
        cv::Point2f(v.y, -v.x),
    };
}

} // namespace

cv::Mat LocateCalibrationColorRegion(const std::string& image_path,
                                     const CalibrationBoardMeta& meta) {
    cv::Mat input = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (input.empty()) { throw std::runtime_error("Failed to read image: " + image_path); }
    cv::Mat bgr = EnsureBgr(input);
    if (bgr.empty()) { throw std::runtime_error("Failed to normalize input image"); }

#ifdef DEBUG_DIR
    SaveDebugImage("00_input.png", bgr);
#endif

    const int grid_rows = meta.grid_rows;
    const int grid_cols = meta.grid_cols;
    if (grid_rows <= 0 || grid_cols <= 0) { throw std::runtime_error("Invalid grid size"); }

    int scale = meta.config.layout.resolution_scale;
    if (scale <= 0) { scale = 1; }

    const int tile   = meta.config.layout.tile_factor * scale;
    const int gap    = meta.config.layout.gap_factor * scale;
    const int margin = meta.config.layout.margin_factor * scale;
    if (tile <= 0) { throw std::runtime_error("Invalid tile factor"); }
    if (gap < 0 || margin < 0) { throw std::runtime_error("Invalid gap or margin factor"); }

    const int color_w = grid_cols * tile + (grid_cols - 1) * gap;
    const int color_h = grid_rows * tile + (grid_rows - 1) * gap;
    const int board_w = color_w + 2 * margin;
    const int board_h = color_h + 2 * margin;
    if (board_w <= 0 || board_h <= 0) { throw std::runtime_error("Invalid board size"); }

    double coarse_scale                     = 1.0;
    std::vector<cv::Point2f> coarse_corners = CoarseLocateBoard(bgr, coarse_scale);

    std::vector<cv::Point2f> board_corners = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(board_w - 1), 0.0f),
        cv::Point2f(static_cast<float>(board_w - 1), static_cast<float>(board_h - 1)),
        cv::Point2f(0.0f, static_cast<float>(board_h - 1)),
    };

    cv::Mat H_img_to_board = cv::getPerspectiveTransform(coarse_corners, board_corners);
    cv::Mat H_board_to_img = H_img_to_board.inv();

    const float offset = static_cast<float>(meta.config.layout.fiducial.offset_factor * scale);
    const float main_d = static_cast<float>(meta.config.layout.fiducial.main_d_factor * scale);
    const float tag_d  = static_cast<float>(meta.config.layout.fiducial.tag_d_factor * scale);
    const float main_r = main_d * 0.5f;
    const float tag_r  = tag_d * 0.5f;

    const std::vector<cv::Point2f> canonical_main = {
        cv::Point2f(offset, offset),
        cv::Point2f(static_cast<float>(board_w) - offset, offset),
        cv::Point2f(static_cast<float>(board_w) - offset, static_cast<float>(board_h) - offset),
        cv::Point2f(offset, static_cast<float>(board_h) - offset),
    };
    const cv::Point2f tag_offset(
        static_cast<float>(meta.config.layout.fiducial.tag_dx_factor * scale),
        static_cast<float>(meta.config.layout.fiducial.tag_dy_factor * scale));

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> main_centers;
    std::vector<float> main_radii;
    main_centers.reserve(4);
    main_radii.reserve(4);
    for (size_t i = 0; i < canonical_main.size(); ++i) {
        const auto& canonical = canonical_main[i];
        cv::Point2f guess     = ProjectPoint(H_board_to_img, canonical);
        float expected_r      = EstimateRadiusPx(H_board_to_img, canonical, main_r);
        float search_r        = std::max(10.0f, expected_r * 2.5f);

        int x0 = std::max(0, static_cast<int>(std::floor(guess.x - search_r)));
        int y0 = std::max(0, static_cast<int>(std::floor(guess.y - search_r)));
        int x1 = std::min(gray.cols - 1, static_cast<int>(std::ceil(guess.x + search_r)));
        int y1 = std::min(gray.rows - 1, static_cast<int>(std::ceil(guess.y + search_r)));
        if (x1 <= x0 || y1 <= y0) { throw std::runtime_error("Main hole ROI is invalid"); }

        cv::Rect roi(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
        const std::string debug_name = "hole_main_" + std::to_string(i);
        CircleResult circle          = DetectCircleInRoi(gray(roi), expected_r, debug_name);
        if (!circle.ok) { throw std::runtime_error("Failed to detect main hole"); }

        circle.center.x += static_cast<float>(roi.x);
        circle.center.y += static_cast<float>(roi.y);
        main_centers.push_back(circle.center);
        main_radii.push_back(circle.radius);
    }

    CircleResult tag_circle;
    float best_tag_score   = std::numeric_limits<float>::infinity();
    const auto tag_offsets = RotatedVectors(tag_offset);

    for (size_t corner_idx = 0; corner_idx < canonical_main.size(); ++corner_idx) {
        const cv::Point2f& corner = canonical_main[corner_idx];
        for (size_t rot_idx = 0; rot_idx < tag_offsets.size(); ++rot_idx) {
            const cv::Point2f& v  = tag_offsets[rot_idx];
            cv::Point2f candidate = corner + v;
            if (candidate.x < 0.0f || candidate.y < 0.0f ||
                candidate.x > static_cast<float>(board_w) ||
                candidate.y > static_cast<float>(board_h)) {
                continue;
            }

            cv::Point2f guess = ProjectPoint(H_board_to_img, candidate);
            float expected_r  = EstimateRadiusPx(H_board_to_img, candidate, tag_r);
            float search_r    = std::max(10.0f, expected_r * 2.5f);
            int tx0           = std::max(0, static_cast<int>(std::floor(guess.x - search_r)));
            int ty0           = std::max(0, static_cast<int>(std::floor(guess.y - search_r)));
            int tx1 = std::min(gray.cols - 1, static_cast<int>(std::ceil(guess.x + search_r)));
            int ty1 = std::min(gray.rows - 1, static_cast<int>(std::ceil(guess.y + search_r)));
            if (tx1 <= tx0 || ty1 <= ty0) { continue; }

            cv::Rect tag_roi(tx0, ty0, tx1 - tx0 + 1, ty1 - ty0 + 1);
            const std::string debug_name =
                "hole_tag_c" + std::to_string(corner_idx) + "_r" + std::to_string(rot_idx);
            CircleResult circle = DetectCircleInRoi(gray(tag_roi), expected_r, debug_name);
            if (!circle.ok) { continue; }

            circle.center.x += static_cast<float>(tag_roi.x);
            circle.center.y += static_cast<float>(tag_roi.y);

            cv::Point2f d = circle.center - guess;
            float error   = std::sqrt(d.x * d.x + d.y * d.y);
            float score   = error / std::max(1.0f, expected_r);
            if (score < best_tag_score) {
                best_tag_score = score;
                tag_circle     = circle;
            }
        }
    }

    if (!tag_circle.ok) { throw std::runtime_error("Failed to detect tag hole"); }

#ifdef DEBUG_DIR
    cv::Mat holes_vis = bgr.clone();
    for (size_t i = 0; i < main_centers.size(); ++i) {
        int r = (i < main_radii.size()) ? static_cast<int>(std::lround(main_radii[i])) : 6;
        cv::circle(holes_vis, main_centers[i], std::max(2, r), cv::Scalar(0, 255, 0), 2);
    }
    cv::circle(holes_vis, tag_circle.center,
               std::max(2, static_cast<int>(std::lround(tag_circle.radius))), cv::Scalar(0, 0, 255),
               2);
    SaveDebugImage("05_holes.png", holes_vis);
#endif

    std::vector<cv::Point2f> ordered_main = OrderMainByTag(main_centers, tag_circle.center);

    std::vector<cv::Point2f> canonical_main_ordered = {
        cv::Point2f(offset, offset),
        cv::Point2f(static_cast<float>(board_w) - offset, offset),
        cv::Point2f(static_cast<float>(board_w) - offset, static_cast<float>(board_h) - offset),
        cv::Point2f(offset, static_cast<float>(board_h) - offset),
    };

    H_img_to_board = cv::getPerspectiveTransform(ordered_main, canonical_main_ordered);

    cv::Mat board;
    cv::warpPerspective(bgr, board, H_img_to_board, cv::Size(board_w, board_h), cv::INTER_LINEAR,
                        cv::BORDER_REPLICATE);

#ifdef DEBUG_DIR
    SaveDebugImage("06_warped_board.png", board);
#endif

    if (margin < 0 || margin + color_w > board.cols || margin + color_h > board.rows) {
        throw std::runtime_error("Color region bounds are invalid");
    }
    cv::Rect color_roi(margin, margin, color_w, color_h);
    cv::Mat color = board(color_roi).clone();

#ifdef DEBUG_DIR
    SaveDebugImage("07_color_region.png", color);
#endif

    return color;
}

} // namespace ChromaPrint3D