#include "chromaprint3d/calib.h"
#include "chromaprint3d/error.h"
#include "detail/cv_utils.h"

#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// #define DEBUG_DIR "/tmp/calib_debug"

#ifdef DEBUG_DIR
#    include <filesystem>
#endif

namespace ChromaPrint3D {
namespace {

constexpr float kPi              = 3.14159265358979323846f;
constexpr int kTargetDim         = 1200;
constexpr double kCannySpread    = 0.33;
constexpr float kAspectMax       = 1.6f;
constexpr double kRectangularity = 0.7;
constexpr float kAngleDeviation  = 25.0f;
constexpr double kAreaMinRatio   = 0.08;
constexpr double kAreaMaxRatio   = 0.98;

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
    } catch (const std::exception& e) {
        spdlog::debug("SaveDebugImage failed: {}", e.what());
    }
}
#endif


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
    double lower   = std::max(0.0, (1.0 - kCannySpread) * v);
    double upper   = std::min(255.0, (1.0 + kCannySpread) * v);
    if (upper < 10.0) {
        lower = 30.0;
        upper = 90.0;
    }
    cv::Mat edges;
    cv::Canny(gray, edges, lower, upper);
    return edges;
}

static float AngleDegrees(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c) {
    const cv::Point2f v1 = a - b;
    const cv::Point2f v2 = c - b;
    const float l1       = std::sqrt(v1.x * v1.x + v1.y * v1.y);
    const float l2       = std::sqrt(v2.x * v2.x + v2.y * v2.y);
    if (l1 < 1e-3f || l2 < 1e-3f) { return 180.0f; }
    float cosang = (v1.x * v2.x + v1.y * v2.y) / (l1 * l2);
    cosang       = std::clamp(cosang, -1.0f, 1.0f);
    return std::acos(cosang) * 180.0f / kPi;
}

static bool EvaluateQuadCandidate(const std::vector<cv::Point>& quad, double contour_area,
                                  std::vector<cv::Point2f>& out, double& score) {
    if (quad.size() != 4) { return false; }
    if (!cv::isContourConvex(quad)) { return false; }

    cv::RotatedRect rect = cv::minAreaRect(quad);
    const float w        = rect.size.width;
    const float h        = rect.size.height;
    if (w <= 1.0f || h <= 1.0f) { return false; }

    const float ratio = std::max(w, h) / std::min(w, h);
    if (ratio > kAspectMax) { return false; }

    const double rect_area = static_cast<double>(w) * static_cast<double>(h);
    if (rect_area <= 1e-3) { return false; }

    const double rectangularity = std::min(1.0, contour_area / rect_area);
    if (rectangularity < kRectangularity) { return false; }

    float max_dev = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const cv::Point2f p0 = quad[(i + 3) % 4];
        const cv::Point2f p1 = quad[i];
        const cv::Point2f p2 = quad[(i + 1) % 4];
        const float angle    = AngleDegrees(p0, p1, p2);
        max_dev              = std::max(max_dev, std::abs(angle - 90.0f));
    }
    if (max_dev > kAngleDeviation) { return false; }

    out.clear();
    out.reserve(4);
    for (const auto& p : quad) {
        out.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    }

    const double angle_weight = 1.0 - std::min(1.0, static_cast<double>(max_dev) / 45.0);
    score                     = contour_area * rectangularity * angle_weight;
    return true;
}

static bool TryCoarseLocateByEdges(const cv::Mat& small, const cv::Mat& edges,
                                   std::vector<cv::Point2f>& corners) {
    if (edges.empty() || small.empty()) { return false; }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) { return false; }

    const double img_area = static_cast<double>(small.cols) * static_cast<double>(small.rows);
    const double min_area = img_area * kAreaMinRatio;
    const double max_area = img_area * kAreaMaxRatio;

    double best_score = -1.0;
    std::vector<cv::Point2f> best_quad;

    for (const auto& contour : contours) {
        const double area = std::abs(cv::contourArea(contour));
        if (area < min_area || area > max_area) { continue; }

        const double peri = cv::arcLength(contour, true);
        if (peri <= 0.0) { continue; }

        std::vector<cv::Point2f> quad;
        double score = -1.0;
        std::vector<cv::Point> approx;

        cv::approxPolyDP(contour, approx, 0.02 * peri, true);
        bool ok = EvaluateQuadCandidate(approx, area, quad, score);
        if (!ok) {
            cv::approxPolyDP(contour, approx, 0.01 * peri, true);
            ok = EvaluateQuadCandidate(approx, area, quad, score);
        }

        if (!ok) {
            cv::RotatedRect rect = cv::minAreaRect(contour);
            const float w        = rect.size.width;
            const float h        = rect.size.height;
            if (w > 1.0f && h > 1.0f) {
                const float ratio           = std::max(w, h) / std::min(w, h);
                const double rect_area      = static_cast<double>(w) * static_cast<double>(h);
                const double rectangularity = (rect_area > 0.0) ? (area / rect_area) : 0.0;
                if (ratio <= 1.8f && rectangularity >= 0.6) {
                    cv::Point2f pts[4];
                    rect.points(pts);
                    quad.assign(pts, pts + 4);
                    score = area * rectangularity * 0.8;
                    ok    = true;
                }
            }
        }

        if (ok && score > best_score) {
            best_score = score;
            best_quad  = quad;
        }
    }

    if (best_score <= 0.0 || best_quad.size() != 4) { return false; }

#ifdef DEBUG_DIR
    cv::Mat contour_vis = small.clone();
    for (const auto& c : contours) {
        cv::drawContours(contour_vis, std::vector{c}, -1, {0, 255, 0}, 1);
    }
    for (int i = 0; i < 4; ++i) {
        cv::line(contour_vis, best_quad[i], best_quad[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
    }
    SaveDebugImage("04_coarse_rect.png", contour_vis);
#endif

    corners = best_quad;
    return true;
}

static std::vector<cv::Point2f> OrderCorners(const std::vector<cv::Point2f>& pts) {
    if (pts.size() != 4) { throw InputError("OrderCorners expects 4 points"); }
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
    if (width <= 0 || height <= 0) {
        throw InputError("Image is empty or unreadable; please check the uploaded image file");
    }

    const int max_dim = std::max(width, height);
    scale_out = (max_dim > kTargetDim) ? (static_cast<double>(kTargetDim) / max_dim) : 1.0;

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

    std::vector<cv::Point2f> corners;
    if (!TryCoarseLocateByEdges(small, edges, corners)) {
        throw InputError(
            "Failed to locate calibration board outline in the image. "
            "Ensure: (1) the full board is visible; (2) there is good contrast with the background; "
            "(3) the shooting angle is correct and lighting is even");
    }

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
    if (centers.size() != 4) { throw InputError("Need 4 main holes"); }
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

cv::Mat LocateCalibrationColorRegion(const cv::Mat& input, const CalibrationBoardMeta& meta) {
    if (input.empty()) {
        throw InputError("Input image is empty; please check the uploaded image");
    }
    cv::Mat bgr = detail::EnsureBgr(input);
    if (bgr.empty()) {
        throw InputError("Image format conversion failed; ensure the uploaded file is a valid RGB/BGR image");
    }

#ifdef DEBUG_DIR
    SaveDebugImage("00_input.png", bgr);
#endif

    const int grid_rows = meta.grid_rows;
    const int grid_cols = meta.grid_cols;
    if (grid_rows <= 0 || grid_cols <= 0) {
        throw InputError(
            "Invalid grid dimensions in meta (grid_rows=" + std::to_string(grid_rows) +
            ", grid_cols=" + std::to_string(grid_cols) + "); check the meta JSON file");
    }

    int scale = meta.config.layout.resolution_scale;
    if (scale <= 0) { scale = 1; }

    const int tile   = meta.config.layout.tile_factor * scale;
    const int gap    = meta.config.layout.gap_factor * scale;
    const int margin = meta.config.layout.margin_factor * scale;
    if (tile <= 0) {
        throw InputError("Invalid tile_factor in meta; check the meta JSON file");
    }
    if (gap < 0 || margin < 0) {
        throw InputError("Invalid gap_factor or margin_factor in meta; check the meta JSON file");
    }

    const int color_w = grid_cols * tile + (grid_cols - 1) * gap;
    const int color_h = grid_rows * tile + (grid_rows - 1) * gap;
    const int board_w = color_w + 2 * margin;
    const int board_h = color_h + 2 * margin;
    if (board_w <= 0 || board_h <= 0) {
        throw InputError("Computed board dimensions are invalid; check the meta JSON file");
    }

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
        if (x1 <= x0 || y1 <= y0) {
            throw InputError(
                "Search area for main fiducial hole exceeds image bounds; "
                "ensure the photo contains the full calibration board with all four corners visible");
        }

        cv::Rect roi(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
        const std::string debug_name = "hole_main_" + std::to_string(i);
        CircleResult circle          = DetectCircleInRoi(gray(roi), expected_r, debug_name);
        if (!circle.ok) {
            throw InputError(
                "Failed to detect main fiducial hole #" + std::to_string(i + 1) +
                ". Ensure: (1) the corner holes are clearly visible; "
                "(2) lighting is even without strong reflections; "
                "(3) the photo matches the meta file's calibration board");
        }

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

    if (!tag_circle.ok) {
        throw InputError(
            "Failed to detect the tag fiducial hole (the small hole indicating board orientation). "
            "Ensure all four corners are fully visible and the photo matches the meta file");
    }

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
        throw InputError("Failed to extract color region: computed area exceeds image bounds");
    }
    cv::Rect color_roi(margin, margin, color_w, color_h);
    cv::Mat color = board(color_roi).clone();

#ifdef DEBUG_DIR
    SaveDebugImage("07_color_region.png", color);
#endif

    return color;
}

cv::Mat LocateCalibrationColorRegion(const std::string& image_path,
                                     const CalibrationBoardMeta& meta) {
    cv::Mat input = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (input.empty()) { throw IOError("Failed to read image: " + image_path); }
    return LocateCalibrationColorRegion(input, meta);
}

} // namespace ChromaPrint3D