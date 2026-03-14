#include "path_metrics.h"

#include "svg_geometry.h"

#include <nanosvg/nanosvg.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

namespace ChromaPrint3D::eval {

namespace {

struct PathInfo {
    double signed_area = 0;
    double perimeter   = 0;
    float bounds[4]    = {};
    std::vector<cv::Point> polyline;
    bool is_hole = false;
};

// Green's theorem signed area for cubic bezier segment via subdivision
double BezierSegmentArea(float x0, float y0, float x1, float y1, float x2, float y2, float x3,
                         float y3) {
    // Use 4-point Gaussian quadrature approximation of ∫ x dy along the curve.
    // Simpler: just use polygon area from flattened points.
    // We accumulate the shoelace contribution from each subdivision segment.
    return 0; // accumulated via polyline
}

double BezierSegmentLength(float x0, float y0, float x1, float y1, float x2, float y2, float x3,
                           float y3, int depth = 0) {
    if (depth > 8) {
        float dx = x3 - x0, dy = y3 - y0;
        return std::sqrt(dx * dx + dy * dy);
    }
    float dx = x3 - x0, dy = y3 - y0;
    float chord       = std::sqrt(dx * dx + dy * dy);
    float d1          = std::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
    float d2          = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    float d3          = std::sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2));
    float control_len = d1 + d2 + d3;
    if (control_len - chord < 0.5f && depth > 2) return (control_len + chord) * 0.5f;

    float x01 = (x0 + x1) * 0.5f, y01 = (y0 + y1) * 0.5f;
    float x12 = (x1 + x2) * 0.5f, y12 = (y1 + y2) * 0.5f;
    float x23 = (x2 + x3) * 0.5f, y23 = (y2 + y3) * 0.5f;
    float xa = (x01 + x12) * 0.5f, ya = (y01 + y12) * 0.5f;
    float xb = (x12 + x23) * 0.5f, yb = (y12 + y23) * 0.5f;
    float xm = (xa + xb) * 0.5f, ym = (ya + yb) * 0.5f;
    return BezierSegmentLength(x0, y0, x01, y01, xa, ya, xm, ym, depth + 1) +
           BezierSegmentLength(xm, ym, xb, yb, x23, y23, x3, y3, depth + 1);
}

void FlattenPath(const float* pts, int npts, std::vector<cv::Point>& out, float sx, float sy) {
    if (npts < 1) return;
    out.emplace_back(static_cast<int>(std::round(pts[0] * sx)),
                     static_cast<int>(std::round(pts[1] * sy)));
    for (int i = 0; i < npts - 1; i += 3) {
        const float* p = &pts[i * 2];
        // Flatten cubic bezier to polyline (same algo as svg_rasterizer)
        auto addPt = [&](float x, float y) {
            out.emplace_back(static_cast<int>(std::round(x * sx)),
                             static_cast<int>(std::round(y * sy)));
        };
        // Simple uniform subdivision with 8 steps per segment
        for (int s = 1; s <= 8; ++s) {
            float t = s / 8.0f;
            float u = 1.0f - t;
            float x =
                u * u * u * p[0] + 3 * u * u * t * p[2] + 3 * u * t * t * p[4] + t * t * t * p[6];
            float y =
                u * u * u * p[1] + 3 * u * u * t * p[3] + 3 * u * t * t * p[5] + t * t * t * p[7];
            addPt(x, y);
        }
    }
}

bool BBoxContains(const float outer[4], const float inner[4]) {
    return inner[0] >= outer[0] && inner[1] >= outer[1] && inner[2] <= outer[2] &&
           inner[3] <= outer[3];
}

cv::Point BBoxCenter(const float b[4]) {
    return {static_cast<int>((b[0] + b[2]) * 0.5f), static_cast<int>((b[1] + b[3]) * 0.5f)};
}

uint32_t PackColor(unsigned int c) { return c & 0x00FFFFFF; }

int PercentileInt(std::vector<int>& v, double p) {
    if (v.empty()) return 0;
    size_t idx = static_cast<size_t>(v.size() * p);
    if (idx >= v.size()) idx = v.size() - 1;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(idx), v.end());
    return v[idx];
}

double PercentileDouble(std::vector<double>& v, double p) {
    if (v.empty()) return 0;
    size_t idx = static_cast<size_t>(v.size() * p);
    if (idx >= v.size()) idx = v.size() - 1;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(idx), v.end());
    return v[idx];
}

double GiniCoefficient(std::vector<double>& areas) {
    if (areas.size() < 2) return 0;
    std::sort(areas.begin(), areas.end());
    double n   = static_cast<double>(areas.size());
    double sum = 0, cumulative = 0;
    for (size_t i = 0; i < areas.size(); ++i) {
        cumulative += areas[i];
        sum += (2.0 * (i + 1) - n - 1.0) * areas[i];
    }
    if (cumulative < 1e-12) return 0;
    return sum / (n * cumulative);
}

} // namespace

PathMetricsResult ComputePathMetrics(const std::string& svg_content, int width, int height,
                                     double tiny_area_threshold, double sliver_threshold) {
    PathMetricsResult r;

    std::vector<char> buf(svg_content.begin(), svg_content.end());
    buf.push_back('\0');
    NSVGimage* img = nsvgParse(buf.data(), "px", 96.0f);
    if (!img) return r;
    auto guard = std::unique_ptr<NSVGimage, decltype(&nsvgDelete)>(img, nsvgDelete);

    float sx = (img->width > 0) ? static_cast<float>(width) / img->width : 1.0f;
    float sy = (img->height > 0) ? static_cast<float>(height) / img->height : 1.0f;

    std::set<uint32_t> color_set;
    std::vector<int> complexities;
    std::vector<double> net_areas;
    std::vector<double> circularities;
    int tiny_count = 0;

    // Per-shape color + bounding box for mergeable/island/gap analysis
    struct ShapeInfo {
        uint32_t color  = 0;
        float bounds[4] = {};
        double net_area = 0;
    };

    std::vector<ShapeInfo> shape_infos;

    for (NSVGshape* shape = img->shapes; shape; shape = shape->next) {
        if (!(shape->flags & NSVG_FLAGS_VISIBLE)) continue;
        r.total_shapes++;

        uint32_t col = 0;
        if (shape->fill.type == NSVG_PAINT_COLOR) col = PackColor(shape->fill.color);
        color_set.insert(col);

        // Collect path infos for this shape
        std::vector<PathInfo> paths;
        int shape_complexity = 0;

        for (NSVGpath* path = shape->paths; path; path = path->next) {
            PathInfo pi;
            std::memcpy(pi.bounds, path->bounds, sizeof(pi.bounds));
            // Scale bounds
            pi.bounds[0] *= sx;
            pi.bounds[1] *= sy;
            pi.bounds[2] *= sx;
            pi.bounds[3] *= sy;

            int nsegs = (path->npts > 1) ? (path->npts - 1) / 3 : 0;
            shape_complexity += nsegs;

            if (path->closed && path->npts >= 4) {
                FlattenPath(path->pts, path->npts, pi.polyline, sx, sy);
                pi.signed_area = PolylineSignedArea(pi.polyline);
                pi.perimeter   = PolylinePerimeter(pi.polyline);
            } else {
                // Accumulate arc length for unclosed paths
                for (int i = 0; i < path->npts - 1; i += 3) {
                    const float* p = &path->pts[i * 2];
                    pi.perimeter += BezierSegmentLength(p[0] * sx, p[1] * sy, p[2] * sx, p[3] * sy,
                                                        p[4] * sx, p[5] * sy, p[6] * sx, p[7] * sy);
                }
            }
            paths.push_back(std::move(pi));
        }
        complexities.push_back(shape_complexity);

        // Hole detection via geometric containment (sorted by |area| desc)
        std::sort(paths.begin(), paths.end(), [](const PathInfo& a, const PathInfo& b) {
            return std::abs(a.signed_area) > std::abs(b.signed_area);
        });

        for (size_t i = 0; i < paths.size(); ++i) {
            if (paths[i].polyline.empty()) continue;
            if (i == 0) {
                paths[i].is_hole = false;
                continue;
            }
            // Find smallest containing path
            cv::Point center = BBoxCenter(paths[i].bounds);
            bool found       = false;
            for (size_t j = 0; j < i; ++j) {
                if (paths[j].polyline.empty()) continue;
                if (!BBoxContains(paths[j].bounds, paths[i].bounds)) continue;
                if (PointInPolyline(paths[j].polyline, center)) {
                    paths[i].is_hole = !paths[j].is_hole; // evenodd alternation
                    found            = true;
                    break;
                }
            }
            if (!found) paths[i].is_hole = false;
        }

        double outer_area = 0, hole_area = 0, total_perim = 0;
        for (auto& pi : paths) {
            double a = std::abs(pi.signed_area);
            if (pi.is_hole)
                hole_area += a;
            else
                outer_area += a;
            total_perim += pi.perimeter;
        }
        double net = outer_area - hole_area;
        net_areas.push_back(net);

        if (total_perim > 0) {
            double circ = 4.0 * M_PI * net / (total_perim * total_perim);
            circularities.push_back(circ);
            if (circ < sliver_threshold && net > 1.0) r.sliver_count++;
        }

        if (net < tiny_area_threshold) tiny_count++;

        ShapeInfo si;
        si.color = col;
        std::memcpy(si.bounds, shape->bounds, sizeof(si.bounds));
        si.bounds[0] *= sx;
        si.bounds[1] *= sy;
        si.bounds[2] *= sx;
        si.bounds[3] *= sy;
        si.net_area = net;
        shape_infos.push_back(si);
    }

    r.unique_colors = static_cast<int>(color_set.size());
    r.color_compression =
        (r.total_shapes > 0) ? static_cast<double>(r.unique_colors) / r.total_shapes : 0;

    if (r.total_shapes > 0) r.tiny_fragment_rate = static_cast<double>(tiny_count) / r.total_shapes;

    r.path_complexity_median = complexities.empty() ? 0 : PercentileInt(complexities, 0.5);
    r.path_complexity_p95    = complexities.empty() ? 0 : PercentileInt(complexities, 0.95);

    r.circularity_p95 = circularities.empty() ? 0 : PercentileDouble(circularities, 0.95);

    r.gini_coefficient = GiniCoefficient(net_areas);

    // Island detection: tiny shapes whose bbox doesn't overlap any same-color shape
    for (size_t i = 0; i < shape_infos.size(); ++i) {
        if (shape_infos[i].net_area >= tiny_area_threshold) continue;
        bool has_neighbor = false;
        for (size_t j = 0; j < shape_infos.size(); ++j) {
            if (i == j) continue;
            if (shape_infos[i].color != shape_infos[j].color) continue;
            // Check bbox overlap
            if (shape_infos[i].bounds[0] <= shape_infos[j].bounds[2] &&
                shape_infos[i].bounds[2] >= shape_infos[j].bounds[0] &&
                shape_infos[i].bounds[1] <= shape_infos[j].bounds[3] &&
                shape_infos[i].bounds[3] >= shape_infos[j].bounds[1]) {
                has_neighbor = true;
                break;
            }
        }
        if (!has_neighbor) r.island_count++;
    }

    // Mergeable ratio and same-color gap via rasterization + connected components
    if (r.total_shapes > 0 && width > 0 && height > 0) {
        // Build per-color label map
        std::map<uint32_t, std::vector<size_t>> color_shapes;
        for (size_t i = 0; i < shape_infos.size(); ++i) {
            color_shapes[shape_infos[i].color].push_back(i);
        }

        // Re-parse SVG to get shape polylines for rasterization
        std::vector<char> buf2(svg_content.begin(), svg_content.end());
        buf2.push_back('\0');
        NSVGimage* img2 = nsvgParse(buf2.data(), "px", 96.0f);
        if (img2) {
            auto g2 = std::unique_ptr<NSVGimage, decltype(&nsvgDelete)>(img2, nsvgDelete);

            // Rasterize shapes to per-color masks
            int shape_idx = 0;
            std::map<uint32_t, cv::Mat> color_masks;
            for (NSVGshape* shape = img2->shapes; shape; shape = shape->next) {
                if (!(shape->flags & NSVG_FLAGS_VISIBLE)) { continue; }
                uint32_t col =
                    (shape->fill.type == NSVG_PAINT_COLOR) ? PackColor(shape->fill.color) : 0;
                if (color_masks.find(col) == color_masks.end()) {
                    color_masks[col] = cv::Mat::zeros(height, width, CV_8UC1);
                }

                std::vector<std::vector<cv::Point>> contours;
                for (NSVGpath* path = shape->paths; path; path = path->next) {
                    if (path->npts < 4) continue;
                    std::vector<cv::Point> pts;
                    FlattenPath(path->pts, path->npts, pts, sx, sy);
                    if (pts.size() >= 3) contours.push_back(std::move(pts));
                }
                if (!contours.empty()) {
                    cv::Mat shape_mask = FillShapeWithHoles(contours, width, height);
                    color_masks[col].setTo(255, shape_mask);
                }
                shape_idx++;
            }

            int total_connected = 0;
            int total_gap       = 0;
            for (auto& [col, mask] : color_masks) {
                cv::Mat labels;
                int n = cv::connectedComponents(mask, labels, 8, CV_32S);
                total_connected += (n - 1); // subtract background

                // Gap: pixels in bbox of same-color shapes that are NOT covered
                // (Simple approach: dilate mask, count difference)
                cv::Mat dilated;
                cv::dilate(mask, dilated,
                           cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
                cv::Mat gap;
                cv::subtract(dilated, mask, gap);
                total_gap += cv::countNonZero(gap);
            }

            r.mergeable_ratio =
                (r.total_shapes > 0) ? static_cast<double>(total_connected) / r.total_shapes : 1.0;
            r.same_color_gap_pixels = total_gap;
        }
    }

    return r;
}

} // namespace ChromaPrint3D::eval
