#include "gradient.h"
#include "chromaprint3d/color.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ChromaPrint3D::detail {

namespace {

float ApplySpreadMethod(float t, SpreadMethod spread) {
    switch (spread) {
    case SpreadMethod::Repeat:
        t = t - std::floor(t);
        break;
    case SpreadMethod::Reflect:
        t = 1.0f - std::abs(std::fmod(t, 2.0f) - 1.0f);
        break;
    case SpreadMethod::Pad:
    default:
        t = std::clamp(t, 0.0f, 1.0f);
        break;
    }
    return t;
}

Rgb InterpolateGradient(const GradientInfo& grad, float t) {
    if (grad.stops.empty()) { return Rgb(0.5f, 0.5f, 0.5f); }

    t = ApplySpreadMethod(t, grad.spread);

    if (t <= grad.stops.front().offset) {
        const auto& s = grad.stops.front();
        return Rgb(s.color.x * s.opacity, s.color.y * s.opacity, s.color.z * s.opacity);
    }
    if (t >= grad.stops.back().offset) {
        const auto& s = grad.stops.back();
        return Rgb(s.color.x * s.opacity, s.color.y * s.opacity, s.color.z * s.opacity);
    }

    for (size_t i = 0; i + 1 < grad.stops.size(); ++i) {
        if (t >= grad.stops[i].offset && t <= grad.stops[i + 1].offset) {
            float range   = grad.stops[i + 1].offset - grad.stops[i].offset;
            float frac    = (range > 0.0f) ? (t - grad.stops[i].offset) / range : 0.0f;
            float a0      = grad.stops[i].opacity;
            float a1      = grad.stops[i + 1].opacity;
            float a       = a0 + (a1 - a0) * frac;
            const Rgb& c0 = grad.stops[i].color;
            const Rgb& c1 = grad.stops[i + 1].color;
            return Rgb((c0.x + (c1.x - c0.x) * frac) * a, (c0.y + (c1.y - c0.y) * frac) * a,
                       (c0.z + (c1.z - c0.z) * frac) * a);
        }
    }
    const auto& s = grad.stops.back();
    return Rgb(s.color.x * s.opacity, s.color.y * s.opacity, s.color.z * s.opacity);
}

} // namespace

bool PointInContours(const std::vector<Contour>& contours, float px, float py) {
    int crossings = 0;
    for (const Contour& c : contours) {
        size_t n = c.size();
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            float yi = c[i].y, yj = c[j].y;
            if ((yi <= py && yj > py) || (yj <= py && yi > py)) {
                float xi = c[i].x, xj = c[j].x;
                float t = (py - yi) / (yj - yi);
                if (px < xi + t * (xj - xi)) { ++crossings; }
            }
        }
    }
    return (crossings % 2) == 1;
}

cv::Mat RasterizeGradientShape(const VectorShape& shape, float pixel_mm, float& out_offset_x,
                               float& out_offset_y) {
    if (shape.contours.empty() || pixel_mm <= 0.0f) { return {}; }

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for (const Contour& c : shape.contours) {
        for (const Vec2f& p : c) {
            min_x = std::min(min_x, p.x);
            min_y = std::min(min_y, p.y);
            max_x = std::max(max_x, p.x);
            max_y = std::max(max_y, p.y);
        }
    }

    int w = std::max(1, static_cast<int>(std::ceil((max_x - min_x) / pixel_mm)));
    int h = std::max(1, static_cast<int>(std::ceil((max_y - min_y) / pixel_mm)));

    out_offset_x = min_x;
    out_offset_y = min_y;

    cv::Mat result(h, w, CV_32FC4, cv::Scalar(0, 0, 0, 0));

    const GradientInfo& grad = shape.gradient;
    bool is_radial           = (shape.fill_type == FillType::RadialGradient);

#pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < h; ++row) {
        float py  = min_y + (static_cast<float>(row) + 0.5f) * pixel_mm;
        auto* ptr = result.ptr<cv::Vec4f>(row);
        for (int col = 0; col < w; ++col) {
            float px = min_x + (static_cast<float>(col) + 0.5f) * pixel_mm;

            if (!PointInContours(shape.contours, px, py)) { continue; }

            float t;
            if (is_radial) {
                float dx   = px - grad.x1;
                float dy   = py - grad.y1;
                float dist = std::sqrt(dx * dx + dy * dy);
                t          = (grad.r > 0.0f) ? dist / grad.r : 0.0f;
            } else {
                float gx    = grad.x2 - grad.x1;
                float gy    = grad.y2 - grad.y1;
                float glen2 = gx * gx + gy * gy;
                if (glen2 > 0.0f) {
                    t = ((px - grad.x1) * gx + (py - grad.y1) * gy) / glen2;
                } else {
                    t = 0.0f;
                }
            }

            Rgb color = InterpolateGradient(grad, t);
            ptr[col]  = cv::Vec4f(color.r(), color.g(), color.b(), 1.0f);
        }
    }

    return result;
}

} // namespace ChromaPrint3D::detail
