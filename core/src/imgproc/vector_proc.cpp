#include "chromaprint3d/vector_proc.h"
#include "chromaprint3d/color.h"
#include "chromaprint3d/error.h"
#include "bezier.h"
#include "clipper_scale.h"
#include "gradient.h"
#include "occlusion.h"

#include <clipper2/clipper.h>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <lunasvg.h>
#include "svgelement.h"
#include "svggeometryelement.h"
#include "svgpaintelement.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <string>

#if defined(_OPENMP)
#    if defined(__has_include)
#        if __has_include(<omp.h>)
#            include <omp.h>
#        endif
#    else
#        include <omp.h>
#    endif
#endif

namespace ChromaPrint3D {

using detail::kClipperScale;

namespace {

struct NormalizeContoursStats {
    std::atomic<int64_t> convert_us{0};
    std::atomic<int64_t> union_us{0};
    std::atomic<int64_t> simplify_us{0};
    std::atomic<int64_t> filter_us{0};
    std::atomic<int64_t> sort_us{0};
    std::atomic<int64_t> calls{0};

    void Reset() {
        convert_us  = 0;
        union_us    = 0;
        simplify_us = 0;
        filter_us   = 0;
        sort_us     = 0;
        calls       = 0;
    }

    void Log(const char* label) const {
        int64_t n = calls.load(std::memory_order_relaxed);
        if (n == 0) return;
        spdlog::debug("[perf] NormalizeContours ({}): {} calls, "
                      "convert={:.1f} ms, union={:.1f} ms, simplify={:.1f} ms, "
                      "filter={:.1f} ms, sort={:.1f} ms",
                      label, n, convert_us.load() / 1000.0, union_us.load() / 1000.0,
                      simplify_us.load() / 1000.0, filter_us.load() / 1000.0,
                      sort_us.load() / 1000.0);
    }
};

static NormalizeContoursStats g_normalize_stats;

static Clipper2Lib::Path64 ContourToPath(const Contour& contour) {
    Clipper2Lib::Path64 path;
    path.reserve(contour.size());
    for (const Vec2f& p : contour) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) continue;
        path.emplace_back(static_cast<int64_t>(std::llround(p.x * kClipperScale)),
                          static_cast<int64_t>(std::llround(p.y * kClipperScale)));
    }
    return path;
}

static Contour PathToContour(const Clipper2Lib::Path64& path) {
    Contour contour;
    contour.reserve(path.size());
    for (const auto& p : path) {
        contour.emplace_back(static_cast<float>(p.x / kClipperScale),
                             static_cast<float>(p.y / kClipperScale));
    }
    return contour;
}

static Clipper2Lib::FillRule ToClipperFillRule(SvgFillRule rule) {
    return (rule == SvgFillRule::EvenOdd) ? Clipper2Lib::FillRule::EvenOdd
                                          : Clipper2Lib::FillRule::NonZero;
}

static std::vector<Contour> NormalizeContours(const std::vector<Contour>& contours,
                                              float tolerance_mm, Clipper2Lib::FillRule fill_rule) {
    std::vector<Contour> normalized;
    if (contours.empty()) return normalized;

    double simplify_eps = std::max(100.0, static_cast<double>(tolerance_mm) * kClipperScale * 0.01);
    double min_area_scaled =
        std::max(1e-5, static_cast<double>(tolerance_mm) * tolerance_mm * 0.01) * kClipperScale *
        kClipperScale;

    auto t0 = std::chrono::steady_clock::now();

    Clipper2Lib::Paths64 all_paths;
    all_paths.reserve(contours.size());
    for (const Contour& contour : contours) {
        Clipper2Lib::Path64 path = ContourToPath(contour);
        if (path.size() >= 3) { all_paths.push_back(std::move(path)); }
    }
    if (all_paths.empty()) return normalized;

    auto t1 = std::chrono::steady_clock::now();

    Clipper2Lib::Paths64 cleaned = Clipper2Lib::Union(all_paths, fill_rule);
    if (cleaned.empty()) return normalized;

    auto t2 = std::chrono::steady_clock::now();

    cleaned = Clipper2Lib::SimplifyPaths(cleaned, simplify_eps, true);

    auto t3 = std::chrono::steady_clock::now();

    std::vector<double> areas;
    areas.reserve(cleaned.size());
    for (auto& cpath : cleaned) {
        if (cpath.size() < 3) continue;
        double a = std::abs(Clipper2Lib::Area(cpath));
        if (a < min_area_scaled) continue;
        areas.push_back(a);
        normalized.push_back(PathToContour(cpath));
    }

    auto t4 = std::chrono::steady_clock::now();

    std::vector<size_t> order(normalized.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return areas[a] > areas[b]; });
    std::vector<Contour> sorted;
    sorted.reserve(normalized.size());
    for (size_t k : order) { sorted.push_back(std::move(normalized[k])); }
    normalized = std::move(sorted);

    auto t5 = std::chrono::steady_clock::now();

    using Us = std::chrono::microseconds;
    g_normalize_stats.convert_us.fetch_add(std::chrono::duration_cast<Us>(t1 - t0).count(),
                                           std::memory_order_relaxed);
    g_normalize_stats.union_us.fetch_add(std::chrono::duration_cast<Us>(t2 - t1).count(),
                                         std::memory_order_relaxed);
    g_normalize_stats.simplify_us.fetch_add(std::chrono::duration_cast<Us>(t3 - t2).count(),
                                            std::memory_order_relaxed);
    g_normalize_stats.filter_us.fetch_add(std::chrono::duration_cast<Us>(t4 - t3).count(),
                                          std::memory_order_relaxed);
    g_normalize_stats.sort_us.fetch_add(std::chrono::duration_cast<Us>(t5 - t4).count(),
                                        std::memory_order_relaxed);
    g_normalize_stats.calls.fetch_add(1, std::memory_order_relaxed);

    return normalized;
}

static Rgb LunaSvgColorToRgb(const lunasvg::Color& color) {
    float r = color.redF();
    float g = color.greenF();
    float b = color.blueF();
    return Rgb{SrgbToLinear(r), SrgbToLinear(g), SrgbToLinear(b)};
}

static SpreadMethod ConvertSpreadMethod(lunasvg::SpreadMethod sm) {
    switch (sm) {
    case lunasvg::SpreadMethod::Reflect:
        return SpreadMethod::Reflect;
    case lunasvg::SpreadMethod::Repeat:
        return SpreadMethod::Repeat;
    default:
        return SpreadMethod::Pad;
    }
}

static void CollectGradientStops(const lunasvg::SVGGradientElement* contentElem,
                                 GradientInfo& info) {
    if (!contentElem) return;
    for (const auto& child : contentElem->children()) {
        auto* childElem = lunasvg::toSVGElement(child);
        if (childElem && childElem->id() == lunasvg::ElementID::Stop) {
            auto* stopElem = static_cast<const lunasvg::SVGStopElement*>(childElem);
            auto gstop     = stopElem->gradientStop(1.0f);
            GradientStop s;
            s.offset  = gstop.offset;
            s.opacity = gstop.color.a;
            s.color   = Rgb{SrgbToLinear(gstop.color.r), SrgbToLinear(gstop.color.g),
                          SrgbToLinear(gstop.color.b)};
            info.stops.push_back(s);
        }
    }
}

static GradientInfo ExtractGradientFromPaintElement(const lunasvg::SVGPaintElement* paintElem,
                                                    const lunasvg::SVGGeometryElement* geoElem,
                                                    FillType& out_fill_type) {
    GradientInfo info;
    out_fill_type = FillType::Solid;
    if (!paintElem) return info;

    auto eid = paintElem->id();

    if (eid == lunasvg::ElementID::LinearGradient) {
        auto* lg        = static_cast<const lunasvg::SVGLinearGradientElement*>(paintElem);
        auto attributes = lg->collectGradientAttributes();
        auto units      = attributes.gradientUnits();
        lunasvg::LengthContext lengthCtx(paintElem, units);

        out_fill_type = FillType::LinearGradient;
        float raw_x1  = lengthCtx.valueForLength(attributes.x1());
        float raw_y1  = lengthCtx.valueForLength(attributes.y1());
        float raw_x2  = lengthCtx.valueForLength(attributes.x2());
        float raw_y2  = lengthCtx.valueForLength(attributes.y2());

        auto gradTransform = attributes.gradientTransform();
        if (units == lunasvg::Units::ObjectBoundingBox) {
            auto bbox = geoElem->fillBoundingBox();
            gradTransform.postMultiply(lunasvg::Transform(bbox.w, 0, 0, bbox.h, bbox.x, bbox.y));
        }

        auto p1     = gradTransform.mapPoint(raw_x1, raw_y1);
        auto p2     = gradTransform.mapPoint(raw_x2, raw_y2);
        info.x1     = p1.x;
        info.y1     = p1.y;
        info.x2     = p2.x;
        info.y2     = p2.y;
        info.spread = ConvertSpreadMethod(attributes.spreadMethod());

        CollectGradientStops(attributes.gradientContentElement(), info);
    } else if (eid == lunasvg::ElementID::RadialGradient) {
        auto* rg        = static_cast<const lunasvg::SVGRadialGradientElement*>(paintElem);
        auto attributes = rg->collectGradientAttributes();
        auto units      = attributes.gradientUnits();
        lunasvg::LengthContext lengthCtx(paintElem, units);

        out_fill_type = FillType::RadialGradient;
        float raw_cx  = lengthCtx.valueForLength(attributes.cx());
        float raw_cy  = lengthCtx.valueForLength(attributes.cy());
        float raw_r   = lengthCtx.valueForLength(attributes.r());
        float raw_fx  = lengthCtx.valueForLength(attributes.fx());
        float raw_fy  = lengthCtx.valueForLength(attributes.fy());

        auto gradTransform = attributes.gradientTransform();
        if (units == lunasvg::Units::ObjectBoundingBox) {
            auto bbox = geoElem->fillBoundingBox();
            gradTransform.postMultiply(lunasvg::Transform(bbox.w, 0, 0, bbox.h, bbox.x, bbox.y));
        }

        auto pc     = gradTransform.mapPoint(raw_cx, raw_cy);
        auto pf     = gradTransform.mapPoint(raw_fx, raw_fy);
        info.x1     = pc.x;
        info.y1     = pc.y;
        info.r      = raw_r * std::sqrt(gradTransform.xScale() * gradTransform.yScale());
        info.x2     = pf.x;
        info.y2     = pf.y;
        info.spread = ConvertSpreadMethod(attributes.spreadMethod());

        CollectGradientStops(attributes.gradientContentElement(), info);
    }

    return info;
}

static lunasvg::Transform ComputeGlobalTransform(const lunasvg::SVGGeometryElement* geo) {
    auto transform = geo->localTransform();
    for (auto* parent = geo->parentElement(); parent; parent = parent->parentElement()) {
        transform.postMultiply(parent->localTransform());
    }
    return transform;
}

static bool IsIdentityTransform(const lunasvg::Transform& t) {
    constexpr float kEps = 1e-6f;
    const auto& m        = t.matrix();
    return std::abs(m.a - 1.0f) < kEps && std::abs(m.b) < kEps && std::abs(m.c) < kEps &&
           std::abs(m.d - 1.0f) < kEps && std::abs(m.e) < kEps && std::abs(m.f) < kEps;
}

static VectorShape ConvertGeometryElement(const lunasvg::SVGGeometryElement* geo, float tolerance) {
    VectorShape vs;

    const auto& paint = geo->fillPaintServer();
    if (!paint.isRenderable()) {
        if (geo->strokePaintServer().isRenderable()) {
            spdlog::warn("VectorProc: stroke-only shape skipped (not supported)");
        }
        return vs;
    }

    if (paint.element()) {
        FillType ft  = FillType::Solid;
        vs.gradient  = ExtractGradientFromPaintElement(paint.element(), geo, ft);
        vs.fill_type = ft;
        if (ft == FillType::Solid) {
            spdlog::warn("VectorProc: unsupported paint type (pattern fill), element skipped");
            return vs;
        }
    } else {
        const auto& color = paint.color();
        if (color.alpha() == 0) return vs;
        vs.fill_type  = FillType::Solid;
        vs.fill_color = LunaSvgColorToRgb(color);
    }

    vs.opacity       = geo->opacity() * paint.opacity();
    vs.svg_fill_rule = (geo->fill_rule() == lunasvg::FillRule::EvenOdd) ? SvgFillRule::EvenOdd
                                                                        : SvgFillRule::NonZero;

    const auto& path          = geo->path();
    lunasvg::Transform global = ComputeGlobalTransform(geo);
    const bool has_transform  = !IsIdentityTransform(global);

    if (has_transform && vs.fill_type != FillType::Solid) {
        auto& g = vs.gradient;
        auto p1 = global.mapPoint(g.x1, g.y1);
        auto p2 = global.mapPoint(g.x2, g.y2);
        g.x1    = p1.x;
        g.y1    = p1.y;
        g.x2    = p2.x;
        g.y2    = p2.y;
        if (vs.fill_type == FillType::RadialGradient) {
            g.r *= std::sqrt(global.xScale() * global.yScale());
        }
    }

    lunasvg::PathIterator it(path);
    std::array<lunasvg::Point, 3> pts;

    Contour current_contour;

    while (!it.isDone()) {
        auto cmd = it.currentSegment(pts);
        if (has_transform) {
            pts[0] = global.mapPoint(pts[0]);
            pts[1] = global.mapPoint(pts[1]);
            pts[2] = global.mapPoint(pts[2]);
        }
        switch (cmd) {
        case lunasvg::PathCommand::MoveTo:
            if (current_contour.size() >= 3) { vs.contours.push_back(std::move(current_contour)); }
            current_contour.clear();
            current_contour.push_back({pts[0].x, pts[0].y});
            break;
        case lunasvg::PathCommand::LineTo:
            current_contour.push_back({pts[0].x, pts[0].y});
            break;
        case lunasvg::PathCommand::CubicTo: {
            if (current_contour.empty()) { current_contour.push_back({0.0f, 0.0f}); }
            Vec2f p0 = current_contour.back();
            Vec2f p1{pts[0].x, pts[0].y};
            Vec2f p2{pts[1].x, pts[1].y};
            Vec2f p3{pts[2].x, pts[2].y};
            detail::FlattenCubicBezier(p0, p1, p2, p3, tolerance, current_contour);
            break;
        }
        case lunasvg::PathCommand::Close:
            if (current_contour.size() >= 3) {
                if (current_contour.front() == current_contour.back()) {
                    current_contour.pop_back();
                }
            }
            if (current_contour.size() >= 3) { vs.contours.push_back(std::move(current_contour)); }
            current_contour.clear();
            break;
        }
        it.next();
    }

    if (current_contour.size() >= 3) { vs.contours.push_back(std::move(current_contour)); }

    vs.contours = NormalizeContours(vs.contours, tolerance, ToClipperFillRule(vs.svg_fill_rule));
    return vs;
}

static std::string PathStem(const std::string& path) {
    if (path.empty()) { return {}; }
    return std::filesystem::path(path).stem().string();
}

static double ElapsedMs(std::chrono::steady_clock::time_point start) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
}

static float ColorDistSq(const Rgb& a, const Rgb& b) {
    float dr = a.r() - b.r();
    float dg = a.g() - b.g();
    float db = a.b() - b.b();
    return dr * dr + dg * dg + db * db;
}

/// Rasterize a gradient shape, quantize to gradient stop colors, extract contours,
/// and return a set of solid-color shapes that approximate the original gradient.
static std::vector<VectorShape> FlattenGradientShape(const VectorShape& shape, float pixel_mm,
                                                     float tolerance_mm) {
    std::vector<VectorShape> result;
    if (shape.fill_type == FillType::Solid || shape.contours.empty()) return result;

    const auto& stops = shape.gradient.stops;
    if (stops.empty()) return result;

    if (stops.size() == 1) {
        VectorShape solid = shape;
        solid.fill_type   = FillType::Solid;
        solid.fill_color  = stops.front().color;
        solid.gradient    = {};
        result.push_back(std::move(solid));
        return result;
    }

    float ox = 0.0f, oy = 0.0f;
    cv::Mat raster = detail::RasterizeGradientShape(shape, pixel_mm, ox, oy);
    if (raster.empty()) return result;

    const int h = raster.rows;
    const int w = raster.cols;

    cv::Mat label_map(h, w, CV_32S, cv::Scalar(-1));
    for (int row = 0; row < h; ++row) {
        const auto* rgba_ptr = raster.ptr<cv::Vec4f>(row);
        auto* lbl_ptr        = label_map.ptr<int32_t>(row);
        for (int col = 0; col < w; ++col) {
            const auto& px = rgba_ptr[col];
            if (px[3] == 0.0f) continue;
            Rgb pixel_color(px[0], px[1], px[2]);
            int best_idx    = 0;
            float best_dist = ColorDistSq(pixel_color, stops[0].color);
            for (size_t si = 1; si < stops.size(); ++si) {
                float d = ColorDistSq(pixel_color, stops[si].color);
                if (d < best_dist) {
                    best_dist = d;
                    best_idx  = static_cast<int>(si);
                }
            }
            lbl_ptr[col] = best_idx;
        }
    }

    const double approx_eps_px = static_cast<double>(tolerance_mm / pixel_mm);

    for (size_t si = 0; si < stops.size(); ++si) {
        cv::Mat mask(h, w, CV_8UC1, cv::Scalar(0));
        for (int row = 0; row < h; ++row) {
            const auto* lbl_ptr = label_map.ptr<int32_t>(row);
            auto* mask_ptr      = mask.ptr<uint8_t>(row);
            for (int col = 0; col < w; ++col) {
                if (lbl_ptr[col] == static_cast<int>(si)) { mask_ptr[col] = 255; }
            }
        }

        std::vector<std::vector<cv::Point>> cv_contours;
        cv::findContours(mask, cv_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        if (cv_contours.empty()) continue;

        VectorShape solid;
        solid.fill_type     = FillType::Solid;
        solid.fill_color    = stops[si].color;
        solid.opacity       = shape.opacity;
        solid.svg_fill_rule = SvgFillRule::NonZero;

        for (auto& cv_cnt : cv_contours) {
            std::vector<cv::Point> approx;
            cv::approxPolyDP(cv_cnt, approx, approx_eps_px, true);
            if (approx.size() < 3) continue;

            Contour contour;
            contour.reserve(approx.size());
            for (const auto& pt : approx) {
                float mx = ox + (static_cast<float>(pt.x) + 0.5f) * pixel_mm;
                float my = oy + (static_cast<float>(pt.y) + 0.5f) * pixel_mm;
                contour.push_back({mx, my});
            }
            solid.contours.push_back(std::move(contour));
        }

        if (!solid.contours.empty()) { result.push_back(std::move(solid)); }
    }

    return result;
}

/// Replace all gradient shapes in the vector with flattened solid-color sub-shapes.
static void FlattenGradientShapes(std::vector<VectorShape>& shapes, float pixel_mm,
                                  float tolerance_mm) {
    std::vector<VectorShape> new_shapes;
    new_shapes.reserve(shapes.size());

    size_t gradient_count  = 0;
    size_t sub_shape_count = 0;

    for (auto& shape : shapes) {
        if (shape.fill_type == FillType::Solid) {
            new_shapes.push_back(std::move(shape));
            continue;
        }
        ++gradient_count;
        auto subs = FlattenGradientShape(shape, pixel_mm, tolerance_mm);
        sub_shape_count += subs.size();
        for (auto& s : subs) { new_shapes.push_back(std::move(s)); }
    }

    shapes = std::move(new_shapes);

    if (gradient_count > 0) {
        spdlog::info("VectorProc: flattened {} gradient shapes into {} solid sub-shapes",
                     gradient_count, sub_shape_count);
    }
}

static VectorProcResult ProcessParsedSvg(lunasvg::Document& doc, const VectorProcConfig& config,
                                         const std::string& result_name) {
    auto t_total = std::chrono::steady_clock::now();
    auto t0      = t_total;

    g_normalize_stats.Reset();

    doc.updateLayout();

    float svg_w = doc.width();
    float svg_h = doc.height();

    if (svg_w <= 0.0f || svg_h <= 0.0f) { throw InputError("SVG has no visible content"); }

    spdlog::debug("VectorProc: SVG dimensions {:.2f} x {:.2f} px", svg_w, svg_h);

    float scale = 1.0f;
    if (config.target_width_mm > 0.0f || config.target_height_mm > 0.0f) {
        float sw = (config.target_width_mm > 0.0f) ? config.target_width_mm / svg_w
                                                   : std::numeric_limits<float>::max();
        float sh = (config.target_height_mm > 0.0f) ? config.target_height_mm / svg_h
                                                    : std::numeric_limits<float>::max();
        scale    = std::min(sw, sh);
    }

    float tol = config.tessellation_tolerance_mm / std::max(scale, 0.001f);

    float final_w = svg_w * scale;
    float final_h = svg_h * scale;

    spdlog::debug("VectorProc: scale={:.4f}, output {:.2f} x {:.2f} mm, tol={:.4f}", scale, final_w,
                  final_h, tol);

    auto* root = doc.documentElement().svgElement(true);

    std::vector<const lunasvg::SVGGeometryElement*> geo_elements;
    int skipped_text = 0, skipped_image = 0;
    root->transverse([&](lunasvg::SVGElement* elem) {
        if (elem->isGeometryElement()) {
            auto* geo = static_cast<lunasvg::SVGGeometryElement*>(elem);
            if (geo->isRenderable()) { geo_elements.push_back(geo); }
            return;
        }
        auto eid = elem->id();
        if (eid == lunasvg::ElementID::Text || eid == lunasvg::ElementID::Tspan) {
            ++skipped_text;
        } else if (eid == lunasvg::ElementID::Image) {
            ++skipped_image;
        }
    });
    if (skipped_text > 0 || skipped_image > 0) {
        spdlog::warn("VectorProc: skipped {} <text> and {} <image> elements (not supported)",
                     skipped_text, skipped_image);
    }

    spdlog::debug(
        "[perf] VectorProc phase 1 (parse+layout+traverse): {:.1f} ms, {} geometry elements",
        ElapsedMs(t0), geo_elements.size());

    auto t1 = std::chrono::steady_clock::now();

    std::vector<VectorShape> converted(geo_elements.size());
    const bool do_flip               = config.flip_y;
    const std::ptrdiff_t shape_count = static_cast<std::ptrdiff_t>(geo_elements.size());

#if defined(_OPENMP)
#    pragma omp parallel for schedule(dynamic)
#endif
    for (std::ptrdiff_t i = 0; i < shape_count; ++i) {
        const size_t idx = static_cast<size_t>(i);
        VectorShape vs   = ConvertGeometryElement(geo_elements[idx], tol);

        if (scale != 1.0f) {
            for (Contour& c : vs.contours) {
                for (Vec2f& p : c) {
                    p.x *= scale;
                    p.y *= scale;
                }
            }
            if (vs.fill_type != FillType::Solid) {
                auto& g = vs.gradient;
                g.x1 *= scale;
                g.y1 *= scale;
                g.x2 *= scale;
                g.y2 *= scale;
                g.r *= scale;
            }
        }
        if (do_flip) {
            for (Contour& c : vs.contours) {
                for (Vec2f& p : c) { p.y = final_h - p.y; }
            }
            if (vs.fill_type != FillType::Solid) {
                auto& g = vs.gradient;
                g.y1    = final_h - g.y1;
                g.y2    = final_h - g.y2;
            }
        }
        converted[idx] = std::move(vs);
    }

    VectorImage vimg;
    vimg.width_mm         = final_w;
    vimg.height_mm        = final_h;
    size_t total_contours = 0;
    size_t total_vertices = 0;
    for (auto& vs : converted) {
        if (!vs.contours.empty()) {
            for (const auto& c : vs.contours) {
                ++total_contours;
                total_vertices += c.size();
            }
            vimg.shapes.push_back(std::move(vs));
        }
    }

    spdlog::debug("[perf] VectorProc phase 2 (ConvertGeometryElement): {:.1f} ms, {} shapes, "
                  "{} contours, {} vertices",
                  ElapsedMs(t1), vimg.shapes.size(), total_contours, total_vertices);
    g_normalize_stats.Log("phase2 per-shape");
    g_normalize_stats.Reset();

    auto t_grad = std::chrono::steady_clock::now();
    {
        float gpx = config.gradient_pixel_mm;
        if (gpx <= 0.0f) { gpx = config.tessellation_tolerance_mm * 3.0f; }
        FlattenGradientShapes(vimg.shapes, gpx, config.tessellation_tolerance_mm);
    }
    spdlog::debug("[perf] VectorProc phase 2.5 (FlattenGradientShapes): {:.1f} ms",
                  ElapsedMs(t_grad));

    auto t2 = std::chrono::steady_clock::now();

    std::vector<VectorShape> clipped = detail::ClipOcclusion(vimg.shapes);

    spdlog::debug("[perf] VectorProc phase 3 (ClipOcclusion): {:.1f} ms, {} -> {} shapes",
                  ElapsedMs(t2), vimg.shapes.size(), clipped.size());

    auto t3 = std::chrono::steady_clock::now();

    {
        const std::ptrdiff_t clipped_count = static_cast<std::ptrdiff_t>(clipped.size());
#if defined(_OPENMP)
#    pragma omp parallel for schedule(dynamic)
#endif
        for (std::ptrdiff_t ci = 0; ci < clipped_count; ++ci) {
            auto& vs    = clipped[static_cast<size_t>(ci)];
            vs.contours = NormalizeContours(vs.contours, tol, ToClipperFillRule(vs.svg_fill_rule));
        }
    }
    clipped.erase(std::remove_if(clipped.begin(), clipped.end(),
                                 [](const VectorShape& vs) { return vs.contours.empty(); }),
                  clipped.end());

    spdlog::debug("[perf] VectorProc phase 4 (post-occlusion normalize+filter): {:.1f} ms, "
                  "{} final shapes",
                  ElapsedMs(t3), clipped.size());
    g_normalize_stats.Log("phase4 post-occlusion");
    g_normalize_stats.Reset();

    VectorProcResult result;
    result.name      = result_name;
    result.width_mm  = final_w;
    result.height_mm = final_h;
    result.y_flipped = config.flip_y;
    result.shapes    = std::move(clipped);

    spdlog::debug("[perf] VectorProc total: {:.1f} ms, output {} shapes, {:.2f} x {:.2f} mm",
                  ElapsedMs(t_total), result.shapes.size(), result.width_mm, result.height_mm);
    spdlog::info("VectorProc: {} shapes, {:.1f}x{:.1f} mm, {:.0f} ms", result.shapes.size(),
                 result.width_mm, result.height_mm, ElapsedMs(t_total));
    return result;
}

} // namespace

VectorProc::VectorProc(const VectorProcConfig& config) : config_(config) {}

VectorProcResult VectorProc::Run(const std::string& svg_path) const {
    spdlog::info("VectorProc: loading SVG from file: {}", svg_path);

    auto doc = lunasvg::Document::loadFromFile(svg_path);
    if (!doc) { throw IOError("Failed to parse SVG: " + svg_path); }

    return ProcessParsedSvg(*doc, config_, PathStem(svg_path));
}

VectorProcResult VectorProc::RunFromBuffer(const std::vector<uint8_t>& buffer,
                                           const std::string& name) const {
    if (buffer.empty()) { throw InputError("VectorProc::RunFromBuffer: buffer is empty"); }

    auto doc = lunasvg::Document::loadFromData(reinterpret_cast<const char*>(buffer.data()),
                                               buffer.size());
    if (!doc) { throw IOError("VectorProc: failed to parse SVG from buffer"); }

    return ProcessParsedSvg(*doc, config_, name);
}

} // namespace ChromaPrint3D
