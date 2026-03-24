#include "chromaprint3d/vector_proc.h"
#include "chromaprint3d/color.h"
#include "chromaprint3d/error.h"
#include "bezier.h"
#include "occlusion.h"

#include <clipper2/clipper.h>
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

namespace {

constexpr double kClipperScale = 100000.0;

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

    double simplify_eps = std::max(100.0, static_cast<double>(tolerance_mm) * kClipperScale * 0.4);
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

static GradientInfo ExtractGradientFromPaintElement(const lunasvg::SVGPaintElement* paintElem,
                                                    const lunasvg::SVGGeometryElement* geoElem,
                                                    FillType& out_fill_type) {
    GradientInfo info;
    out_fill_type = FillType::Solid;
    if (!paintElem) return info;

    auto eid = paintElem->id();
    lunasvg::LengthContext lengthCtx(paintElem);

    if (eid == lunasvg::ElementID::LinearGradient) {
        auto* lg      = static_cast<const lunasvg::SVGLinearGradientElement*>(paintElem);
        out_fill_type = FillType::LinearGradient;
        info.x1       = lengthCtx.valueForLength(lg->x1());
        info.y1       = lengthCtx.valueForLength(lg->y1());
        info.x2       = lengthCtx.valueForLength(lg->x2());
        info.y2       = lengthCtx.valueForLength(lg->y2());
    } else if (eid == lunasvg::ElementID::RadialGradient) {
        auto* rg      = static_cast<const lunasvg::SVGRadialGradientElement*>(paintElem);
        out_fill_type = FillType::RadialGradient;
        info.x1       = lengthCtx.valueForLength(rg->cx());
        info.y1       = lengthCtx.valueForLength(rg->cy());
        info.r        = lengthCtx.valueForLength(rg->r());
        info.x2       = lengthCtx.valueForLength(rg->fx());
        info.y2       = lengthCtx.valueForLength(rg->fy());
    } else {
        return info;
    }

    for (const auto& child : paintElem->children()) {
        auto* childElem = lunasvg::toSVGElement(child);
        if (childElem && childElem->id() == lunasvg::ElementID::Stop) {
            auto* stopElem = static_cast<const lunasvg::SVGStopElement*>(childElem);
            auto gstop     = stopElem->gradientStop(1.0f);
            GradientStop s;
            s.offset = gstop.offset;
            s.color  = Rgb{SrgbToLinear(gstop.color.r), SrgbToLinear(gstop.color.g),
                          SrgbToLinear(gstop.color.b)};
            info.stops.push_back(s);
        }
    }

    return info;
}

static VectorShape ConvertGeometryElement(const lunasvg::SVGGeometryElement* geo, float tolerance) {
    VectorShape vs;

    const auto& paint = geo->fillPaintServer();
    if (!paint.isRenderable()) return vs;

    if (paint.element()) {
        FillType ft  = FillType::Solid;
        vs.gradient  = ExtractGradientFromPaintElement(paint.element(), geo, ft);
        vs.fill_type = ft;
        if (ft == FillType::Solid) return vs;
    } else {
        const auto& color = paint.color();
        if (color.alpha() == 0) return vs;
        vs.fill_type  = FillType::Solid;
        vs.fill_color = LunaSvgColorToRgb(color);
    }

    vs.opacity       = geo->opacity() * paint.opacity();
    vs.svg_fill_rule = (geo->fill_rule() == lunasvg::FillRule::EvenOdd) ? SvgFillRule::EvenOdd
                                                                        : SvgFillRule::NonZero;

    const auto& path = geo->path();
    lunasvg::PathIterator it(path);
    std::array<lunasvg::Point, 3> pts;

    Contour current_contour;

    while (!it.isDone()) {
        auto cmd = it.currentSegment(pts);
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
    root->transverse([&](lunasvg::SVGElement* elem) {
        if (!elem->isGeometryElement()) return;
        auto* geo = static_cast<lunasvg::SVGGeometryElement*>(elem);
        if (!geo->isRenderable()) return;
        geo_elements.push_back(geo);
    });

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
        }
        if (do_flip) {
            for (Contour& c : vs.contours) {
                for (Vec2f& p : c) { p.y = final_h - p.y; }
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
