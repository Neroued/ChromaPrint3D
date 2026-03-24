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
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <limits>
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

static bool IsDegenerateContour(const Contour& contour, float min_area_mm2) {
    if (contour.size() < 3) return true;
    Clipper2Lib::Path64 path = ContourToPath(contour);
    if (path.size() < 3) return true;
    double area_mm2 = std::abs(Clipper2Lib::Area(path)) / (kClipperScale * kClipperScale);
    return area_mm2 < min_area_mm2;
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
    float min_area_mm2  = std::max(1e-5f, tolerance_mm * tolerance_mm * 0.01f);

    Clipper2Lib::Paths64 all_paths;
    all_paths.reserve(contours.size());
    for (const Contour& contour : contours) {
        Clipper2Lib::Path64 path = ContourToPath(contour);
        if (path.size() >= 3) { all_paths.push_back(std::move(path)); }
    }
    if (all_paths.empty()) return normalized;

    Clipper2Lib::Paths64 cleaned = Clipper2Lib::Union(all_paths, fill_rule);
    if (cleaned.empty()) return normalized;
    cleaned = Clipper2Lib::SimplifyPaths(cleaned, simplify_eps, true);

    for (auto& cpath : cleaned) {
        if (cpath.size() < 3) continue;
        Contour out = PathToContour(cpath);
        if (IsDegenerateContour(out, min_area_mm2)) continue;
        normalized.push_back(std::move(out));
    }

    std::sort(normalized.begin(), normalized.end(), [](const Contour& a, const Contour& b) {
        Clipper2Lib::Path64 pa = ContourToPath(a);
        Clipper2Lib::Path64 pb = ContourToPath(b);
        return std::abs(Clipper2Lib::Area(pa)) > std::abs(Clipper2Lib::Area(pb));
    });

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

static VectorProcResult ProcessParsedSvg(lunasvg::Document& doc, const VectorProcConfig& config,
                                         const std::string& result_name) {
    doc.updateLayout();

    float svg_w = doc.width();
    float svg_h = doc.height();

    if (svg_w <= 0.0f || svg_h <= 0.0f) { throw InputError("SVG has no visible content"); }

    spdlog::info("VectorProc: SVG dimensions {:.2f} x {:.2f} px", svg_w, svg_h);

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

    spdlog::info("VectorProc: scale={:.4f}, output {:.2f} x {:.2f} mm", scale, final_w, final_h);

    auto* root = doc.documentElement().svgElement(true);

    std::vector<const lunasvg::SVGGeometryElement*> geo_elements;
    root->transverse([&](lunasvg::SVGElement* elem) {
        if (!elem->isGeometryElement()) return;
        auto* geo = static_cast<lunasvg::SVGGeometryElement*>(elem);
        if (!geo->isRenderable()) return;
        geo_elements.push_back(geo);
    });

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
    vimg.width_mm  = final_w;
    vimg.height_mm = final_h;
    for (auto& vs : converted) {
        if (!vs.contours.empty()) { vimg.shapes.push_back(std::move(vs)); }
    }

    spdlog::info("VectorProc: extracted {} shapes", vimg.shapes.size());

    std::vector<VectorShape> clipped = detail::ClipOcclusion(vimg.shapes);
    for (auto& vs : clipped) {
        vs.contours = NormalizeContours(vs.contours, tol, ToClipperFillRule(vs.svg_fill_rule));
    }
    clipped.erase(std::remove_if(clipped.begin(), clipped.end(),
                                 [](const VectorShape& vs) { return vs.contours.empty(); }),
                  clipped.end());

    VectorProcResult result;
    result.name      = result_name;
    result.width_mm  = final_w;
    result.height_mm = final_h;
    result.y_flipped = config.flip_y;
    result.shapes    = std::move(clipped);
    spdlog::info("VectorProc: output {} clipped shapes, {:.2f} x {:.2f} mm", result.shapes.size(),
                 result.width_mm, result.height_mm);
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
