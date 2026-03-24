#pragma once

/// \file vector_image.h
/// \brief Data types for parsed vector (SVG) images.

#include "vec2.h"
#include "color.h"

#include <cstdint>
#include <vector>

namespace ChromaPrint3D {

/// A closed polyline representing a contour boundary.
using Contour = std::vector<Vec2f>;

/// Fill type of a vector shape.
enum class FillType : uint8_t {
    Solid,
    LinearGradient,
    RadialGradient,
};

/// SVG fill rule.
enum class SvgFillRule : uint8_t {
    NonZero = 0,
    EvenOdd = 1,
};

/// Gradient color stop.
struct GradientStop {
    float offset = 0.0f;
    Rgb color;
};

/// Gradient definition extracted from SVG.
struct GradientInfo {
    std::vector<GradientStop> stops;
    float x1 = 0.0f, y1 = 0.0f; ///< Start point (linear) or center (radial).
    float x2 = 0.0f, y2 = 0.0f; ///< End point (linear).
    float r = 0.0f;             ///< Radius (radial only).
};

/// A single shape from a vector image, with contours and fill information.
struct VectorShape {
    std::vector<Contour> contours; ///< contours[0] = outer, rest = holes.

    FillType fill_type        = FillType::Solid;
    SvgFillRule svg_fill_rule = SvgFillRule::NonZero;
    Rgb fill_color;        ///< Valid when fill_type == Solid.
    GradientInfo gradient; ///< Valid when fill_type != Solid.
    float opacity = 1.0f;
};

/// A parsed vector image: a list of shapes in draw order.
struct VectorImage {
    float width_mm  = 0.0f;
    float height_mm = 0.0f;
    std::vector<VectorShape> shapes; ///< Draw order: higher index = on top.
};

} // namespace ChromaPrint3D
