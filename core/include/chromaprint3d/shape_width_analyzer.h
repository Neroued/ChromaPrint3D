#pragma once

/// \file shape_width_analyzer.h
/// \brief Per-shape minimum width analysis for printability estimation.

#include "color.h"
#include "vector_proc.h"

#include <vector>

namespace ChromaPrint3D {

/// Width statistics for a single shape.
struct ShapeWidthInfo {
    int index = 0;
    Rgb color;
    float area_mm2        = 0.0f;
    float min_width_mm    = 0.0f;
    float median_width_mm = 0.0f;
};

/// Aggregate result of width analysis across all shapes.
struct WidthAnalysisResult {
    float image_width_mm         = 0.0f;
    float image_height_mm        = 0.0f;
    int total_shapes             = 0;
    int filtered_count           = 0;
    float min_area_threshold_mm2 = 0.0f;
    std::vector<ShapeWidthInfo> shapes;
};

/// Analyze the minimum width of each shape in a preprocessed SVG result.
///
/// For each VectorShape whose area >= min_area_mm2, the contours are rasterized
/// at the given resolution, a distance transform + morphological ridge detection
/// is computed, and the P5 percentile / median "local width" along the ridge is
/// reported. Ridge points within 1px of the boundary are excluded to suppress
/// edge artifacts.
///
/// \param vpr      Output of VectorProc (shapes already scaled + clipped).
/// \param min_area_mm2  Shapes with area below this threshold are skipped.
/// \param raster_px_per_mm  Rasterization resolution for width measurement.
/// \return Per-shape width statistics.
WidthAnalysisResult AnalyzeShapeWidths(const VectorProcResult& vpr, float min_area_mm2 = 1.0f,
                                       float raster_px_per_mm = 10.0f);

} // namespace ChromaPrint3D
