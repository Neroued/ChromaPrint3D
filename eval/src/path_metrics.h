#pragma once

#include <string>

namespace ChromaPrint3D::eval {

struct PathMetricsResult {
    int total_shapes           = 0;
    int unique_colors          = 0;
    double mergeable_ratio     = 1.0;
    double tiny_fragment_rate  = 0;
    double gini_coefficient    = 0;
    int path_complexity_median = 0;
    int path_complexity_p95    = 0;
    double circularity_p95     = 0;
    int sliver_count           = 0;
    int island_count           = 0;
    int same_color_gap_pixels  = 0;
    double color_compression   = 0;
};

/// Compute path-structure metrics from an SVG string.
/// \param svg_content   Complete SVG document string.
/// \param width         Image width in pixels (for raster-based sub-metrics).
/// \param height        Image height in pixels.
/// \param tiny_area_threshold  Shapes with |net_area| below this are "tiny" (px²).
/// \param sliver_threshold     Shapes with circularity below this are "slivers".
PathMetricsResult ComputePathMetrics(const std::string& svg_content, int width, int height,
                                     double tiny_area_threshold = 50.0,
                                     double sliver_threshold    = 0.02);

} // namespace ChromaPrint3D::eval
