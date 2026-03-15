#pragma once

/// \file curve_fitting.h
/// \brief Schneider cubic Bezier fitting and corner detection for boundary edges.

#include "imgproc/bezier.h"
#include "imgproc/boundary_graph.h"

#include <vector>

namespace ChromaPrint3D::detail {

struct CurveFitConfig {
    float error_threshold            = 0.8f;
    int max_recursion_depth          = 8;
    int reparameterize_iterations    = 3;
    float corner_angle_threshold_deg = 135.0f;
    int corner_neighbor_k            = 6;

    bool enable_multiscale_corners = true;  ///< Check corners at multiple k-scales.
    bool enable_curvature_corners  = true;  ///< Detect corners via curvature jump.
    float curvature_jump_threshold = 0.35f; ///< Min curvature-jump ratio to flag a corner.
};

/// Detect corner indices in a polyline based on angle threshold.
/// Returns sorted indices where the polyline should be split for piecewise fitting.
std::vector<int> DetectCorners(const std::vector<Vec2f>& pts, const CurveFitConfig& cfg);

/// Fit a sequence of cubic Bezier curves to an open polyline using the Schneider algorithm.
///
/// Endpoints are fixed (shared junction constraint for watertight topology).
/// Falls back to degenerate linear Bezier if recursion depth is exceeded.
std::vector<CubicBezier> FitBezierToPolyline(const std::vector<Vec2f>& pts,
                                             const CurveFitConfig& cfg = {});

/// Fit cubic Bezier curves to a closed polyline (loop).
///
/// Handles wrap-around corner detection and tangent estimation so the fitted
/// curve chain closes smoothly without a "cutting chord" artifact.
std::vector<CubicBezier> FitBezierToClosedPolyline(const std::vector<Vec2f>& pts,
                                                   const CurveFitConfig& cfg = {});

/// Fit Bezier curves to all edges of a BoundaryGraph in-place.
/// Returns the total number of edges that fell back to polyline.
int FitBezierOnGraph(BoundaryGraph& graph, const CurveFitConfig& cfg = {});

/// Merge consecutive near-linear Bezier segments into single curves.
/// \param segments  Bezier segment list (modified in-place).
/// \param tolerance Max control-point deviation as fraction of chord length to consider linear.
void MergeNearLinearSegments(std::vector<CubicBezier>& segments, float tolerance);

} // namespace ChromaPrint3D::detail
