#pragma once

/// \file contour_assembly.h
/// \brief Assemble closed contours per label from a BoundaryGraph, with hole hierarchy.

#include "imgproc/bezier.h"
#include "imgproc/boundary_graph.h"
#include "imgproc/curve_fitting.h"
#include "imgproc/svg_writer.h"

#include <vector>

namespace ChromaPrint3D::detail {

/// Parameters controlling contour smoothing during assembly.
struct ContourSmoothConfig {
    float decimate_epsilon        = 0.15f; ///< Near-collinear point removal threshold.
    float smooth_max_displacement = 0.5f;  ///< Max displacement per smoothing iteration.
    int smooth_iterations         = 2;     ///< Number of smoothing passes.
};

/// Derive ContourSmoothConfig from a normalised smoothness value in [0,1].
ContourSmoothConfig ContourSmoothFromLevel(float smoothness);

/// Assemble VectorizedShapes from a BoundaryGraph using polyline (degenerate Bezier) segments.
///
/// For each label, collects all boundary edges, chains them into closed contours,
/// determines outer/hole hierarchy by signed area, and packages them as VectorizedShape.
///
/// \param graph         The shared boundary graph.
/// \param num_labels    Total number of labels (0-based).
/// \param palette       Color palette indexed by label.
/// \param min_contour_area  Minimum absolute area to keep a contour.
/// \param min_hole_area     Minimum absolute area to keep a hole.
/// \param fit_cfg       Optional Bezier curve fitting configuration.
/// \param smooth_cfg    Contour smoothing configuration.
/// \return  One VectorizedShape per label that has valid contours.
std::vector<VectorizedShape> AssembleContoursFromGraph(const BoundaryGraph& graph, int num_labels,
                                                       const std::vector<Rgb>& palette,
                                                       float min_contour_area, float min_hole_area,
                                                       const CurveFitConfig* fit_cfg = nullptr,
                                                       const ContourSmoothConfig& smooth_cfg = {});

} // namespace ChromaPrint3D::detail
