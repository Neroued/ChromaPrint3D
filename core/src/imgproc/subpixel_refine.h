#pragma once

/// \file subpixel_refine.h
/// \brief Gradient-guided sub-pixel boundary refinement for BoundaryGraph edges.
///
/// After BuildBoundaryGraph produces crack-grid chains at integer coordinates,
/// this pass shifts each interior point along its local normal to the position
/// of maximum colour gradient in the original (unsmoothed) LAB image, recovering
/// sub-pixel edge locations and eliminating the staircase artefact.

#include "imgproc/boundary_graph.h"

#include <opencv2/core.hpp>

namespace ChromaPrint3D::detail {

struct SubpixelRefineConfig {
    float max_displacement = 0.7f; ///< Max shift along the normal (pixels).
    int tangent_window     = 5;    ///< Neighbour radius for tangent estimation (points).
    int num_samples        = 9;    ///< Sampling points along the normal.
    float sample_range     = 1.0f; ///< Sampling half-range along the normal (pixels).
    float min_gradient     = 3.0f; ///< Minimum gradient peak to apply a shift (LAB deltaE/px).
};

/// Refine BoundaryGraph edge points to sub-pixel positions using colour gradients.
///
/// Junction nodes (first/last point of each edge) are kept fixed to preserve
/// the multi-label topology.  The \p lab image should be the **original
/// unsmoothed** LAB image (CV_32FC3) so that gradients are sharp.
void RefineEdgesSubpixel(BoundaryGraph& graph, const cv::Mat& lab,
                         const SubpixelRefineConfig& cfg = {});

/// AA-enhanced variant: where the AAMap indicates a blended pixel at the
/// boundary point, use the blend ratio alpha to position the edge at
/// `offset = alpha - 0.5` along the normal instead of gradient-peak finding.
void RefineEdgesSubpixelAA(BoundaryGraph& graph, const cv::Mat& lab, const cv::Mat& aa_is_aa,
                           const cv::Mat& aa_alpha, const SubpixelRefineConfig& cfg = {});

} // namespace ChromaPrint3D::detail
