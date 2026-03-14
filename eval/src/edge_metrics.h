#pragma once

#include <opencv2/core.hpp>

namespace ChromaPrint3D::eval {

struct EdgeMetricsResult {
    double edge_f1          = 0;
    double chamfer_distance = 0;
};

/// Compute edge-fidelity metrics between original and rendered images.
/// \param original    Original image (CV_8UC3 BGR).
/// \param rendered    SVG-rasterized image (CV_8UC3 BGR, same size).
/// \param tolerance   Pixel tolerance for edge matching.
/// \param alpha_mask  Optional alpha mask (CV_8UC1, 255 = opaque). Edges in transparent
///                    regions are excluded from comparison.
EdgeMetricsResult ComputeEdgeMetrics(const cv::Mat& original, const cv::Mat& rendered,
                                     int tolerance = 2, const cv::Mat& alpha_mask = cv::Mat());

} // namespace ChromaPrint3D::eval
