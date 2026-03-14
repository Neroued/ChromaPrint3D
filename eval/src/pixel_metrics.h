#pragma once

#include <opencv2/core.hpp>

namespace ChromaPrint3D::eval {

struct PixelMetricsResult {
    double psnr                = 0;
    double ssim                = 0;
    double coverage            = 0;
    double overlap             = 0;
    double delta_e_mean        = 0;
    double delta_e_p95         = 0;
    double delta_e_max         = 0;
    double border_delta_e_mean = 0;
};

/// Compute pixel-level fidelity metrics.
/// \param original     Original image (CV_8UC3 BGR).
/// \param rendered     SVG-rasterized image (CV_8UC3 BGR, same size).
/// \param coverage     Coverage mask (CV_8UC1, 255 = covered by any SVG shape).
/// \param shape_count  Per-pixel shape overlap count (CV_16UC1).
/// \param alpha_mask   Optional alpha mask from original image (CV_8UC1, 255 = opaque).
///                     When provided, all metrics are computed only over opaque pixels.
PixelMetricsResult ComputePixelMetrics(const cv::Mat& original, const cv::Mat& rendered,
                                       const cv::Mat& coverage, const cv::Mat& shape_count,
                                       const cv::Mat& alpha_mask = cv::Mat());

} // namespace ChromaPrint3D::eval
