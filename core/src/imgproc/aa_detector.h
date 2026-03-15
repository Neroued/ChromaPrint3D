#pragma once

/// \file aa_detector.h
/// \brief Anti-aliasing pixel detection and blend-ratio estimation.
///
/// Identifies pixels whose colour is a convex combination of two adjacent
/// region centres, indicating the original image contained sub-pixel
/// anti-aliased edges. The blend ratio alpha can drive more accurate
/// sub-pixel boundary positioning than pure gradient peak finding.

#include <opencv2/core.hpp>

namespace ChromaPrint3D::detail {

struct AADetectConfig {
    float tolerance = 10.0f; ///< Max LAB Delta-E fit residual to accept as AA pixel.
};

struct AAMap {
    cv::Mat is_aa;   ///< CV_8UC1 — non-zero where pixel is detected as AA blend.
    cv::Mat alpha;   ///< CV_32FC1 — blend ratio [0,1] (0 = fully label_a, 1 = fully label_b).
    cv::Mat label_a; ///< CV_32SC1 — first neighbour label for this AA pixel.
    cv::Mat label_b; ///< CV_32SC1 — second neighbour label for this AA pixel.
};

/// Detect anti-aliased (blended) boundary pixels.
///
/// For each pixel on a label boundary, checks whether its LAB colour can be
/// explained as `alpha * center_a + (1-alpha) * center_b` within tolerance.
///
/// \param lab          CV_32FC3 LAB image (unsmoothed preferred).
/// \param labels       CV_32SC1 label map.
/// \param centers_lab  Per-label LAB centres (indexed by label id).
/// \param cfg          Detection parameters.
/// \return             AAMap with per-pixel flags and blend ratios.
AAMap DetectAAPixels(const cv::Mat& lab, const cv::Mat& labels,
                     const std::vector<cv::Vec3f>& centers_lab, const AADetectConfig& cfg = {});

} // namespace ChromaPrint3D::detail
