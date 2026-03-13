#pragma once

/// \file thin_line_vectorizer.h
/// \brief Skeleton-based thin-line vectorization via Zhang-Suen thinning.

#include "imgproc/bezier.h"
#include "imgproc/morphology_utils.h"
#include "imgproc/svg_writer.h"

#include <opencv2/core.hpp>

#include <vector>

namespace ChromaPrint3D::detail {

/// Detect thin sub-regions within a binary mask using distance transform.
/// Returns a mask of pixels where max inscribed radius <= threshold.
cv::Mat DetectThinRegion(const cv::Mat& mask, float max_radius = 2.5f);

/// Extract stroke paths from a skeleton mask, fit Bezier curves, and estimate width.
/// Returns VectorizedShapes with is_stroke=true and appropriate stroke_width.
std::vector<VectorizedShape> ExtractStrokePaths(const cv::Mat& skeleton, const cv::Mat& dist_map,
                                                const Rgb& color, float min_length = 3.0f);

} // namespace ChromaPrint3D::detail
