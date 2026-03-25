#pragma once

#include "chromaprint3d/vector_image.h"

#include <opencv2/core.hpp>

#include <vector>

namespace ChromaPrint3D::detail {

/// Ray-casting point-in-polygon test against a set of contours.
bool PointInContours(const std::vector<Contour>& contours, float px, float py);

/// Rasterize a gradient-filled shape into a local bitmap.
/// Returns an OpenCV Mat (CV_32FC4, linear RGBA; alpha=1 inside, 0 outside).
/// `out_offset_x` and `out_offset_y` receive the bounding box origin in mm.
cv::Mat RasterizeGradientShape(const VectorShape& shape, float pixel_mm, float& out_offset_x,
                               float& out_offset_y);

} // namespace ChromaPrint3D::detail
