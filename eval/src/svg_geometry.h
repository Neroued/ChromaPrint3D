#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace ChromaPrint3D::eval {

double PolylineSignedArea(const std::vector<cv::Point>& pts);
double PolylinePerimeter(const std::vector<cv::Point>& pts);
bool PointInPolyline(const std::vector<cv::Point>& poly, cv::Point pt);

/// Fill contours from a single SVG shape onto a binary mask, respecting holes
/// detected via geometric containment (even-odd nesting).
/// Contours are sorted by |area| descending; each contour contained inside
/// the nearest larger contour alternates between outer and hole.
cv::Mat FillShapeWithHoles(const std::vector<std::vector<cv::Point>>& contours, int width,
                           int height);

} // namespace ChromaPrint3D::eval
