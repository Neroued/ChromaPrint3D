#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace ChromaPrint3D::eval {

struct RasterizedSvg {
    cv::Mat bgr;         // CV_8UC3 — rendered image
    cv::Mat coverage;    // CV_8UC1 — 255 where any shape covers, 0 elsewhere
    cv::Mat shape_count; // CV_16UC1 — per-pixel count of overlapping shapes
};

/// Rasterize an SVG string to BGR + coverage + overlap count maps.
/// The output size matches \p width x \p height (typically the original image dimensions).
RasterizedSvg RasterizeSvg(const std::string& svg_content, int width, int height);

} // namespace ChromaPrint3D::eval
