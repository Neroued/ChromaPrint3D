/// \file encoding.h
/// \brief Image encoding and file I/O utilities.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace ChromaPrint3D {

/// Encodes an OpenCV image to PNG format.
/// \param image Input image (BGR or grayscale)
/// \return PNG-encoded image data
std::vector<uint8_t> EncodePng(const cv::Mat& image);

/// Encodes an OpenCV image to JPEG format.
/// \param image Input image (BGR or grayscale)
/// \param quality JPEG quality [1,100] (default: 95)
/// \return JPEG-encoded image data
std::vector<uint8_t> EncodeJpeg(const cv::Mat& image, int quality = 95);

/// Saves an OpenCV image to a file (format determined by extension).
/// \param image Input image to save
/// \param path Output file path (extension determines format: .png, .jpg, etc.)
/// \return True if successful, false otherwise
bool SaveImage(const cv::Mat& image, const std::string& path);

constexpr int kPreviewMinDim = 512;
constexpr int kPreviewMaxDim = 2048;

/// Resize an image so its longest side falls within [\p min_dim, \p max_dim].
/// Downsamples with INTER_AREA; upsamples with INTER_NEAREST to preserve
/// discrete color boundaries in recipe maps.
/// Returns the original image unmodified when already in range.
///
/// \note Upsampling (min_dim > 0) must NOT be used when the output image must
/// stay pixel-aligned with a coordinate map of fixed resolution (e.g. raster
/// recipe editor where the region map matches the recipe map dimensions).
/// The default min_dim is 0 (downscale only) for this reason.
cv::Mat ResizeForPreview(const cv::Mat& image, int min_dim = 0, int max_dim = kPreviewMaxDim);

/// Compute an adaptive pixels-per-mm value so the longest side of a
/// \p width_mm x \p height_mm model hits exactly \p max_dim pixels.
float ComputePreviewPixelsPerMm(float width_mm, float height_mm, int max_dim = kPreviewMaxDim);

} // namespace ChromaPrint3D
