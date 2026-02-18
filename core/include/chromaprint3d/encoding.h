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

} // namespace ChromaPrint3D
