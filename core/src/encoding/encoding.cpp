#include "chromaprint3d/encoding.h"
#include "chromaprint3d/error.h"

#include <spdlog/spdlog.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace ChromaPrint3D {

constexpr int kPngCompression = 6;

std::vector<uint8_t> EncodePng(const cv::Mat& image) {
    if (image.empty()) { throw InputError("EncodePng: image is empty"); }
    std::vector<uint8_t> buf;
    const std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, kPngCompression};
    if (!cv::imencode(".png", image, buf, params)) {
        throw IOError("EncodePng: cv::imencode failed");
    }
    spdlog::debug("EncodePng: {}x{} -> {} bytes", image.cols, image.rows, buf.size());
    return buf;
}

std::vector<uint8_t> EncodeJpeg(const cv::Mat& image, int quality) {
    if (image.empty()) { throw InputError("EncodeJpeg: image is empty"); }
    if (quality < 0) { quality = 0; }
    if (quality > 100) { quality = 100; }
    std::vector<uint8_t> buf;
    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    if (!cv::imencode(".jpg", image, buf, params)) {
        throw IOError("EncodeJpeg: cv::imencode failed");
    }
    spdlog::debug("EncodeJpeg: {}x{} (q={}) -> {} bytes", image.cols, image.rows, quality,
                  buf.size());
    return buf;
}

bool SaveImage(const cv::Mat& image, const std::string& path) {
    if (image.empty() || path.empty()) { return false; }
    return cv::imwrite(path, image);
}

cv::Mat ResizeForPreview(const cv::Mat& image, int min_dim, int max_dim) {
    if (image.empty() || max_dim <= 0) return image;
    const int max_side = std::max(image.cols, image.rows);
    if (max_side > max_dim) {
        const double scale = static_cast<double>(max_dim) / max_side;
        const int new_w    = std::max(1, static_cast<int>(std::round(image.cols * scale)));
        const int new_h    = std::max(1, static_cast<int>(std::round(image.rows * scale)));
        cv::Mat dst;
        cv::resize(image, dst, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
        spdlog::debug("ResizeForPreview: {}x{} -> {}x{} (downsample)", image.cols, image.rows,
                      new_w, new_h);
        return dst;
    }
    if (min_dim > 0 && max_side < min_dim) {
        const double scale = static_cast<double>(min_dim) / max_side;
        const int new_w    = std::max(1, static_cast<int>(std::round(image.cols * scale)));
        const int new_h    = std::max(1, static_cast<int>(std::round(image.rows * scale)));
        cv::Mat dst;
        cv::resize(image, dst, cv::Size(new_w, new_h), 0, 0, cv::INTER_NEAREST);
        spdlog::debug("ResizeForPreview: {}x{} -> {}x{} (upsample)", image.cols, image.rows, new_w,
                      new_h);
        return dst;
    }
    return image;
}

float ComputePreviewPixelsPerMm(float width_mm, float height_mm, int max_dim) {
    const float long_side = std::max(width_mm, height_mm);
    if (long_side <= 0.0f || max_dim <= 0) return 5.0f;
    return static_cast<float>(max_dim) / long_side;
}

} // namespace ChromaPrint3D
