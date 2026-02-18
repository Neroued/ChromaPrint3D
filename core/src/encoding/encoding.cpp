#include "chromaprint3d/encoding.h"
#include "chromaprint3d/error.h"

#include <spdlog/spdlog.h>

#include <opencv2/imgcodecs.hpp>

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

} // namespace ChromaPrint3D
