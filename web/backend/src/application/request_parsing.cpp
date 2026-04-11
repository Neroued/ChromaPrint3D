#include "application/request_parsing.h"

#include "chromaprint3d/image_io.h"
#include "chromaprint3d/pipeline.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace chromaprint3d::backend {

using json = nlohmann::json;
using namespace ChromaPrint3D;

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string StripExtension(const std::string& filename) {
    auto dot = filename.rfind('.');
    if (dot == std::string::npos) return filename;
    return filename.substr(0, dot);
}

ColorSpace ParseColorSpace(const std::string& value) {
    const std::string lowered = ToLowerAscii(value);
    if (lowered == "lab") return ColorSpace::Lab;
    if (lowered == "rgb") return ColorSpace::Rgb;
    throw std::runtime_error("Invalid color_space: " + value);
}

ClusterMethod ParseClusterMethod(const std::string& value) {
    const std::string lowered = ToLowerAscii(value);
    if (lowered == "slic") return ClusterMethod::Slic;
    if (lowered == "kmeans" || lowered == "k_means" || lowered == "k-means") {
        return ClusterMethod::KMeans;
    }
    throw std::runtime_error("Invalid cluster_method: " + value);
}

ServiceResult ParseJsonObject(const std::optional<std::string>& raw, json& out) {
    out = json::object();
    if (!raw || raw->empty()) return ServiceResult::Success(200, json::object());
    try {
        out = json::parse(*raw);
        if (!out.is_object()) {
            return ServiceResult::Error(400, "invalid_json", "Expected JSON object");
        }
        return ServiceResult::Success(200, json::object());
    } catch (const std::exception& e) {
        return ServiceResult::Error(400, "invalid_json", std::string("Invalid JSON: ") + e.what());
    }
}

ServiceResult ValidateDecodedImage(const std::vector<uint8_t>& image, std::int64_t max_pixels) {
    cv::Mat decoded;
    try {
        decoded = DecodeImageWithIcc(image);
    } catch (...) { return ServiceResult::Error(400, "invalid_image", "Failed to decode image"); }
    if (decoded.empty())
        return ServiceResult::Error(400, "invalid_image", "Failed to decode image");
    std::int64_t pixels = static_cast<std::int64_t>(decoded.cols) * decoded.rows;
    if (pixels > max_pixels) {
        return ServiceResult::Error(413, "image_too_large",
                                    "Decoded image exceeds max pixels limit");
    }
    return ServiceResult::Success(200, json::object());
}

} // namespace chromaprint3d::backend
