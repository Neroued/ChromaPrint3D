#pragma once

#include "application/service_result.h"

#include <nlohmann/json.hpp>

#include "chromaprint3d/common.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace chromaprint3d::backend {

std::string ToLowerAscii(std::string value);
std::string StripExtension(const std::string& filename);
ChromaPrint3D::ColorSpace ParseColorSpace(const std::string& value);
ChromaPrint3D::ClusterMethod ParseClusterMethod(const std::string& value);
ServiceResult ParseJsonObject(const std::optional<std::string>& raw, nlohmann::json& out);
ServiceResult ValidateDecodedImage(const std::vector<uint8_t>& image, std::int64_t max_pixels);

} // namespace chromaprint3d::backend
