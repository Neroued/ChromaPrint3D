#pragma once

// Internal shared utilities for the match module.
// NOT part of the public API.

#include "chromaprint3d/common.h"
#include "chromaprint3d/color.h"
#include "chromaprint3d/error.h"
#include "chromaprint3d/print_profile.h"
#include "detail/json_utils.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cmath>
#include <string>
#include <utility>

namespace ChromaPrint3D {
namespace detail {

constexpr float kTargetColorThicknessMm = 0.4f;
constexpr float kLayerHeight08          = 0.08f;
constexpr float kLayerHeight04          = 0.04f;
constexpr int kColorLayers5             = 5;
constexpr int kColorLayers10            = 10;
constexpr float kFloatEps               = 1e-3f;

inline bool NearlyEqual(float a, float b, float eps = kFloatEps) {
    return std::fabs(a - b) <= eps;
}

inline std::pair<float, int> ModeSpec(PrintMode mode) {
    switch (mode) {
    case PrintMode::Mode0p08x5:
        return {kLayerHeight08, kColorLayers5};
    case PrintMode::Mode0p04x10:
        return {kLayerHeight04, kColorLayers10};
    }
    throw FormatError("Unsupported PrintMode");
}

inline std::string NormalizeLabel(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isalnum(c)) { out.push_back(static_cast<char>(std::tolower(c))); }
    }
    return out;
}

inline std::string NormalizeChannelKeyString(const std::string& value) {
    const std::size_t delim = value.find('|');
    if (delim == std::string::npos) { return NormalizeLabel(value); }
    const std::string color_key    = NormalizeLabel(value.substr(0, delim));
    const std::string material_key = NormalizeLabel(value.substr(delim + 1));
    return color_key + "|" + material_key;
}

inline std::string BuildChannelKey(const Channel& channel) {
    return NormalizeChannelKeyString(channel.color + "|" + channel.material);
}

inline PrintMode ParsePrintModeString(const std::string& value) {
    if (value == "0.08x5" || value == "0p08x5" || value == "Mode0p08x5") {
        return PrintMode::Mode0p08x5;
    }
    if (value == "0.04x10" || value == "0p04x10" || value == "Mode0p04x10") {
        return PrintMode::Mode0p04x10;
    }
    throw FormatError("Unsupported print mode string: " + value);
}

inline LayerOrder ParseLayerOrderValue(const nlohmann::json& value, LayerOrder fallback) {
    return ::ChromaPrint3D::detail::ParseLayerOrder(value, fallback);
}

inline float Dist2(const Lab& a, const Lab& b) {
    const float dl = a.l() - b.l();
    const float da = a.a() - b.a();
    const float db = a.b() - b.b();
    return dl * dl + da * da + db * db;
}

inline float Dist2(const Rgb& a, const Rgb& b) {
    const float dr = a.r() - b.r();
    const float dg = a.g() - b.g();
    const float db = a.b() - b.b();
    return dr * dr + dg * dg + db * db;
}

} // namespace detail
} // namespace ChromaPrint3D
