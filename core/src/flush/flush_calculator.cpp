/// \file flush_calculator.cpp
/// \brief Implementation of color-driven flush-volume calculator.
///
/// HSV/luminance formula derived from BambuStudio (AGPL-3.0,
/// `src/libslic3r/FlushVolCalc.cpp::calc_flush_vol_rgb`). We deliberately
/// drop the dataset lookup and dark→light boost: the goal is reasonable
/// per-color-pair purge volumes, not byte-level parity with Bambu Studio's
/// "Re-calculate" button.
///
/// The formula operates on **gamma-encoded** sRGB (byte / 255), matching
/// BambuStudio (RGB2HSV is given gamma-encoded inputs without decoding).

#include "chromaprint3d/flush_calculator.h"

#include "detail/layer_preview_color.h" // TryParseHexColor

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>

namespace ChromaPrint3D {

namespace {

/// 24-bit gamma-encoded sRGB triplet (parsed from `#RRGGBB`).
struct RgbU8 {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
};

bool ParseHexToRgbU8(std::string_view hex, RgbU8& out) {
    std::uint8_t r = 0, g = 0, b = 0;
    if (!detail::TryParseHexColor(std::string(hex), r, g, b)) return false;
    out = RgbU8{r, g, b};
    return true;
}

// RGB → HSV (BBS-equivalent). Input r,g,b in [0,1] *gamma-encoded*; output
// h ∈ [0,360], s,v ∈ [0,1]. Mirrors BambuStudio
// `slic3r/Utils/ColorSpaceConvert.cpp::RGB2HSV` byte-for-byte.
void RgbToHsv(float r, float g, float b, float& h, float& s, float& v) {
    const float Cmax  = std::max({r, g, b});
    const float Cmin  = std::min({r, g, b});
    const float delta = Cmax - Cmin;

    if (std::abs(delta) < 0.001f) {
        h = 0.0f;
    } else if (Cmax == r) {
        h = 60.0f * std::fmod((g - b) / delta, 6.0f);
    } else if (Cmax == g) {
        h = 60.0f * ((b - r) / delta + 2.0f);
    } else {
        h = 60.0f * ((r - g) / delta + 4.0f);
    }

    s = (std::abs(Cmax) < 0.001f) ? 0.0f : (delta / Cmax);
    v = Cmax;
}

float DegToRad(float deg) {
    constexpr float kPi = std::numbers::pi_v<float>;
    return deg / 180.0f * kPi;
}

// HSV "color distance" used by the BBS fallback formula.
float DeltaHsBbs(float h1, float s1, float v1, float h2, float s2, float v2) {
    const float h1r = DegToRad(h1);
    const float h2r = DegToRad(h2);
    const float dx  = std::cos(h1r) * s1 * v1 - std::cos(h2r) * s2 * v2;
    const float dy  = std::sin(h1r) * s1 * v1 - std::sin(h2r) * s2 * v2;
    return std::min(1.2f, std::sqrt(dx * dx + dy * dy));
}

float CalcTriangle3rdEdge(float a, float b, float deg_ab) {
    return std::sqrt(a * a + b * b - 2.0f * a * b * std::cos(DegToRad(deg_ab)));
}

float Luminance(float r, float g, float b) { return 0.30f * r + 0.59f * g + 0.11f * b; }

// HSV-based formula: BambuStudio `calc_flush_vol_rgb`, gamma-encoded inputs.
// Returns flush volume in mm³ (no min/max clamp here; caller adds the
// constant baseline and clamps).
float CalcFlushVolRgb(const RgbU8& src, const RgbU8& dst) {
    const float src_r_f = src.r / 255.0f;
    const float src_g_f = src.g / 255.0f;
    const float src_b_f = src.b / 255.0f;
    const float dst_r_f = dst.r / 255.0f;
    const float dst_g_f = dst.g / 255.0f;
    const float dst_b_f = dst.b / 255.0f;

    float h_from = 0, s_from = 0, v_from = 0;
    float h_to = 0, s_to = 0, v_to = 0;
    RgbToHsv(src_r_f, src_g_f, src_b_f, h_from, s_from, v_from);
    RgbToHsv(dst_r_f, dst_g_f, dst_b_f, h_to, s_to, v_to);
    float hs_dist = DeltaHsBbs(h_from, s_from, v_from, h_to, s_to, v_to);

    const float from_lumi = Luminance(src_r_f, src_g_f, src_b_f);
    const float to_lumi   = Luminance(dst_r_f, dst_g_f, dst_b_f);
    float lumi_flush      = 0.0f;
    if (to_lumi >= from_lumi) {
        lumi_flush = std::pow(to_lumi - from_lumi, 0.7f) * 560.0f;
    } else {
        lumi_flush              = (from_lumi - to_lumi) * 80.0f;
        const float inter_hsv_v = 0.67f * v_to + 0.33f * v_from;
        hs_dist                 = std::min(inter_hsv_v, hs_dist);
    }
    const float hs_flush = 230.0f * hs_dist;

    float volume = CalcTriangle3rdEdge(hs_flush, lumi_flush, 120.0f);
    return std::max(volume, 60.0f);
}

} // namespace

FlushVolumeCalculator::FlushVolumeCalculator(int min_flush_volume, int max_flush_volume)
    : min_flush_volume_(min_flush_volume), max_flush_volume_(max_flush_volume) {}

int FlushVolumeCalculator::Calc(std::string_view src_hex, std::string_view dst_hex) const {
    RgbU8 src{255, 255, 255};
    RgbU8 dst{255, 255, 255};
    // Unparseable hex (or transparent) is treated as white — matches BBS
    // behaviour and keeps a malformed palette entry from blowing up export.
    ParseHexToRgbU8(src_hex, src);
    ParseHexToRgbU8(dst_hex, dst);

    float volume = CalcFlushVolRgb(src, dst) + static_cast<float>(min_flush_volume_);
    return std::min(static_cast<int>(volume), max_flush_volume_);
}

} // namespace ChromaPrint3D
