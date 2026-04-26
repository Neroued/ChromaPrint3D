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
/// All color types come from the unified `chromaprint3d/color/*` module:
/// `SrgbU8` carries the gamma-encoded byte triple, `Hsv::FromSrgbU8` runs
/// the BBS-parity HSV conversion, and `HsvDistanceBbs` gives the polar
/// HSV distance used by the flush formula.

#include "chromaprint3d/flush_calculator.h"

#include "chromaprint3d/color.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>

namespace ChromaPrint3D {

namespace {

float DegToRad(float deg) {
    constexpr float kPi = std::numbers::pi_v<float>;
    return deg / 180.0f * kPi;
}

float CalcTriangle3rdEdge(float a, float b, float deg_ab) {
    return std::sqrt(a * a + b * b - 2.0f * a * b * std::cos(DegToRad(deg_ab)));
}

float Luminance(const SrgbU8& c) {
    return 0.30f * (c.r / 255.0f) + 0.59f * (c.g / 255.0f) + 0.11f * (c.b / 255.0f);
}

// HSV-based formula: BambuStudio `calc_flush_vol_rgb`, gamma-encoded inputs.
// Returns flush volume in mm³ (no min/max clamp here; caller adds the
// constant baseline and clamps).
float CalcFlushVolRgb(const SrgbU8& src, const SrgbU8& dst) {
    Hsv hsv_from = Hsv::FromSrgbU8(src);
    Hsv hsv_to   = Hsv::FromSrgbU8(dst);

    float hs_dist = HsvDistanceBbs(hsv_from, hsv_to);

    const float from_lumi = Luminance(src);
    const float to_lumi   = Luminance(dst);
    float lumi_flush      = 0.0f;
    if (to_lumi >= from_lumi) {
        lumi_flush = std::pow(to_lumi - from_lumi, 0.7f) * 560.0f;
    } else {
        lumi_flush              = (from_lumi - to_lumi) * 80.0f;
        const float inter_hsv_v = 0.67f * hsv_to.v() + 0.33f * hsv_from.v();
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
    // Unparseable hex (or transparent) is treated as white — matches BBS
    // behaviour and keeps a malformed palette entry from blowing up export.
    SrgbU8 src{255, 255, 255};
    SrgbU8 dst{255, 255, 255};
    if (auto parsed = SrgbU8::FromHex(src_hex); parsed) src = *parsed;
    if (auto parsed = SrgbU8::FromHex(dst_hex); parsed) dst = *parsed;

    float volume = CalcFlushVolRgb(src, dst) + static_cast<float>(min_flush_volume_);
    return std::min(static_cast<int>(volume), max_flush_volume_);
}

} // namespace ChromaPrint3D
