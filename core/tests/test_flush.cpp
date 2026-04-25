/// \file test_flush.cpp
/// \brief Tests for FlushVolumeCalculator (HSV-only formula).
///
/// Ground-truth values were produced by `scripts/dev/flush_reference.py`
/// (independent Python translation of the BambuStudio HSV formula). To
/// recompute, run that script and compare numerically — we tolerate ±1 mm³
/// to absorb the C++ float vs Python double last-bit difference.

#include <gtest/gtest.h>

#include "chromaprint3d/flush_calculator.h"

using ChromaPrint3D::FlushVolumeCalculator;
using ChromaPrint3D::kMaxFlushVolume;

// ── HSV formula known values ────────────────────────────────────────────────
//
// Reference values from `scripts/dev/flush_reference.py` (Python translation
// of `calc_flush_vol_rgb` from BambuStudio). Tolerance ±1 mm³ accounts for
// float-vs-double last-bit differences between C++ and Python.

TEST(FlushCalculator, BlackToWhite) {
    // Greyscale switch: hs_dist = 0; luminance term dominates.
    // `pow(1.0 - 0.0, 0.7) * 560 = 560` → triangle_3rd_edge(0, 560, 120°) = 560
    FlushVolumeCalculator calc(0, kMaxFlushVolume);
    EXPECT_NEAR(calc.Calc("#000000", "#FFFFFF"), 560, 1);
}

TEST(FlushCalculator, WhiteToBlack) {
    // Luminance branch flips: `(1-0) * 80 = 80`, hs_dist = 0.
    FlushVolumeCalculator calc(0, kMaxFlushVolume);
    EXPECT_NEAR(calc.Calc("#FFFFFF", "#000000"), 80, 1);
}

TEST(FlushCalculator, SameColorMinFloor) {
    // Same color → hs_dist = 0, lumi_flush = 0, max(0, 60) = 60.
    FlushVolumeCalculator calc(0, kMaxFlushVolume);
    EXPECT_EQ(calc.Calc("#FFFFFF", "#FFFFFF"), 60);
    EXPECT_EQ(calc.Calc("#000000", "#000000"), 60);
    EXPECT_EQ(calc.Calc("#C12E1F", "#C12E1F"), 60);
}

TEST(FlushCalculator, KnownColorPairs) {
    // From `scripts/dev/flush_reference.py` (HSV formula, integer mm³):
    //   #C12E1F (BBL red) → #0086D6 (BBL cyan): 319
    //   #F4EE2A (BBL yellow) → #000000 (black): 122
    //   #2850E0 (custom blue) → #46A8F9 (custom light blue): 245
    //   #0086D6 (cyan) → #C12E1F (red): 182  (asymmetric of red→cyan)
    FlushVolumeCalculator calc(0, kMaxFlushVolume);
    EXPECT_NEAR(calc.Calc("#C12E1F", "#0086D6"), 319, 1);
    EXPECT_NEAR(calc.Calc("#F4EE2A", "#000000"), 122, 1);
    EXPECT_NEAR(calc.Calc("#2850E0", "#46A8F9"), 245, 1);
    EXPECT_NEAR(calc.Calc("#0086D6", "#C12E1F"), 182, 1);
}

TEST(FlushCalculator, MinFlushVolumeAddsToFallback) {
    FlushVolumeCalculator calc_zero(0, kMaxFlushVolume);
    FlushVolumeCalculator calc_min100(100, kMaxFlushVolume);
    // Off-diagonal: `result_with_min = result_zero + 100` (exactly).
    int v0 = calc_zero.Calc("#2850E0", "#46A8F9");
    int v1 = calc_min100.Calc("#2850E0", "#46A8F9");
    EXPECT_EQ(v1 - v0, 100);
}

TEST(FlushCalculator, RespectsMaxClamp) {
    FlushVolumeCalculator calc(/*min*/ 5000, /*max*/ 200);
    // Any pair would otherwise exceed 5000; clamp to 200.
    EXPECT_EQ(calc.Calc("#000000", "#FFFFFF"), 200);
}

TEST(FlushCalculator, BadHexTreatedAsWhite) {
    // Unparseable hex defaults to white; (white, white) → 60 mm³.
    FlushVolumeCalculator calc(0, kMaxFlushVolume);
    EXPECT_EQ(calc.Calc("not-a-hex", "also-bad"), 60);
}
