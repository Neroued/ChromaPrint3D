/// \file core/src/color/image.cpp
/// \brief Self-implemented `BgrToLab` and related image-level color entries.
///
/// Channel-order contract (CRITICAL):
///   - OpenCV inputs are BGR-ordered.
///   - Project sRGB→XYZ matrix is RGB-column-ordered.
///   - We pre-compute `kBgrToXyzD65Matrix` with columns `(B, G, R)` so a
///     single `cv::transform` does both the channel reorder and the
///     XYZ projection. **Never** apply the standard RGB→XYZ matrix to
///     BGR-ordered pixel data — `B` would be projected through the `R`
///     row and vice versa, producing severe errors.
///
/// Lab math: project D65 (`(6/29)^3` threshold, true cbrt). The 4096-entry
/// `LabF` LUT accelerates the cube-root step; precision verified against
/// the scalar reference path in `core/tests/test_color_image.cpp`.

#include "chromaprint3d/color/image.h"
#include "chromaprint3d/color/conversions.h"
#include "chromaprint3d/color/types.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace ChromaPrint3D {

namespace {

// ── 256-entry sRGB-byte → linear-float LUT (cv::Mat-shaped for cv::LUT) ─────

const cv::Mat& SrgbU8ToLinearLutMat() {
    static const cv::Mat kLut = []() {
        cv::Mat m(1, 256, CV_32FC1);
        for (int i = 0; i < 256; ++i) {
            m.at<float>(0, i) = SrgbToLinearByte(static_cast<std::uint8_t>(i));
        }
        return m;
    }();
    return kLut;
}

// ── BGR-column-order RGB→XYZ matrix (D65, sRGB primaries) ──────────────────
//
// Standard RGB→XYZ rows × BGR-permuted columns:
//   col 0 = B, col 1 = G, col 2 = R
// So `[X, Y, Z]^T = M_bgr · [B_lin, G_lin, R_lin]^T` produces XYZ from
// linear-BGR pixel data in one matrix multiply.
const cv::Matx33f& BgrToXyzD65Matrix() {
    // Original RGB-column matrix:
    //   X = 0.4124564 R + 0.3575761 G + 0.1804375 B
    //   Y = 0.2126729 R + 0.7151522 G + 0.0721750 B
    //   Z = 0.0193339 R + 0.1191920 G + 0.9503041 B
    // BGR-column form swaps columns 0 and 2:
    static const cv::Matx33f kM(
        0.1804375f, 0.3575761f, 0.4124564f,  // X = .B + .G + .R
        0.0721750f, 0.7151522f, 0.2126729f,  // Y = .B + .G + .R
        0.9503041f, 0.1191920f, 0.0193339f); // Z = .B + .G + .R
    return kM;
}

// XYZ → linear RGB matrix (RGB column-order, used for LabToBgr / LinearRgbToBgr).
const cv::Matx33f& XyzToRgbD65Matrix() {
    static const cv::Matx33f kM(
        3.2404542f, -1.5371385f, -0.4985314f,
        -0.9692660f, 1.8760108f, 0.0415560f,
        0.0556434f, -0.2040259f, 1.0572252f);
    return kM;
}

// ── 4096-entry LabF LUT ─────────────────────────────────────────────────────
//
// Input domain: `xyz_n = xyz / D65_white_point`. For sRGB primaries this
// stays inside [0, 1.5]. We use a uniform 4096-entry LUT with linear
// interpolation; per-step error in the cbrt domain is ~3.66e-4, contributing
// at most ~0.005 to L*, well below the test_color_image.cpp threshold
// (max ΔE76 < 0.05).

constexpr int kLabFLutEntries = 4096;
constexpr float kLabFLutMax   = 1.5f;
constexpr float kLabFLutScale = static_cast<float>(kLabFLutEntries - 1) / kLabFLutMax;

const std::array<float, kLabFLutEntries>& LabFLut() {
    static const std::array<float, kLabFLutEntries> kLut = []() {
        std::array<float, kLabFLutEntries> arr{};
        for (int i = 0; i < kLabFLutEntries; ++i) {
            const float t = (static_cast<float>(i) / kLabFLutScale);
            arr[i]        = LabF(t);
        }
        return arr;
    }();
    return kLut;
}

inline float LabFFast(float t) {
    if (!(t >= 0.0f)) return 4.0f / 29.0f; // NaN / negative — sentinel
    if (t >= kLabFLutMax) return std::cbrt(t);
    const float fidx = t * kLabFLutScale;
    const int i      = static_cast<int>(fidx);
    const float frac = fidx - static_cast<float>(i);
    const auto& lut  = LabFLut();
    const float a    = lut[i];
    const float b    = (i + 1 < kLabFLutEntries) ? lut[i + 1] : a;
    return a + (b - a) * frac;
}

constexpr float kXn = 0.95047f;
constexpr float kYn = 1.00000f;
constexpr float kZn = 1.08883f;

} // namespace

// ── BgrToLab (authoritative project-D65 path, no cv::cvtColor(BGR2Lab)) ─────

cv::Mat BgrToLab(const cv::Mat& bgr_u8) {
    if (bgr_u8.empty()) return cv::Mat();
    CV_Assert(bgr_u8.type() == CV_8UC3);

    // Step 1: BGR uint8 → BGR linear float via per-channel LUT.
    cv::Mat linear_bgr_f;
    cv::LUT(bgr_u8, SrgbU8ToLinearLutMat(), linear_bgr_f);
    // After cv::LUT on a CV_8UC3 with a CV_32FC1 LUT, OpenCV produces CV_32FC3.
    CV_Assert(linear_bgr_f.type() == CV_32FC3);

    // Step 2: BGR-ordered matrix multiply → XYZ (channel reorder is folded in).
    cv::Mat xyz_f;
    cv::transform(linear_bgr_f, xyz_f, BgrToXyzD65Matrix());

    // Step 3: parallel_for_ row-major LabF + affine.
    cv::Mat lab_f(xyz_f.size(), CV_32FC3);
    const int rows = xyz_f.rows;
    const int cols = xyz_f.cols;

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int r = range.start; r < range.end; ++r) {
            const cv::Vec3f* xyz_row = xyz_f.ptr<cv::Vec3f>(r);
            cv::Vec3f* lab_row       = lab_f.ptr<cv::Vec3f>(r);
            for (int c = 0; c < cols; ++c) {
                const float fx = LabFFast(xyz_row[c][0] / kXn);
                const float fy = LabFFast(xyz_row[c][1] / kYn);
                const float fz = LabFFast(xyz_row[c][2] / kZn);
                lab_row[c][0]  = 116.0f * fy - 16.0f;
                lab_row[c][1]  = 500.0f * (fx - fy);
                lab_row[c][2]  = 200.0f * (fy - fz);
            }
        }
    });

    return lab_f;
}

// ── BgrToLinearRgb (replaces raster_proc.cpp::BgrToRgbLinear) ───────────────

cv::Mat BgrToLinearRgb(const cv::Mat& bgr_u8) {
    if (bgr_u8.empty()) return cv::Mat();
    CV_Assert(bgr_u8.type() == CV_8UC3);

    // BGR uint8 → BGR linear float via LUT.
    cv::Mat linear_bgr_f;
    cv::LUT(bgr_u8, SrgbU8ToLinearLutMat(), linear_bgr_f);
    CV_Assert(linear_bgr_f.type() == CV_32FC3);

    // BGR → RGB channel swap.
    cv::Mat linear_rgb_f;
    cv::cvtColor(linear_bgr_f, linear_rgb_f, cv::COLOR_BGR2RGB);
    return linear_rgb_f;
}

// ── LinearRgbToBgr (preview helper, RGB linear float → BGR uint8) ───────────

cv::Mat LinearRgbToBgr(const cv::Mat& linear_rgb_f) {
    if (linear_rgb_f.empty()) return cv::Mat();
    CV_Assert(linear_rgb_f.type() == CV_32FC3);

    const int rows = linear_rgb_f.rows;
    const int cols = linear_rgb_f.cols;
    cv::Mat bgr_u8(rows, cols, CV_8UC3);

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int r = range.start; r < range.end; ++r) {
            const cv::Vec3f* src = linear_rgb_f.ptr<cv::Vec3f>(r);
            cv::Vec3b* dst       = bgr_u8.ptr<cv::Vec3b>(r);
            for (int c = 0; c < cols; ++c) {
                // RGB linear float → sRGB byte.
                const std::uint8_t r8 = LinearToSrgbByte(src[c][0]);
                const std::uint8_t g8 = LinearToSrgbByte(src[c][1]);
                const std::uint8_t b8 = LinearToSrgbByte(src[c][2]);
                dst[c]                = cv::Vec3b(b8, g8, r8); // BGR order
            }
        }
    });
    return bgr_u8;
}

// ── LabToBgr (round-trip diagnostic) ────────────────────────────────────────

cv::Mat LabToBgr(const cv::Mat& lab_f) {
    if (lab_f.empty()) return cv::Mat();
    CV_Assert(lab_f.type() == CV_32FC3);

    const int rows = lab_f.rows;
    const int cols = lab_f.cols;
    cv::Mat bgr_u8(rows, cols, CV_8UC3);

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int r = range.start; r < range.end; ++r) {
            const cv::Vec3f* src = lab_f.ptr<cv::Vec3f>(r);
            cv::Vec3b* dst       = bgr_u8.ptr<cv::Vec3b>(r);
            for (int c = 0; c < cols; ++c) {
                const Lab lab(src[c][0], src[c][1], src[c][2]);
                const Rgb rgb     = lab.ToRgb();
                const Rgb clamped = Rgb::Clamp01(rgb);
                dst[c]            = cv::Vec3b(LinearToSrgbByte(clamped.b()),
                                              LinearToSrgbByte(clamped.g()),
                                              LinearToSrgbByte(clamped.r()));
            }
        }
    });
    return bgr_u8;
}

} // namespace ChromaPrint3D
