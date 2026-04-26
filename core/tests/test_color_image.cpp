/// \file test_color_image.cpp
/// \brief Tests for the self-implemented project-D65 image-level entries —
///        `BgrToLab`, `BgrToLinearRgb`, `LinearRgbToBgr`, `LabToBgr`.
///
/// Correctness oracle is the **scalar reference path** — `Rgb::ToLab` after
/// `SrgbU8::ToRgb` — NOT OpenCV `cvtColor(BGR2Lab)` (which uses a
/// 33×33×33 LUT and produces sub-ΔE drift; project D65 is the new
/// authoritative path).

#include <gtest/gtest.h>

#include "chromaprint3d/color.h"
#include "chromaprint3d/color/image.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace ChromaPrint3D;

// ── BgrToLab: precision against scalar reference path ──────────────────────

TEST(ColorImage, BgrToLabMatchesScalarReference) {
    // 10000 random BGR_U8 pixels — verify against scalar path
    //     Rgb::ToLab(SrgbU8::ToRgb(SrgbU8{r, g, b}))
    // max ΔE76 < 0.05 (per plan §3.7.1).
    std::mt19937 rng(0xC010ABE1);
    std::uniform_int_distribution<int> byte(0, 255);

    constexpr int kN = 10000;
    cv::Mat bgr_u8(1, kN, CV_8UC3);
    std::vector<Lab> reference;
    reference.reserve(kN);
    for (int i = 0; i < kN; ++i) {
        const std::uint8_t b = static_cast<std::uint8_t>(byte(rng));
        const std::uint8_t g = static_cast<std::uint8_t>(byte(rng));
        const std::uint8_t r = static_cast<std::uint8_t>(byte(rng));
        bgr_u8.at<cv::Vec3b>(0, i) = cv::Vec3b(b, g, r);
        // Note: SrgbU8 is RGB-ordered.
        reference.push_back(SrgbU8{r, g, b}.ToRgb().ToLab());
    }

    cv::Mat lab = BgrToLab(bgr_u8);
    ASSERT_EQ(lab.type(), CV_32FC3);
    ASSERT_EQ(lab.rows, 1);
    ASSERT_EQ(lab.cols, kN);

    float max_de = 0.0f;
    for (int i = 0; i < kN; ++i) {
        const cv::Vec3f& v = lab.at<cv::Vec3f>(0, i);
        Lab actual(v[0], v[1], v[2]);
        const float de = Lab::DeltaE76(actual, reference[i]);
        max_de         = std::max(max_de, de);
    }
    EXPECT_LT(max_de, 0.05f) << "Max ΔE76 between BgrToLab and scalar reference exceeded";
}

TEST(ColorImage, BgrToLabEmptyInput) {
    cv::Mat empty;
    cv::Mat result = BgrToLab(empty);
    EXPECT_TRUE(result.empty());
}

TEST(ColorImage, BgrToLabBlackAndWhite) {
    cv::Mat bgr_u8(1, 2, CV_8UC3);
    bgr_u8.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 0);       // black
    bgr_u8.at<cv::Vec3b>(0, 1) = cv::Vec3b(255, 255, 255); // white

    cv::Mat lab = BgrToLab(bgr_u8);
    EXPECT_NEAR(lab.at<cv::Vec3f>(0, 0)[0], 0.0f, 0.1f);
    EXPECT_NEAR(lab.at<cv::Vec3f>(0, 1)[0], 100.0f, 0.5f);
}

// ── Channel order safety: BgrToLab must NOT confuse B and R ────────────────

TEST(ColorImage, BgrToLabChannelOrderSafety) {
    // Pure red BGR: Vec3b(0, 0, 255). After project D65 conversion, a* must
    // be strongly positive (red has +a*). If channels were swapped (B used
    // as R), the result would shift to blue and a* would be near zero.
    cv::Mat bgr_u8(1, 1, CV_8UC3);
    bgr_u8.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 255); // pure red in BGR order

    cv::Mat lab        = BgrToLab(bgr_u8);
    const cv::Vec3f& v = lab.at<cv::Vec3f>(0, 0);
    EXPECT_GT(v[1], 60.0f) << "Pure-red a* should be strongly positive";
    EXPECT_LT(v[2], 70.0f) << "Pure-red b* < 70 expected";
    EXPECT_GT(v[2], 30.0f) << "Pure-red b* > 30 expected (yellow-ish)";
}

// ── BgrToLinearRgb / LinearRgbToBgr round-trip ──────────────────────────────

TEST(ColorImage, BgrToLinearRgbRoundTrip) {
    cv::Mat bgr_u8(1, 5, CV_8UC3);
    bgr_u8.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 0);
    bgr_u8.at<cv::Vec3b>(0, 1) = cv::Vec3b(255, 255, 255);
    bgr_u8.at<cv::Vec3b>(0, 2) = cv::Vec3b(128, 64, 200);
    bgr_u8.at<cv::Vec3b>(0, 3) = cv::Vec3b(50, 150, 100);
    bgr_u8.at<cv::Vec3b>(0, 4) = cv::Vec3b(200, 100, 50);

    cv::Mat linear_rgb = BgrToLinearRgb(bgr_u8);
    ASSERT_EQ(linear_rgb.type(), CV_32FC3);

    cv::Mat back = LinearRgbToBgr(linear_rgb);
    ASSERT_EQ(back.type(), CV_8UC3);

    for (int i = 0; i < 5; ++i) {
        const cv::Vec3b a = bgr_u8.at<cv::Vec3b>(0, i);
        const cv::Vec3b b = back.at<cv::Vec3b>(0, i);
        EXPECT_LE(std::abs(a[0] - b[0]), 1) << "B channel @ " << i;
        EXPECT_LE(std::abs(a[1] - b[1]), 1) << "G channel @ " << i;
        EXPECT_LE(std::abs(a[2] - b[2]), 1) << "R channel @ " << i;
    }
}

// ── LabToBgr: round-trip with BgrToLab ──────────────────────────────────────

TEST(ColorImage, LabToBgrFromBgrToLabRoundTrip) {
    std::mt19937 rng(0xDADA01);
    std::uniform_int_distribution<int> byte(0, 255);

    constexpr int kN = 256;
    cv::Mat bgr_u8(1, kN, CV_8UC3);
    for (int i = 0; i < kN; ++i) {
        bgr_u8.at<cv::Vec3b>(0, i) = cv::Vec3b(static_cast<std::uint8_t>(byte(rng)),
                                                static_cast<std::uint8_t>(byte(rng)),
                                                static_cast<std::uint8_t>(byte(rng)));
    }

    cv::Mat lab  = BgrToLab(bgr_u8);
    cv::Mat back = LabToBgr(lab);
    ASSERT_EQ(back.type(), CV_8UC3);

    for (int i = 0; i < kN; ++i) {
        const cv::Vec3b a = bgr_u8.at<cv::Vec3b>(0, i);
        const cv::Vec3b b = back.at<cv::Vec3b>(0, i);
        // Round-trip via Lab loses 1-2 byte precision in extreme corners.
        EXPECT_LE(std::abs(a[0] - b[0]), 2) << "B channel @ " << i;
        EXPECT_LE(std::abs(a[1] - b[1]), 2) << "G channel @ " << i;
        EXPECT_LE(std::abs(a[2] - b[2]), 2) << "R channel @ " << i;
    }
}
