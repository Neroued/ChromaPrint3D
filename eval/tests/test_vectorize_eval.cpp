#include "chromaprint3d/vectorize_eval.h"
#include "pixel_metrics.h"
#include "edge_metrics.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <limits>

using namespace ChromaPrint3D;

// ── PSNR formula tests ──────────────────────────────────────────────────────

TEST(PixelMetrics, IdenticalImagesPsnrInfSsimOne) {
    cv::Mat img(64, 64, CV_8UC3, cv::Scalar(128, 64, 200));
    cv::Mat coverage(64, 64, CV_8UC1, cv::Scalar(255));
    cv::Mat shape_count(64, 64, CV_16UC1, cv::Scalar(1));

    auto r = eval::ComputePixelMetrics(img, img, coverage, shape_count);
    EXPECT_TRUE(std::isinf(r.psnr));
    EXPECT_NEAR(r.ssim, 1.0, 0.001);
    EXPECT_NEAR(r.delta_e_mean, 0.0, 0.01);
    EXPECT_NEAR(r.coverage, 1.0, 0.001);
}

TEST(PixelMetrics, BlackVsWhiteLowSsim) {
    cv::Mat black(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat white(64, 64, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat coverage(64, 64, CV_8UC1, cv::Scalar(255));
    cv::Mat shape_count(64, 64, CV_16UC1, cv::Scalar(1));

    auto r = eval::ComputePixelMetrics(black, white, coverage, shape_count);
    EXPECT_LT(r.psnr, 1.0);
    EXPECT_LT(r.ssim, 0.05);
    EXPECT_GT(r.delta_e_mean, 50.0);
}

TEST(PixelMetrics, CoverageAndOverlap) {
    cv::Mat orig(100, 100, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat rend(100, 100, CV_8UC3, cv::Scalar(100, 100, 100));

    // Half covered
    cv::Mat coverage                  = cv::Mat::zeros(100, 100, CV_8UC1);
    coverage(cv::Rect(0, 0, 50, 100)) = 255;

    // 25% overlap
    cv::Mat shape_count = cv::Mat::ones(100, 100, CV_16UC1);
    shape_count(cv::Rect(0, 0, 50, 50)).setTo(2);

    auto r = eval::ComputePixelMetrics(orig, rend, coverage, shape_count);
    EXPECT_NEAR(r.coverage, 0.5, 0.01);
    EXPECT_NEAR(r.overlap, 0.25, 0.01);
}

// ── Alpha mask tests ────────────────────────────────────────────────────────

TEST(PixelMetrics, AlphaMaskExcludesTransparentRegion) {
    // 100x100 image: left half is "content" (red), right half is "transparent" (garbage data).
    // Without alpha mask: rendered white vs original garbage → low scores.
    // With alpha mask: transparent half is excluded → correct comparison.
    cv::Mat orig(100, 100, CV_8UC3, cv::Scalar(0, 0, 200)); // red everywhere
    cv::Mat rend(100, 100, CV_8UC3, cv::Scalar(0, 0, 200)); // red everywhere (perfect match)

    // Make the right half of original "garbage" (simulating undefined alpha=0 pixel data)
    orig(cv::Rect(50, 0, 50, 100)).setTo(cv::Scalar(0, 0, 0)); // black garbage

    // SVG coverage only covers the left half (the actual content)
    cv::Mat coverage                  = cv::Mat::zeros(100, 100, CV_8UC1);
    coverage(cv::Rect(0, 0, 50, 100)) = 255;

    cv::Mat shape_count(100, 100, CV_16UC1, cv::Scalar(1));

    // Alpha mask: left half = opaque (255), right half = transparent (0)
    cv::Mat alpha_mask                  = cv::Mat::zeros(100, 100, CV_8UC1);
    alpha_mask(cv::Rect(0, 0, 50, 100)) = 255;

    auto r = eval::ComputePixelMetrics(orig, rend, coverage, shape_count, alpha_mask);

    // With alpha mask excluding the garbage half:
    // - Coverage should be 1.0 (all opaque pixels are covered)
    // - PSNR should be inf (red vs red in the opaque region)
    // - SSIM should be ~1.0
    EXPECT_NEAR(r.coverage, 1.0, 0.01);
    EXPECT_TRUE(std::isinf(r.psnr));
    EXPECT_GT(r.ssim, 0.95);
}

TEST(PixelMetrics, AlphaMaskAffectsCoverage) {
    cv::Mat orig(100, 100, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat rend(100, 100, CV_8UC3, cv::Scalar(100, 100, 100));

    // SVG covers left 30 columns
    cv::Mat coverage                  = cv::Mat::zeros(100, 100, CV_8UC1);
    coverage(cv::Rect(0, 0, 30, 100)) = 255;

    cv::Mat shape_count(100, 100, CV_16UC1, cv::Scalar(1));

    // Alpha mask: only left 30 columns are opaque (content area)
    cv::Mat alpha_mask                  = cv::Mat::zeros(100, 100, CV_8UC1);
    alpha_mask(cv::Rect(0, 0, 30, 100)) = 255;

    auto r = eval::ComputePixelMetrics(orig, rend, coverage, shape_count, alpha_mask);
    // Coverage should be 1.0: all opaque pixels are covered
    EXPECT_NEAR(r.coverage, 1.0, 0.01);

    // Without alpha mask: coverage would be 0.30
    auto r2 = eval::ComputePixelMetrics(orig, rend, coverage, shape_count);
    EXPECT_NEAR(r2.coverage, 0.30, 0.01);
}

// ── Edge metrics tests ───────────────────────────────────────────────────────

TEST(EdgeMetrics, IdenticalImagesF1One) {
    cv::Mat img(64, 64, CV_8UC3, cv::Scalar(128, 64, 200));
    auto r = eval::ComputeEdgeMetrics(img, img, 2);
    EXPECT_NEAR(r.edge_f1, 1.0, 0.01);
    EXPECT_NEAR(r.chamfer_distance, 0.0, 0.5);
}

// ── Score tests ──────────────────────────────────────────────────────────────

TEST(Scoring, PerfectScore) {
    VectorizeMetrics m;
    m.coverage           = 1.0;
    m.delta_e_mean       = 0;
    m.ssim               = 1.0;
    m.edge_f1            = 1.0;
    m.tiny_fragment_rate = 0;

    double score = ComputeScore(m);
    // 40*1 + 30*1 + 15*1 + 15*(1-0)*1 = 100
    EXPECT_NEAR(score, 100.0, 0.01);
}

TEST(Scoring, DegradedScore) {
    VectorizeMetrics m;
    m.delta_e_mean       = 20.0;
    m.ssim               = 0.5;
    m.edge_f1            = 0.7;
    m.coverage           = 0.95;
    m.tiny_fragment_rate = 0.8;

    ScoreWeights w;
    double score = ComputeScore(m, w);
    // fidelity  = max(0, 1 - 20/40) = 0.5  -> 40*0.5 = 20.0
    // structure = 0.5                       -> 30*0.5 = 15.0
    // edge      = 0.7                       -> 15*0.7 = 10.5
    // efficiency = (1-0.8)*0.95 = 0.19      -> 15*0.19 = 2.85
    // total = 48.35
    EXPECT_NEAR(score, 48.35, 0.1);
    EXPECT_GT(score, 0.0);
    EXPECT_LT(score, 100.0);
}

// ── PartialVectorizerConfig merge tests ──────────────────────────────────────

TEST(Config, MergeOverridesOnlySet) {
    VectorizerConfig base;
    base.num_colors       = 16;
    base.contour_simplify = 0.45f;

    PartialVectorizerConfig partial;
    partial.num_colors = 24;

    auto merged = partial.MergeInto(base);
    EXPECT_EQ(merged.num_colors, 24);
    EXPECT_FLOAT_EQ(merged.contour_simplify, 0.45f);
}

TEST(Config, ThreeLayerMerge) {
    VectorizerConfig base;
    PartialVectorizerConfig global;
    global.num_colors       = 24;
    global.contour_simplify = 0.6f;

    PartialVectorizerConfig per_image;
    per_image.num_colors = 8;

    auto cfg = base;
    cfg      = global.MergeInto(cfg);
    cfg      = per_image.MergeInto(cfg);

    EXPECT_EQ(cfg.num_colors, 8);
    EXPECT_FLOAT_EQ(cfg.contour_simplify, 0.6f);
}

// ── Expectations tests ───────────────────────────────────────────────────────

TEST(Expectations, PassAll) {
    VectorizeMetrics m;
    m.coverage     = 0.98;
    m.ssim         = 0.92;
    m.delta_e_mean = 5.0;

    Expectations e;
    e.min_coverage     = 0.95;
    e.min_ssim         = 0.9;
    e.max_delta_e_mean = 10.0;

    auto fails = e.Check(m);
    EXPECT_TRUE(fails.empty());
}

TEST(Expectations, FailCoverage) {
    VectorizeMetrics m;
    m.coverage = 0.90;

    Expectations e;
    e.min_coverage = 0.95;

    auto fails = e.Check(m);
    EXPECT_EQ(fails.size(), 1u);
    EXPECT_NE(fails[0].find("coverage"), std::string::npos);
}

// ── VectorizeMetrics JSON ────────────────────────────────────────────────────

TEST(Metrics, ToJsonContainsAllFields) {
    VectorizeMetrics m;
    m.psnr         = 25.5;
    m.ssim         = 0.9;
    m.coverage     = 0.99;
    m.total_shapes = 42;

    auto json = m.ToJson();
    EXPECT_NE(json.find("\"psnr\""), std::string::npos);
    EXPECT_NE(json.find("\"ssim\""), std::string::npos);
    EXPECT_NE(json.find("\"coverage\""), std::string::npos);
    EXPECT_NE(json.find("\"total_shapes\""), std::string::npos);
    EXPECT_NE(json.find("\"chamfer_distance\""), std::string::npos);
    EXPECT_NE(json.find("\"gini_coefficient\""), std::string::npos);
    EXPECT_NE(json.find("\"vectorize_time_ms\""), std::string::npos);
}
