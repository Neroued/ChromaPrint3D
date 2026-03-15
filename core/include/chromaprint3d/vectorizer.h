#pragma once

/// \file vectorizer.h
/// \brief Public API for raster-to-SVG vectorization.

#include "color.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ChromaPrint3D {

/// Configuration for the vectorization pipeline.
struct VectorizerConfig {
    // ── Color segmentation ──────────────────────────────────────────────────
    int num_colors      = 0;  ///< K-Means palette size. 0 = auto-detect optimal count.
    int min_region_area = 50; ///< Force-merge regions smaller than this (pixels²).

    // ── Curve fitting ───────────────────────────────────────────────────────
    float curve_fit_error = 0.8f; ///< Schneider curve fitting error threshold (pixels).
    float corner_angle_threshold =
        135.0f;              ///< Corner detection threshold angle in degrees for curve splitting.
    float smoothness = 0.5f; ///< Contour smoothness [0,1]. 0 = preserve all detail, 1 = maximum
                             ///< smoothing. Controls decimation epsilon, smoothing displacement
                             ///< and iterations on assembled contours.

    // ── Preprocessing ───────────────────────────────────────────────────────
    float smoothing_spatial = 15.0f; ///< Mean Shift spatial window radius.
    float smoothing_color   = 25.0f; ///< Mean Shift color window radius.
    int upscale_short_edge =
        600; ///< Auto-upscale when image short edge is below this threshold (0 disables).
    int max_working_pixels =
        3000000; ///< Auto-downscale when input pixels exceed this threshold (0 disables).

    // ── Segmentation ────────────────────────────────────────────────────────
    int slic_region_size   = 20;   ///< SLIC target region size for multicolor mode.
    float slic_compactness = 6.0f; ///< SLIC compactness (lower = follow color edges more).
    float edge_sensitivity = 0.8f; ///< Edge-aware SLIC spatial weight reduction [0,1].
    int refine_passes      = 6;    ///< Boundary label refinement iterations (0 disables).
    float max_merge_color_dist =
        200.0f; ///< Max LAB ΔE² for small-region merging (higher = merge more aggressively).

    // ── Subpixel boundary refinement ────────────────────────────────────────
    bool enable_subpixel_refine     = true; ///< Gradient-guided sub-pixel boundary refinement.
    float subpixel_max_displacement = 0.7f; ///< Max normal displacement for sub-pixel refine (px).

    // ── Anti-aliasing detection ──────────────────────────────────────────────
    bool enable_antialias_detect = false; ///< Detect AA mixed-edge pixels for better boundaries.
    float aa_tolerance = 10.0f; ///< Max LAB Delta-E for a pixel to qualify as an AA blend.

    // ── Thin-line enhancement ───────────────────────────────────────────────
    float thin_line_max_radius =
        2.5f; ///< Distance-transform radius threshold for thin-line extraction.

    // ── SVG output ──────────────────────────────────────────────────────────
    bool svg_enable_stroke = true; ///< Optional stroke output for visual debugging.
    float svg_stroke_width = 0.5f; ///< Stroke width when svg_enable_stroke is true.

    // ── Detail / node-count control ─────────────────────────────────────────
    float detail_level = -1.0f; ///< Unified detail control [0,1]. When >= 0, auto-derives
                                ///< curve_fit_error and contour_simplify unless they were
                                ///< explicitly set. -1 means disabled (use explicit params).
    float merge_segment_tolerance =
        0.05f; ///< Max control-point deviation (fraction of chord) to merge near-linear Bezier
               ///< segments. 0 disables merging.

    // ── Potrace pipeline knobs ──────────────────────────────────────────────
    float min_contour_area   = 10.0f;  ///< Discard shapes smaller than this (pixels²).
    float min_hole_area      = 4.0f;   ///< Minimum hole area retained in final paths.
    float contour_simplify   = 0.45f;  ///< Contour simplification strength (larger => fewer nodes).
    bool enable_coverage_fix = true;   ///< Patch uncovered pixels after vectorization.
    float min_coverage_ratio = 0.998f; ///< Minimum coverage ratio before patching.
};

/// Result of the vectorization pipeline.
struct VectorizerResult {
    std::string svg_content;     ///< Complete SVG document.
    int width               = 0; ///< Image width in pixels.
    int height              = 0; ///< Image height in pixels.
    int num_shapes          = 0; ///< Number of shapes in the SVG.
    int resolved_num_colors = 0; ///< Actual color count used (from auto-detection or config).
    std::vector<Rgb> palette;    ///< Color palette used.
};

/// Vectorize a raster image file to SVG.
///
/// \param image_path  Path to input image (PNG, JPG, BMP, etc.).
/// \param config      Pipeline configuration.
/// \return            Vectorization result with SVG content.
VectorizerResult Vectorize(const std::string& image_path, const VectorizerConfig& config = {});

/// Vectorize an in-memory image buffer to SVG (ICC-aware).
///
/// \param image_data  Pointer to encoded image bytes (JPEG, PNG, etc.).
/// \param image_size  Number of bytes.
/// \note              For RGBA PNG inputs, fully transparent pixels are excluded from
/// vectorization. \param config      Pipeline configuration. \return            Vectorization
/// result with SVG content.
VectorizerResult Vectorize(const uint8_t* image_data, size_t image_size,
                           const VectorizerConfig& config = {});

/// Vectorize a BGR cv::Mat to SVG.
///
/// \param bgr_image   Input image in BGR/BGRA/GRAY format.
/// \note              For BGRA input, pixels with alpha==0 are excluded from vectorization.
/// \param config      Pipeline configuration.
/// \return            Vectorization result with SVG content.
VectorizerResult Vectorize(const cv::Mat& bgr_image, const VectorizerConfig& config = {});

} // namespace ChromaPrint3D
