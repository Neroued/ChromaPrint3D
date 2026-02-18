#pragma once

/// \file recipe_map.h
/// \brief Per-pixel recipe mapping produced by color matching.

#include "common.h"
#include "color.h"
#include "model_package.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ChromaPrint3D {

// Forward declarations — full definitions live in their own headers.
struct ImgProcResult;
class ColorDB;
struct PrintProfile;
/// Aggregate statistics produced by RecipeMap::MatchFromImage().
struct MatchStats {
    int clusters_total = 0;
    int db_only        = 0;
    int model_fallback = 0;
    int model_queries  = 0;
    float avg_db_de    = 0.0f;
    float avg_model_de = 0.0f;
};

/// Parameters that control the matching algorithm.
struct MatchConfig {
    int k_candidates       = 1;  ///< k <= 1 uses single nearest neighbour.
    ColorSpace color_space = ColorSpace::Lab;
    int cluster_count      = 64; ///< <= 1 degrades to per-pixel matching.
};

/// Per-pixel recipe map: for every pixel stores which channel to use at each
/// color layer, plus a mapped Lab color and optional source mask.
struct RecipeMap {
    std::string name;

    int width        = 0;
    int height       = 0;
    int color_layers = 0;
    int num_channels = 0;

    LayerOrder layer_order = LayerOrder::Top2Bottom;

    std::vector<uint8_t> recipes;     ///< H * W * color_layers
    std::vector<uint8_t> mask;        ///< H * W  (0 = transparent)
    std::vector<Lab> mapped_color;    ///< H * W
    std::vector<uint8_t> source_mask; ///< H * W, 0 = DB, 1 = Model fallback

    const uint8_t* RecipeAt(int r, int c) const;
    const uint8_t* MaskAt(int r, int c) const;
    const Lab ColorAt(int r, int c) const;

    /// Render the mapped colors as a BGR image.
    cv::Mat ToBgrImage(uint8_t background_b = 0, uint8_t background_g = 0,
                       uint8_t background_r = 0) const;

    /// Render the source mask (0/255) as a single-channel image.
    cv::Mat ToSourceMaskImage() const;

    /// Match every pixel in \p img to the nearest recipe from \p dbs under
    /// the given \p profile, optionally aided by a trained model package.
    static RecipeMap MatchFromImage(const ImgProcResult& img, std::span<const ColorDB> dbs,
                                    const PrintProfile& profile,
                                    const MatchConfig& cfg            = MatchConfig{},
                                    const ModelPackage* model_package = nullptr,
                                    const ModelGateConfig& model_gate = ModelGateConfig{},
                                    MatchStats* out_stats             = nullptr);
};

} // namespace ChromaPrint3D
