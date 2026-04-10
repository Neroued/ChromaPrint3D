#pragma once

/// \file recipe_map.h
/// \brief Per-pixel recipe mapping produced by color matching.

#include "common.h"
#include "color.h"
#include "model_package.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ChromaPrint3D {

// Forward declarations — full definitions live in their own headers.
struct RasterProcResult;
class ColorDB;
struct PrintProfile;
struct Channel;

/// Aggregate statistics produced by RecipeMap::MatchFromRaster().
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
    int k_candidates             = 1; ///< k <= 1 uses single nearest neighbour.
    ColorSpace color_space       = ColorSpace::Lab;
    ClusterMethod cluster_method = ClusterMethod::KMeans; ///< Non-dither clustering method.
    int cluster_count            = 0;     ///< 0 = auto-detect, 1 = per-pixel, >= 2 = manual.
    int slic_target_superpixels  = 256;   ///< SLIC target superpixel count.
    float slic_compactness       = 10.0f; ///< SLIC compactness weight (> 0).
    int slic_iterations          = 10;    ///< SLIC refinement iterations.
    float slic_min_region_ratio  = 0.25f; ///< Small-region merge ratio [0, 1].
    DitherMethod dither          = DitherMethod::None; ///< Dithering method.
    float dither_strength        = 0.8f;               ///< Dither intensity [0, 1].
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

    /// Render the mapped colors as a BGRA image.
    /// Transparent pixels in \c mask are exported with alpha=0.
    cv::Mat ToBgraImage() const;

    /// Render the source mask (0/255) as a single-channel image.
    cv::Mat ToSourceMaskImage() const;

    /// Render one layer assignment as a BGR image using palette channel colors.
    cv::Mat ToLayerBgrImage(int layer_idx, const std::vector<Channel>& palette,
                            uint8_t background_b = 255, uint8_t background_g = 255,
                            uint8_t background_r = 255) const;

    /// Render one layer assignment as a BGRA image using palette channel colors.
    /// Transparent pixels in \c mask are exported with alpha=0.
    cv::Mat ToLayerBgraImage(int layer_idx, const std::vector<Channel>& palette) const;

    /// Match every pixel in \p img to the nearest recipe from \p dbs under
    /// the given \p profile, optionally aided by a trained model package.
    static RecipeMap MatchFromRaster(const RasterProcResult& img, std::span<const ColorDB> dbs,
                                     const PrintProfile& profile,
                                     const MatchConfig& cfg            = MatchConfig{},
                                     const ModelPackage* model_package = nullptr,
                                     const ModelGateConfig& model_gate = ModelGateConfig{},
                                     MatchStats* out_stats             = nullptr);

    /// Replace recipes for all pixels in \p target_region_ids with \p new_recipe.
    /// Also updates \c mapped_color and \c source_mask for affected pixels.
    void ReplaceRecipeInRegions(const struct RasterRegionMap& region_map,
                                const std::vector<int>& target_region_ids,
                                const std::vector<uint8_t>& new_recipe, const Lab& new_mapped_color,
                                bool new_from_model);

    /// Upgrade the recipe buffer to a higher color_layers count by padding
    /// each pixel's recipe with \p pad_channel at the bottom (near base plate).
    /// No-op if \p new_color_layers <= current color_layers.
    void UpgradeColorLayers(int new_color_layers, uint8_t pad_channel);
};

} // namespace ChromaPrint3D
