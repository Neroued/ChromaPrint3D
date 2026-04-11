/// \file pipeline.h
/// \brief Conversion pipelines for raster/vector image to 3D model conversion.

#pragma once

#include "common.h"
#include "color_db.h"
#include "model_package.h"
#include "print_profile.h"
#include "recipe_map.h"
#include "vector_proc.h"
#include "vector_recipe_map.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace ChromaPrint3D {

// ── Raster (bitmap) conversion pipeline ──────────────────────────────────────

/// Request parameters for raster-image-to-3D model conversion.
struct ConvertRasterRequest {
    // Image input (buffer takes priority if non-empty)
    std::string image_path; ///< Path to input image file (ignored if image_buffer is non-empty).
    std::vector<uint8_t> image_buffer; ///< Image data buffer (takes priority over image_path).
    std::string image_name;            ///< Image name used for naming when loading from buffer.

    // ColorDB input (preloaded_dbs takes priority if non-empty)
    std::vector<std::string>
        db_paths; ///< ColorDB file paths or directories (ignored if preloaded_dbs is non-empty).
    std::vector<const ColorDB*>
        preloaded_dbs; ///< Preloaded ColorDB instances (takes priority over db_paths).
    /// Owned copies of session-uploaded ColorDBs; keeps preloaded_dbs pointers valid
    /// after the originating session snapshot is destroyed.
    std::vector<std::shared_ptr<const ColorDB>> session_owned_dbs;

    // Model package (optional, preloaded takes priority)
    std::string
        model_pack_path; ///< Path to model package file (ignored if preloaded_model_pack is set).
    const ModelPackage* preloaded_model_pack =
        nullptr; ///< Preloaded model package (takes priority over model_pack_path).

    // Image processing
    float scale    = 1.0f; ///< Image scaling factor.
    int max_width  = 512;  ///< Maximum image width in pixels.
    int max_height = 512;  ///< Maximum image height in pixels.

    // Target physical size (mm). When > 0, overrides max_width/max_height.
    float target_width_mm  = 0.0f; ///< Target physical width in mm (0 = use max_width in pixels).
    float target_height_mm = 0.0f; ///< Target physical height in mm (0 = use max_height in pixels).

    // Matching
    int color_layers             = PrintProfile::kDefaultColorLayers; ///< Number of color layers.
    ColorSpace color_space       = ColorSpace::Lab; ///< Color space used for matching.
    int k_candidates             = 1; ///< Number of candidate colors to consider per pixel.
    ClusterMethod cluster_method = ClusterMethod::KMeans; ///< Non-dither clustering method.
    int cluster_count            = 0;     ///< 0 = auto-detect, 1 = per-pixel, >= 2 = manual.
    int slic_target_superpixels  = 256;   ///< SLIC target superpixel count.
    float slic_compactness       = 10.0f; ///< SLIC compactness weight (> 0).
    int slic_iterations          = 10;    ///< SLIC refinement iterations.
    float slic_min_region_ratio  = 0.25f; ///< SLIC small-region merge ratio [0, 1].
    DitherMethod dither          = DitherMethod::None; ///< Dithering method for matching.
    float dither_strength        = 0.8f;               ///< Dither intensity [0, 1].
    std::vector<std::string> allowed_channel_keys; ///< Allowed channel filters (empty = use all;
                                                   ///< "color|material" format).

    // Model gate
    bool model_enable     = true;  ///< Enable model-based filtering.
    bool model_only       = false; ///< Use only model predictions (skip color matching).
    float model_threshold = -1.0f; ///< Model confidence threshold (<0 uses package default).
    float model_margin    = -1.0f; ///< Model margin (<0 uses package default).

    // Geometry
    bool flip_y                = true; ///< Flip image vertically.
    float pixel_mm             = 0.0f; ///< Pixel size in millimeters (0 = derive from profile).
    float layer_height_mm      = 0.0f; ///< Layer height in millimeters (0 = derive from profile).
    int base_layers            = -1; ///< Base layers override (-1 = inherit from profile/ColorDB).
    bool double_sided          = false; ///< Generate mirrored color layers on both sides.
    float transparent_layer_mm = 0.0f;  ///< Transparent coating thickness (0=off, 0.04 or 0.08).

    // Output control
    bool generate_preview     = true;  ///< Generate preview image.
    bool generate_source_mask = true;  ///< Generate source mask image.
    std::string output_3mf_path;       ///< Output path for 3MF model file (empty = don't write).
    bool output_3mf_file_only = false; ///< Write only to file; model_3mf stays empty.
    std::string preview_path;          ///< Output path for preview PNG (empty = don't write).
    std::string source_mask_path;      ///< Output path for source mask PNG (empty = don't write).

    std::string preset_dir; ///< Directory containing preset JSON files (empty = no preset).

    NozzleSize nozzle_size           = NozzleSize::N04;
    FaceOrientation face_orientation = FaceOrientation::FaceUp;
};

/// Palette channel metadata used by layer preview artifacts.
struct LayerPreviewChannel {
    int channel_idx = 0;
    std::string color;
    std::string material;
};

/// Layered preview artifacts for debugging recipe assignments.
struct LayerPreviewResult {
    int layers             = 0;
    int width              = 0;
    int height             = 0;
    LayerOrder layer_order = LayerOrder::Top2Bottom;
    std::vector<LayerPreviewChannel> palette;
    std::vector<std::vector<uint8_t>> layer_pngs; ///< One PNG per layer.
};

/// Result of image-to-3D model conversion (shared by raster and vector pipelines).
struct ConvertResult {
    MatchStats stats; ///< Color matching statistics.

    int input_width  = 0; ///< Processed input width in pixels (raster) or 0 (vector).
    int input_height = 0; ///< Processed input height in pixels (raster) or 0 (vector).

    float resolved_pixel_mm  = 0.0f; ///< Actual pixel size used (mm), raster only.
    float physical_width_mm  = 0.0f; ///< Output physical width (mm).
    float physical_height_mm = 0.0f; ///< Output physical height (mm).

    std::vector<uint8_t> model_3mf;       ///< 3MF model file data.
    std::vector<uint8_t> preview_png;     ///< Preview PNG image data.
    std::vector<uint8_t> source_mask_png; ///< Source mask PNG image data.
    LayerPreviewResult layer_previews;    ///< Per-layer preview images and metadata.

    std::vector<std::string> warnings; ///< Non-fatal warnings for the user.
};

/// Conversion pipeline stage (shared by raster and vector pipelines).
enum class ConvertStage : uint8_t {
    LoadingResources, ///< Loading ColorDBs and model packages.
    Preprocessing,    ///< Preprocessing input (raster resize/denoise or SVG parse/clip).
    Matching,         ///< Matching colors and generating recipes.
    BuildingModel,    ///< Building 3D model geometry.
    Exporting,        ///< Exporting 3MF and images.
};

/// Progress callback function type.
using ProgressCallback = std::function<void(ConvertStage stage, float progress)>;

/// Raster conversion: converts a bitmap image to a 3D model using ColorDB matching.
ConvertResult ConvertRaster(const ConvertRasterRequest& request,
                            ProgressCallback progress = nullptr);

/// Intermediate result of the matching phase (stages 1-3).
/// Holds all state needed for recipe editing and subsequent model generation.
struct MatchRasterResult {
    RecipeMap recipe_map;
    PrintProfile profile;
    MatchStats stats;

    float resolved_pixel_mm          = 0.0f;
    float layer_height_mm            = 0.0f;
    int base_layers                  = 0;
    bool custom_base_layers          = false;
    bool double_sided                = false;
    float transparent_layer_mm       = 0.0f;
    bool flip_y                      = true;
    NozzleSize nozzle_size           = NozzleSize::N04;
    FaceOrientation face_orientation = FaceOrientation::FaceUp;
    std::string preset_dir;

    std::vector<ColorDB> dbs;
    MatchConfig match_config;
    ModelGateConfig model_gate;

    int input_width  = 0;
    int input_height = 0;

    bool generate_preview     = true;
    bool generate_source_mask = true;

    std::vector<std::string> warnings; ///< Non-fatal warnings for the user.

    std::size_t EstimateBytes() const;

    /// Upgrade both recipe_map and profile to a higher color_layers count,
    /// padding with base material. Ensures recipe_map.color_layers == profile.color_layers.
    void UpgradeColorLayers(int new_color_layers);
};

/// Raster matching phase: loads resources, preprocesses image, matches colors.
MatchRasterResult MatchRaster(const ConvertRasterRequest& request,
                              ProgressCallback progress = nullptr);

/// Generates a 3D model from a (possibly edited) MatchRasterResult.
ConvertResult GenerateRasterModel(MatchRasterResult& match_result,
                                  ProgressCallback progress = nullptr);

/// Resolves ColorDB file paths from input paths (files or directories).
std::vector<std::string> ResolveDBPaths(const std::vector<std::string>& input_paths);

// ── Vector (SVG) conversion pipeline ─────────────────────────────────────────

/// Request parameters for SVG-to-3D model conversion.
struct ConvertVectorRequest {
    std::string svg_path;            ///< Path to SVG file.
    std::vector<uint8_t> svg_buffer; ///< SVG data buffer (takes priority over svg_path).
    std::string svg_name;            ///< Name when loading from buffer.

    std::vector<std::string> db_paths;
    std::vector<const ColorDB*> preloaded_dbs;
    std::vector<std::shared_ptr<const ColorDB>> session_owned_dbs;

    std::string model_pack_path;
    const ModelPackage* preloaded_model_pack = nullptr;

    float target_width_mm  = 0.0f; ///< 0 = use SVG original size.
    float target_height_mm = 0.0f; ///< 0 = use SVG original size.

    int color_layers       = PrintProfile::kDefaultColorLayers; ///< Number of color layers.
    ColorSpace color_space = ColorSpace::Lab;
    int k_candidates       = 1;
    bool model_enable      = true;
    bool model_only        = false;
    float model_threshold  = -1.0f;
    float model_margin     = -1.0f;
    std::vector<std::string> allowed_channel_keys;

    float gradient_pixel_mm = 0.0f; ///< Gradient rasterization resolution (0 = auto).

    float layer_height_mm           = 0.0f; ///< 0 = derive from profile.
    float tessellation_tolerance_mm = 0.03f;
    bool flip_y                     = false;
    int base_layers            = -1; ///< Base layers override (-1 = inherit from profile/ColorDB).
    bool double_sided          = false; ///< Generate mirrored color layers on both sides.
    float transparent_layer_mm = 0.0f;  ///< Transparent coating thickness (0=off, 0.04 or 0.08).

    std::string output_3mf_path;
    bool output_3mf_file_only = false; ///< Write only to file; model_3mf stays empty.
    bool generate_preview     = true;
    bool generate_source_mask = true; ///< Generate source mask image for vector matching source.

    std::string preset_dir; ///< Directory containing preset JSON files (empty = no preset).

    NozzleSize nozzle_size           = NozzleSize::N04;
    FaceOrientation face_orientation = FaceOrientation::FaceUp;
};

/// Intermediate result of the vector matching phase (stages 1-3).
/// Holds all state needed for recipe editing and subsequent model generation.
struct MatchVectorResult {
    VectorRecipeMap recipe_map;
    VectorProcResult proc_result;
    PrintProfile profile;
    MatchStats stats;

    float layer_height_mm            = 0.0f;
    int base_layers                  = 0;
    bool custom_base_layers          = false;
    bool double_sided                = false;
    float transparent_layer_mm       = 0.0f;
    NozzleSize nozzle_size           = NozzleSize::N04;
    FaceOrientation face_orientation = FaceOrientation::FaceUp;
    std::string preset_dir;

    std::vector<ColorDB> dbs;
    MatchConfig match_config;
    ModelGateConfig model_gate;

    bool generate_preview     = true;
    bool generate_source_mask = true;

    std::vector<std::string> warnings; ///< Non-fatal warnings for the user.

    std::size_t EstimateBytes() const;

    /// Upgrade both recipe_map and profile to a higher color_layers count,
    /// padding with base material. Ensures recipe_map.color_layers == profile.color_layers.
    void UpgradeColorLayers(int new_color_layers);
};

/// Vector matching phase: loads resources, preprocesses SVG, matches colors.
MatchVectorResult MatchVector(const ConvertVectorRequest& request,
                              ProgressCallback progress = nullptr);

/// Generates a 3D model from a (possibly edited) MatchVectorResult.
ConvertResult GenerateVectorModel(MatchVectorResult& match_result,
                                  ProgressCallback progress = nullptr);

/// Vector conversion: converts an SVG to a 3D model.
ConvertResult ConvertVector(const ConvertVectorRequest& request,
                            ProgressCallback progress = nullptr);

} // namespace ChromaPrint3D
