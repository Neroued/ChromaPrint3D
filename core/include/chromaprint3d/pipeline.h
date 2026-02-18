/// \file pipeline.h
/// \brief Main conversion pipeline for image to 3D model conversion.

#pragma once

#include "common.h"
#include "color_db.h"
#include "print_profile.h"
#include "recipe_map.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ChromaPrint3D {

/// Request parameters for image-to-3D model conversion.
struct ConvertRequest {
    // Image input (buffer takes priority if non-empty)
    std::string image_path; ///< Path to input image file (ignored if image_buffer is non-empty).
    std::vector<uint8_t> image_buffer; ///< Image data buffer (takes priority over image_path).
    std::string image_name; ///< Image name used for naming when loading from buffer.

    // ColorDB input (preloaded_dbs takes priority if non-empty)
    std::vector<std::string> db_paths; ///< ColorDB file paths or directories (ignored if preloaded_dbs is non-empty).
    std::vector<const ColorDB*> preloaded_dbs; ///< Preloaded ColorDB instances (takes priority over db_paths).

    // Model package (optional, preloaded takes priority)
    std::string model_pack_path; ///< Path to model package file (ignored if preloaded_model_pack is set).
    const ModelPackage* preloaded_model_pack = nullptr; ///< Preloaded model package (takes priority over model_pack_path).

    // Image processing
    float scale    = 1.0f; ///< Image scaling factor.
    int max_width  = 512; ///< Maximum image width in pixels.
    int max_height = 512; ///< Maximum image height in pixels.

    // Matching
    PrintMode print_mode   = PrintMode::Mode0p08x5; ///< Print mode/profile.
    ColorSpace color_space = ColorSpace::Lab; ///< Color space used for matching.
    int k_candidates       = 1; ///< Number of candidate colors to consider per pixel.
    int cluster_count      = 64; ///< Number of color clusters for quantization.
    std::vector<std::string> allowed_channel_keys; ///< Allowed channel filters (empty = use all; "color|material" format).

    // Model gate
    bool model_enable     = true; ///< Enable model-based filtering.
    bool model_only       = false; ///< Use only model predictions (skip color matching).
    float model_threshold = -1.0f; ///< Model confidence threshold (<0 uses package default).
    float model_margin    = -1.0f; ///< Model margin (<0 uses package default).

    // Geometry
    bool flip_y           = true; ///< Flip image vertically.
    float pixel_mm        = 0.0f; ///< Pixel size in millimeters (0 = derive from profile).
    float layer_height_mm = 0.0f; ///< Layer height in millimeters (0 = derive from profile).

    // Output control
    bool generate_preview     = true; ///< Generate preview image.
    bool generate_source_mask = true; ///< Generate source mask image.
    // File output paths (empty = don't write file, only return buffer)
    std::string output_3mf_path; ///< Output path for 3MF model file (empty = don't write).
    std::string preview_path; ///< Output path for preview PNG (empty = don't write).
    std::string source_mask_path; ///< Output path for source mask PNG (empty = don't write).
};

/// Result of image-to-3D model conversion.
struct ConvertResult {
    MatchStats stats; ///< Color matching statistics.

    int image_width  = 0; ///< Processed image width in pixels.
    int image_height = 0; ///< Processed image height in pixels.

    // In-memory buffer outputs (always populated when corresponding generate flag is true)
    std::vector<uint8_t> model_3mf; ///< 3MF model file data.
    std::vector<uint8_t> preview_png; ///< Preview PNG image data.
    std::vector<uint8_t> source_mask_png; ///< Source mask PNG image data.
};

/// Conversion pipeline stage.
enum class ConvertStage : uint8_t {
    LoadingResources, ///< Loading ColorDBs and model packages.
    ProcessingImage,  ///< Processing and resizing input image.
    Matching,         ///< Matching colors and generating recipes.
    BuildingModel,    ///< Building 3D model geometry.
    Exporting,        ///< Exporting 3MF and images.
};

/// Progress callback function type.
/// \param stage Current conversion stage
/// \param progress Progress value [0.0, 1.0] within the current stage
using ProgressCallback = std::function<void(ConvertStage stage, float progress)>;

/// Main conversion function: converts an image to a 3D model using ColorDB matching.
/// \param request Conversion request parameters
/// \param progress Optional progress callback function
/// \return Conversion result containing model and images
ConvertResult Convert(const ConvertRequest& request, ProgressCallback progress = nullptr);

/// Resolves ColorDB file paths from input paths (files or directories).
/// \param input_paths Input file paths or directory paths
/// \return Resolved list of ColorDB file paths
std::vector<std::string> ResolveDBPaths(const std::vector<std::string>& input_paths);

} // namespace ChromaPrint3D
