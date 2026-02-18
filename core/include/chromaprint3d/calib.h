#pragma once

#include "common.h"
#include "color_db.h"
#include "mesh.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ChromaPrint3D {

struct CalibrationRecipeSpec {
    int num_channels       = 4;
    int color_layers       = 5; // 允许 5 或 10
    LayerOrder layer_order = LayerOrder::Top2Bottom;

    static constexpr int kMinChannels      = 2;
    static constexpr int kMaxChannels      = 8;
    static constexpr int kFixedColorLayers = 5;
    static constexpr int kAltColorLayers   = 10;
    static constexpr int kMaxRecipes       = 1024;

    size_t NumRecipes() const {
        if (num_channels <= 0 || color_layers <= 0) { return 0; }
        size_t count = 1;
        for (int i = 0; i < color_layers; ++i) { count *= static_cast<size_t>(num_channels); }
        return count;
    }

    bool IsSupported() const {
        if (color_layers != kFixedColorLayers && color_layers != kAltColorLayers) { return false; }
        if (num_channels < kMinChannels || num_channels > kMaxChannels) { return false; }
        return true;
    }

    std::vector<uint8_t> RecipeAt(size_t index) const {
        std::vector<uint8_t> recipe;
        if (num_channels <= 0 || color_layers <= 0) { return recipe; }
        recipe.resize(static_cast<size_t>(color_layers), 0);
        size_t idx = index;
        for (int i = 0; i < color_layers; ++i) {
            uint8_t v = static_cast<uint8_t>(idx % static_cast<size_t>(num_channels));
            idx /= static_cast<size_t>(num_channels);

            int layer = (layer_order == LayerOrder::Top2Bottom) ? (color_layers - 1 - i) : i;
            recipe[static_cast<size_t>(layer)] = v;
        }
        return recipe;
    }
};

struct CalibrationFiducialSpec {
    int offset_factor = 14;
    int main_d_factor = 12; // 主孔
    int tag_d_factor  = 8;  // 标签孔直径，在主孔旁边
    int tag_dx_factor = 20;
    int tag_dy_factor = 0;
};

struct CalibrationBoardLayout {
    float line_width_mm = 0.42f;
    int resolution_scale = 8; // 内部分辨率倍率，像素尺寸为 line_width_mm / resolution_scale
    int tile_factor   = 10; // 色块的大小为 tile_factor * line_width_mm
    int gap_factor    = 2;
    int margin_factor = 24;

    CalibrationFiducialSpec fiducial;
};

struct CalibrationBoardConfig {
    CalibrationRecipeSpec recipe;

    int base_layers       = 10;
    int base_channel_idx  = 0;
    float layer_height_mm = 0.08f;

    std::vector<Channel> palette; // size == recipe.num_channels

    CalibrationBoardLayout layout;

    size_t NumRecipes() const { return recipe.NumRecipes(); }

    bool IsSupported() const { return recipe.IsSupported(); }

    bool HasValidPalette() const { return static_cast<int>(palette.size()) == recipe.num_channels; }

    static CalibrationBoardConfig ForChannels(int num_channels) {
        CalibrationBoardConfig cfg;
        cfg.recipe.num_channels = num_channels;
        cfg.palette.resize(static_cast<size_t>(num_channels));
        return cfg;
    }
};

struct CalibrationBoardMeta {
    std::string name;
    CalibrationBoardConfig config;
    int grid_rows = 0;
    int grid_cols = 0;
    std::vector<uint16_t> patch_recipe_idx;          // row-major order
    std::vector<std::vector<uint8_t>> patch_recipes; // row-major order

    size_t NumPatches() const { return patch_recipe_idx.size(); }

    void SaveToJson(const std::string& path) const;
    static CalibrationBoardMeta LoadFromJson(const std::string& path);

    std::string ToJsonString() const;
    static CalibrationBoardMeta FromJsonString(const std::string& json_str);
};

struct CalibrationBoardResult {
    std::vector<uint8_t> model_3mf;
    CalibrationBoardMeta meta;
};

/// Intermediate result holding pre-built meshes for caching/re-export.
struct CalibrationBoardMeshes {
    CalibrationBoardMeta meta;
    std::vector<Mesh> meshes;     ///< One per channel + optional base grid.
    BuildMeshConfig mesh_cfg;
    int base_channel_idx = -1;
    int base_layers      = 0;
};

CalibrationBoardMeta BuildCalibrationBoardMeta(const CalibrationBoardConfig& cfg);

void GenCalibrationBoard(const CalibrationBoardConfig& cfg, const std::string& board_path,
                         const std::string& meta_path);
void GenCalibrationBoardFromMeta(const CalibrationBoardMeta& meta, const std::string& board_path,
                                 const std::string& meta_path);

CalibrationBoardResult GenCalibrationBoardToBuffer(const CalibrationBoardConfig& cfg);

/// Generate calibration board meshes (without 3MF export) for caching.
CalibrationBoardMeshes GenCalibrationBoardMeshes(const CalibrationBoardConfig& cfg);

/// Build a CalibrationBoardResult from cached meshes with a (possibly updated) palette.
CalibrationBoardResult BuildResultFromMeshes(const CalibrationBoardMeshes& cached,
                                             const std::vector<Channel>& palette);

ColorDB GenColorDBFromImage(const std::string& image_path, const CalibrationBoardMeta& meta);
ColorDB GenColorDBFromImage(const std::string& image_path, const std::string& json_path);
ColorDB GenColorDBFromBuffer(const std::vector<uint8_t>& image_buffer,
                             const CalibrationBoardMeta& meta);

} // namespace ChromaPrint3D