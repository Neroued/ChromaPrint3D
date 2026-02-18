#include "chromaprint3d/calib.h"
#include "chromaprint3d/voxel.h"
#include "chromaprint3d/mesh.h"
#include "chromaprint3d/export_3mf.h"
#include "chromaprint3d/recipe_map.h"
#include "chromaprint3d/error.h"
#include "chromaprint3d/logging.h"
#include "detail/cv_utils.h"
#include "detail/json_utils.h"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>

namespace ChromaPrint3D {

cv::Mat LocateCalibrationColorRegion(const std::string& image_path,
                                     const CalibrationBoardMeta& meta);
cv::Mat LocateCalibrationColorRegion(const cv::Mat& input, const CalibrationBoardMeta& meta);

using nlohmann::json;

namespace {

constexpr uint16_t kInvalidRecipeIdx = 0xFFFF;

static void ValidateConfig(const CalibrationBoardConfig& cfg) {
    if (!cfg.recipe.IsSupported()) {
        throw ConfigError("Calibration recipe is not supported");
    }
    if (!cfg.palette.empty() && static_cast<int>(cfg.palette.size()) != cfg.recipe.num_channels) {
        throw ConfigError("Calibration palette size does not match num_channels");
    }
    if (cfg.base_layers < 0) { throw InputError("base_layers is invalid"); }
    if (cfg.base_layers > 0 &&
        (cfg.base_channel_idx < 0 || cfg.base_channel_idx >= cfg.recipe.num_channels)) {
        throw InputError("base_channel_idx is out of range");
    }
    if (cfg.layout.line_width_mm <= 0.0f) {
        throw InputError("line_width_mm must be positive");
    }
    if (cfg.layout.resolution_scale <= 0) {
        throw InputError("resolution_scale must be positive");
    }
    if (cfg.layout.tile_factor <= 0) { throw InputError("tile_factor must be positive"); }
    if (cfg.layout.gap_factor < 0) { throw InputError("gap_factor is invalid"); }
    if (cfg.layout.margin_factor < 0) { throw InputError("margin_factor is invalid"); }
    if (cfg.layout.fiducial.main_d_factor <= 0) {
        throw InputError("fiducial main_d_factor must be positive");
    }
}

static int ComputeGridSize(std::size_t recipe_count) {
    if (recipe_count == 0) { throw InputError("recipe_count is zero"); }
    const double side = std::ceil(std::sqrt(static_cast<double>(recipe_count)));
    const int grid    = static_cast<int>(side);
    if (grid <= 0) { throw InputError("grid size is invalid"); }
    return grid;
}

static std::vector<uint8_t> BuildBackgroundRecipe(const CalibrationBoardConfig& cfg) {
    std::vector<uint8_t> recipe(static_cast<size_t>(cfg.recipe.color_layers), 0);
    int channel = 0;
    if (cfg.base_channel_idx >= 0 && cfg.base_channel_idx < cfg.recipe.num_channels) {
        channel = cfg.base_channel_idx;
    }
    std::fill(recipe.begin(), recipe.end(), static_cast<uint8_t>(channel));
    return recipe;
}

static void ValidateMetaRecipes(const CalibrationBoardMeta& meta) {
    const int grid_rows = meta.grid_rows;
    const int grid_cols = meta.grid_cols;
    if (grid_rows <= 0 || grid_cols <= 0) { throw InputError("grid size is invalid"); }
    const size_t expected = static_cast<size_t>(grid_rows) * static_cast<size_t>(grid_cols);
    if (meta.patch_recipe_idx.size() < expected) {
        throw InputError("patch_recipe_idx size mismatch");
    }
    if (meta.patch_recipes.size() < expected) {
        throw InputError("patch_recipes size mismatch");
    }
    const size_t layers = static_cast<size_t>(meta.config.recipe.color_layers);
    for (size_t i = 0; i < expected; ++i) {
        const auto& recipe = meta.patch_recipes[i];
        if (recipe.size() != layers) {
            throw InputError("patch_recipes layer size mismatch");
        }
    }
}

static void ApplyHoleMask(std::vector<uint8_t>& mask, int width, int height, float center_x,
                          float center_y, float diameter) {
    if (width <= 0 || height <= 0) { return; }
    if (diameter <= 0.0f) { return; }
    if (mask.size() < static_cast<size_t>(width) * static_cast<size_t>(height)) {
        throw InputError("mask size mismatch");
    }
    const float radius = diameter * 0.5f;
    const float r2     = radius * radius;

    int min_x = static_cast<int>(std::floor(center_x - radius));
    int max_x = static_cast<int>(std::ceil(center_x + radius));
    int min_y = static_cast<int>(std::floor(center_y - radius));
    int max_y = static_cast<int>(std::ceil(center_y + radius));

    min_x = std::max(min_x, 0);
    min_y = std::max(min_y, 0);
    max_x = std::min(max_x, width - 1);
    max_y = std::min(max_y, height - 1);

    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            const float dx = (static_cast<float>(x) + 0.5f) - center_x;
            const float dy = (static_cast<float>(y) + 0.5f) - center_y;
            if (dx * dx + dy * dy <= r2) {
                const size_t idx =
                    static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
                mask[idx] = 0;
            }
        }
    }
}


static int ResolveColorRegionScale(const CalibrationBoardMeta& meta, int width, int height) {
    if (width <= 0 || height <= 0) { throw InputError("Color region size is invalid"); }
    const int grid_rows = meta.grid_rows;
    const int grid_cols = meta.grid_cols;
    if (grid_rows <= 0 || grid_cols <= 0) { throw InputError("Grid size is invalid"); }

    const int tile_factor = meta.config.layout.tile_factor;
    const int gap_factor  = meta.config.layout.gap_factor;
    if (tile_factor <= 0) { throw InputError("tile_factor is invalid"); }
    if (gap_factor < 0) { throw InputError("gap_factor is invalid"); }

    const int base_w = grid_cols * tile_factor + (grid_cols - 1) * gap_factor;
    const int base_h = grid_rows * tile_factor + (grid_rows - 1) * gap_factor;
    if (base_w <= 0 || base_h <= 0) {
        throw InputError("Color region base size is invalid");
    }

    int scale = meta.config.layout.resolution_scale;
    if (scale <= 0) { scale = 1; }

    auto near = [](int a, int b) { return std::abs(a - b) <= 1; };
    if (near(base_w * scale, width) && near(base_h * scale, height)) { return scale; }

    const double scale_x = static_cast<double>(width) / static_cast<double>(base_w);
    const double scale_y = static_cast<double>(height) / static_cast<double>(base_h);
    const int inferred_x = static_cast<int>(std::lround(scale_x));
    const int inferred_y = static_cast<int>(std::lround(scale_y));
    if (inferred_x <= 0 || inferred_y <= 0 || inferred_x != inferred_y) {
        throw InputError("Color region size does not match meta layout");
    }
    if (!near(base_w * inferred_x, width) || !near(base_h * inferred_y, height)) {
        throw InputError("Color region size does not match meta layout");
    }
    return inferred_x;
}

static ColorDB BuildColorDBFromColorRegion(const cv::Mat& input, const CalibrationBoardMeta& meta) {
    if (input.empty()) { throw InputError("Color region image is empty"); }
    const cv::Mat bgr = detail::EnsureBgr(input);
    if (bgr.empty()) { throw InputError("Failed to normalize color region image"); }

    const int grid_rows = meta.grid_rows;
    const int grid_cols = meta.grid_cols;
    if (grid_rows <= 0 || grid_cols <= 0) { throw InputError("Grid size is invalid"); }

    const int scale = ResolveColorRegionScale(meta, bgr.cols, bgr.rows);
    const int tile  = meta.config.layout.tile_factor * scale;
    const int gap   = meta.config.layout.gap_factor * scale;

    const size_t expected_patches = static_cast<size_t>(grid_rows) * static_cast<size_t>(grid_cols);
    if (meta.patch_recipe_idx.size() < expected_patches) {
        throw InputError("patch_recipe_idx size mismatch");
    }

    if (meta.patch_recipes.size() < expected_patches) {
        throw InputError("patch_recipes size mismatch");
    }

    size_t recipe_count = 0;
    for (size_t i = 0; i < expected_patches; ++i) {
        const uint16_t idx = meta.patch_recipe_idx[i];
        if (idx == kInvalidRecipeIdx) { continue; }
        recipe_count = std::max(recipe_count, static_cast<size_t>(idx) + 1);
    }
    if (recipe_count == 0) { throw InputError("Recipe count is zero"); }

    const cv::Mat lab = detail::BgrToLab(bgr);
    if (lab.empty() || lab.type() != CV_32FC3) {
        throw InputError("Failed to convert color region to Lab");
    }

    std::vector<Vec3f> sum(recipe_count, Vec3f());
    std::vector<int> counts(recipe_count, 0);
    std::vector<std::vector<uint8_t>> recipes_by_idx(recipe_count);

    for (int r = 0; r < grid_rows; ++r) {
        for (int c = 0; c < grid_cols; ++c) {
            const size_t patch_idx =
                static_cast<size_t>(r) * static_cast<size_t>(grid_cols) + static_cast<size_t>(c);
            const uint16_t recipe_idx = meta.patch_recipe_idx[patch_idx];
            if (recipe_idx == kInvalidRecipeIdx) { continue; }
            if (recipe_idx >= recipe_count) { continue; }

            const auto& patch_recipe = meta.patch_recipes[patch_idx];
            if (patch_recipe.size() != meta.config.recipe.color_layers) {
                throw InputError("patch_recipes layer size mismatch");
            }
            auto& stored_recipe = recipes_by_idx[recipe_idx];
            if (!stored_recipe.empty() && stored_recipe != patch_recipe) {
                throw InputError("patch_recipes mismatch for recipe_idx");
            }
            if (stored_recipe.empty()) { stored_recipe = patch_recipe; }

            const int x0 = c * (tile + gap);
            const int y0 = r * (tile + gap);
            const int x1 = x0 + tile;
            const int y1 = y0 + tile;
            if (x0 < 0 || y0 < 0 || x1 > lab.cols || y1 > lab.rows) {
                throw InputError("Patch ROI is out of bounds");
            }

            int inset = std::max(1, tile / 10);
            if (tile <= inset * 2) { inset = 0; }
            int sx0 = x0 + inset;
            int sy0 = y0 + inset;
            int sx1 = x1 - inset;
            int sy1 = y1 - inset;
            if (sx1 <= sx0 || sy1 <= sy0) {
                sx0 = x0;
                sy0 = y0;
                sx1 = x1;
                sy1 = y1;
            }

            cv::Rect roi(sx0, sy0, sx1 - sx0, sy1 - sy0);
            cv::Scalar mean = cv::mean(lab(roi));

            sum[recipe_idx] += Vec3f(static_cast<float>(mean[0]), static_cast<float>(mean[1]),
                                     static_cast<float>(mean[2]));
            counts[recipe_idx] += 1;
        }
    }

    ColorDB db;
    db.name             = meta.name.empty() ? "ColorDB" : meta.name;
    db.max_color_layers = meta.config.recipe.color_layers;
    db.base_layers      = meta.config.base_layers;
    db.base_channel_idx = meta.config.base_channel_idx;
    db.layer_height_mm  = meta.config.layer_height_mm;
    db.line_width_mm    = meta.config.layout.line_width_mm;
    db.layer_order      = meta.config.recipe.layer_order;
    db.palette          = meta.config.palette;

    db.entries.reserve(recipe_count);
    for (size_t idx = 0; idx < recipe_count; ++idx) {
        if (counts[idx] <= 0) {
            throw InputError("Missing patch for recipe index " + std::to_string(idx));
        }
        Vec3f avg = sum[idx] / static_cast<float>(counts[idx]);
        Entry entry;
        entry.lab = Lab(avg.x, avg.y, avg.z);
        if (recipes_by_idx[idx].empty()) {
            throw InputError("Missing patch recipe for index " + std::to_string(idx));
        }
        entry.recipe = recipes_by_idx[idx];
        db.entries.push_back(entry);
    }
    return db;
}

} // namespace

CalibrationBoardMeta BuildCalibrationBoardMeta(const CalibrationBoardConfig& cfg) {
    ValidateConfig(cfg);

    CalibrationBoardMeta meta;
    meta.config = cfg;
    meta.name   = "CalibrationBoard_" + std::to_string(cfg.recipe.num_channels) + "ch";

    const size_t recipe_count = cfg.recipe.NumRecipes();
    const int grid_size       = ComputeGridSize(recipe_count);
    meta.grid_rows            = grid_size;
    meta.grid_cols            = grid_size;

    const size_t total = static_cast<size_t>(grid_size) * static_cast<size_t>(grid_size);
    meta.patch_recipe_idx.assign(total, kInvalidRecipeIdx);
    for (size_t i = 0; i < total; ++i) {
        if (i < recipe_count) { meta.patch_recipe_idx[i] = static_cast<uint16_t>(i); }
    }
    meta.patch_recipes.assign(total, {});
    const std::vector<uint8_t> background = BuildBackgroundRecipe(cfg);
    for (size_t i = 0; i < total; ++i) {
        if (meta.patch_recipe_idx[i] == kInvalidRecipeIdx) {
            meta.patch_recipes[i] = background;
            continue;
        }
        meta.patch_recipes[i] = cfg.recipe.RecipeAt(meta.patch_recipe_idx[i]);
    }
    return meta;
}

static json MetaToJson(const CalibrationBoardMeta& meta) {
    if (!meta.patch_recipes.empty() && meta.patch_recipes.size() != meta.patch_recipe_idx.size()) {
        throw InputError("patch_recipes size mismatch");
    }
    json j;
    j["name"]      = meta.name;
    j["grid_rows"] = meta.grid_rows;
    j["grid_cols"] = meta.grid_cols;

    json cfg;
    cfg["base_layers"]      = meta.config.base_layers;
    cfg["base_channel_idx"] = meta.config.base_channel_idx;
    cfg["layer_height_mm"]  = meta.config.layer_height_mm;

    json recipe;
    recipe["num_channels"] = meta.config.recipe.num_channels;
    recipe["color_layers"] = meta.config.recipe.color_layers;
    recipe["layer_order"]  = ToLayerOrderString(meta.config.recipe.layer_order);
    cfg["recipe"]          = recipe;

    json layout;
    layout["line_width_mm"]    = meta.config.layout.line_width_mm;
    layout["resolution_scale"] = meta.config.layout.resolution_scale;
    layout["tile_factor"]      = meta.config.layout.tile_factor;
    layout["gap_factor"]       = meta.config.layout.gap_factor;
    layout["margin_factor"]    = meta.config.layout.margin_factor;

    json fid;
    fid["offset_factor"] = meta.config.layout.fiducial.offset_factor;
    fid["main_d_factor"] = meta.config.layout.fiducial.main_d_factor;
    fid["tag_d_factor"]  = meta.config.layout.fiducial.tag_d_factor;
    fid["tag_dx_factor"] = meta.config.layout.fiducial.tag_dx_factor;
    fid["tag_dy_factor"] = meta.config.layout.fiducial.tag_dy_factor;
    layout["fiducial"]   = fid;

    cfg["layout"] = layout;

    json palette = json::array();
    for (const auto& channel : meta.config.palette) {
        json c;
        c["color"]    = channel.color;
        c["material"] = channel.material;
        palette.push_back(c);
    }
    cfg["palette"] = palette;

    j["config"] = cfg;

    json patches = json::array();
    for (uint16_t v : meta.patch_recipe_idx) { patches.push_back(static_cast<int>(v)); }
    j["patch_recipe_idx"] = patches;

    json patch_recipes_json = json::array();
    for (const auto& r : meta.patch_recipes) {
        json recipe_json = json::array();
        for (uint8_t v : r) { recipe_json.push_back(static_cast<int>(v)); }
        patch_recipes_json.push_back(recipe_json);
    }
    j["patch_recipes"] = patch_recipes_json;

    return j;
}

void CalibrationBoardMeta::SaveToJson(const std::string& path) const {
    json j = MetaToJson(*this);
    std::ofstream out(path);
    if (!out.is_open()) { throw IOError("Failed to open file: " + path); }
    out << j.dump(4);
    if (!out.good()) { throw IOError("Failed to write json: " + path); }
}

std::string CalibrationBoardMeta::ToJsonString() const { return MetaToJson(*this).dump(4); }

static CalibrationBoardMeta MetaFromJson(const json& j) {
    CalibrationBoardMeta meta;
    meta.name      = j.value("name", meta.name);
    meta.grid_rows = j.value("grid_rows", meta.grid_rows);
    meta.grid_cols = j.value("grid_cols", meta.grid_cols);

    if (j.contains("config")) {
        const auto& c                = j.at("config");
        meta.config.base_layers      = c.value("base_layers", meta.config.base_layers);
        meta.config.base_channel_idx = c.value("base_channel_idx", meta.config.base_channel_idx);
        meta.config.layer_height_mm  = c.value("layer_height_mm", meta.config.layer_height_mm);

        if (c.contains("recipe")) {
            const auto& r = c.at("recipe");
            meta.config.recipe.num_channels =
                r.value("num_channels", meta.config.recipe.num_channels);
            meta.config.recipe.color_layers =
                r.value("color_layers", meta.config.recipe.color_layers);
            if (r.contains("layer_order")) {
                meta.config.recipe.layer_order = detail::ParseLayerOrder(r.at("layer_order"));
            }
        }

        if (c.contains("layout")) {
            const auto& l = c.at("layout");
            meta.config.layout.line_width_mm =
                l.value("line_width_mm", meta.config.layout.line_width_mm);
            meta.config.layout.resolution_scale =
                l.value("resolution_scale", meta.config.layout.resolution_scale);
            meta.config.layout.tile_factor = l.value("tile_factor", meta.config.layout.tile_factor);
            meta.config.layout.gap_factor  = l.value("gap_factor", meta.config.layout.gap_factor);
            meta.config.layout.margin_factor =
                l.value("margin_factor", meta.config.layout.margin_factor);

            if (l.contains("fiducial")) {
                const auto& f = l.at("fiducial");
                meta.config.layout.fiducial.offset_factor =
                    f.value("offset_factor", meta.config.layout.fiducial.offset_factor);
                meta.config.layout.fiducial.main_d_factor =
                    f.value("main_d_factor", meta.config.layout.fiducial.main_d_factor);
                meta.config.layout.fiducial.tag_d_factor =
                    f.value("tag_d_factor", meta.config.layout.fiducial.tag_d_factor);
                meta.config.layout.fiducial.tag_dx_factor =
                    f.value("tag_dx_factor", meta.config.layout.fiducial.tag_dx_factor);
                meta.config.layout.fiducial.tag_dy_factor =
                    f.value("tag_dy_factor", meta.config.layout.fiducial.tag_dy_factor);
            }
        }

        meta.config.palette.clear();
        if (c.contains("palette")) {
            const auto& p = c.at("palette");
            if (!p.is_array()) { throw FormatError("palette must be an array"); }
            for (const auto& item : p) {
                Channel ch;
                ch.color    = item.value("color", ch.color);
                ch.material = item.value("material", ch.material);
                meta.config.palette.push_back(ch);
            }
        }
    }

    meta.patch_recipe_idx.clear();
    if (j.contains("patch_recipe_idx")) {
        const auto& p = j.at("patch_recipe_idx");
        if (!p.is_array()) { throw FormatError("patch_recipe_idx must be an array"); }
        meta.patch_recipe_idx.reserve(p.size());
        for (const auto& v : p) {
            int value = v.get<int>();
            if (value < 0 || value > static_cast<int>(kInvalidRecipeIdx)) {
                throw FormatError("patch_recipe_idx value out of range");
            }
            meta.patch_recipe_idx.push_back(static_cast<uint16_t>(value));
        }
    }

    if (!j.contains("patch_recipes")) { throw FormatError("patch_recipes missing in meta"); }
    const auto& pr = j.at("patch_recipes");
    if (!pr.is_array()) { throw FormatError("patch_recipes must be an array"); }
    meta.patch_recipes.reserve(pr.size());
    for (const auto& item : pr) {
        if (!item.is_array()) { throw FormatError("patch_recipes item must be an array"); }
        std::vector<uint8_t> recipe;
        recipe.reserve(item.size());
        for (const auto& v : item) {
            int value = v.get<int>();
            if (value < 0 || value > 255) {
                throw FormatError("patch_recipes value out of range");
            }
            recipe.push_back(static_cast<uint8_t>(value));
        }
        meta.patch_recipes.push_back(recipe);
    }

    if (meta.grid_rows > 0 && meta.grid_cols > 0) {
        const size_t expected =
            static_cast<size_t>(meta.grid_rows) * static_cast<size_t>(meta.grid_cols);
        if (meta.patch_recipe_idx.size() < expected) {
            throw FormatError("patch_recipe_idx size mismatch");
        }
        if (meta.patch_recipes.size() < expected) {
            throw FormatError("patch_recipes size mismatch");
        }
    }

    return meta;
}

CalibrationBoardMeta CalibrationBoardMeta::LoadFromJson(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) { throw IOError("Failed to open file: " + path); }
    json j;
    in >> j;
    return MetaFromJson(j);
}

CalibrationBoardMeta CalibrationBoardMeta::FromJsonString(const std::string& json_str) {
    json j = json::parse(json_str);
    return MetaFromJson(j);
}

struct BoardBuildResult {
    ModelIR model;
    BuildMeshConfig mesh_cfg;
};

static BoardBuildResult BuildBoardModel(const CalibrationBoardMeta& meta) {
    ValidateConfig(meta.config);
    ValidateMetaRecipes(meta);

    const int grid_rows = meta.grid_rows;
    const int grid_cols = meta.grid_cols;

    const int scale  = meta.config.layout.resolution_scale;
    const int tile   = meta.config.layout.tile_factor * scale;
    const int gap    = meta.config.layout.gap_factor * scale;
    const int margin = meta.config.layout.margin_factor * scale;

    const int color_width  = grid_cols * tile + (grid_cols - 1) * gap;
    const int color_height = grid_rows * tile + (grid_rows - 1) * gap;
    const int width        = color_width + 2 * margin;
    const int height       = color_height + 2 * margin;
    if (width <= 0 || height <= 0) { throw InputError("board size is invalid"); }

    RecipeMap map;
    map.name         = meta.name;
    map.width        = width;
    map.height       = height;
    map.color_layers = meta.config.recipe.color_layers;
    map.num_channels = meta.config.recipe.num_channels;
    map.layer_order  = meta.config.recipe.layer_order;

    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t layers      = static_cast<size_t>(map.color_layers);
    map.recipes.assign(pixel_count * layers, 0);
    map.mask.assign(pixel_count, 255);
    std::vector<uint8_t> base_only_mask(pixel_count, 1);

    const std::vector<uint8_t> background = BuildBackgroundRecipe(meta.config);
    {
        const size_t row_bytes = static_cast<size_t>(width) * layers;
        for (size_t c = 0; c < static_cast<size_t>(width); ++c) {
            std::memcpy(&map.recipes[c * layers], background.data(), layers);
        }
        for (size_t r = 1; r < static_cast<size_t>(height); ++r) {
            std::memcpy(&map.recipes[r * row_bytes], &map.recipes[0], row_bytes);
        }
    }

    for (int r = 0; r < grid_rows; ++r) {
        for (int c = 0; c < grid_cols; ++c) {
            const size_t patch_idx =
                static_cast<size_t>(r) * static_cast<size_t>(grid_cols) + static_cast<size_t>(c);
            if (patch_idx >= meta.patch_recipe_idx.size()) { continue; }
            const uint16_t recipe_idx = meta.patch_recipe_idx[patch_idx];

            const bool is_patch = (recipe_idx != kInvalidRecipeIdx);
            const uint8_t* recipe_data =
                is_patch ? meta.patch_recipes[patch_idx].data() : background.data();
            const size_t recipe_len =
                is_patch ? meta.patch_recipes[patch_idx].size() : background.size();
            if (recipe_len != layers) {
                throw InputError("patch_recipes layer size mismatch");
            }

            const int x0 = margin + c * (tile + gap);
            const int y0 = margin + r * (tile + gap);
            const int x1 = std::min(x0 + tile, width);
            const int y1 = std::min(y0 + tile, height);
            if (x0 < 0 || y0 < 0) { continue; }

            const size_t tile_row_bytes = static_cast<size_t>(x1 - x0) * layers;
            for (int y = y0; y < y1; ++y) {
                const size_t row_start =
                    static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x0);
                std::memcpy(&map.recipes[row_start * layers], recipe_data, layers);
                for (size_t px = 1; px < static_cast<size_t>(x1 - x0); ++px) {
                    std::memcpy(&map.recipes[(row_start + px) * layers], recipe_data, layers);
                }
                if (is_patch) {
                    std::memset(&base_only_mask[row_start], 0,
                                static_cast<size_t>(x1 - x0));
                }
            }
        }
    }

    const float offset = static_cast<float>(meta.config.layout.fiducial.offset_factor * scale);
    const float main_d = static_cast<float>(meta.config.layout.fiducial.main_d_factor * scale);
    const float tag_d  = static_cast<float>(meta.config.layout.fiducial.tag_d_factor * scale);

    ApplyHoleMask(map.mask, width, height, offset, offset, main_d);
    ApplyHoleMask(map.mask, width, height, static_cast<float>(width) - offset, offset, main_d);
    ApplyHoleMask(map.mask, width, height, offset, static_cast<float>(height) - offset, main_d);
    ApplyHoleMask(map.mask, width, height, static_cast<float>(width) - offset,
                  static_cast<float>(height) - offset, main_d);

    const float tag_dx = static_cast<float>(meta.config.layout.fiducial.tag_dx_factor * scale);
    const float tag_dy = static_cast<float>(meta.config.layout.fiducial.tag_dy_factor * scale);
    ApplyHoleMask(map.mask, width, height, offset + tag_dx, offset + tag_dy, tag_d);
    for (size_t i = 0; i < map.mask.size() && i < base_only_mask.size(); ++i) {
        if (map.mask[i] == 0) { base_only_mask[i] = 0; }
    }

    ColorDB db;
    db.name             = meta.name;
    db.max_color_layers = meta.config.recipe.color_layers;
    db.base_layers      = meta.config.base_layers;
    db.base_channel_idx = meta.config.base_channel_idx;
    db.layer_height_mm  = meta.config.layer_height_mm;
    db.line_width_mm    = meta.config.layout.line_width_mm;
    db.layer_order      = meta.config.recipe.layer_order;
    db.palette          = meta.config.palette;

    BuildModelIRConfig build_cfg;
    build_cfg.flip_y         = true;
    build_cfg.base_layers    = meta.config.base_layers;
    build_cfg.base_only_mask = base_only_mask;

    ModelIR model = ModelIR::Build(map, db, build_cfg);

    BuildMeshConfig mesh_cfg;
    mesh_cfg.layer_height_mm = meta.config.layer_height_mm;
    mesh_cfg.pixel_mm        = meta.config.layout.line_width_mm / static_cast<float>(scale);

    return BoardBuildResult{std::move(model), mesh_cfg};
}

void GenCalibrationBoard(const CalibrationBoardConfig& cfg, const std::string& board_path,
                         const std::string& meta_path) {
    CalibrationBoardMeta meta = BuildCalibrationBoardMeta(cfg);
    if (board_path.empty()) { throw InputError("board_path is empty"); }
    if (meta_path.empty()) { throw InputError("meta_path is empty"); }
    meta.SaveToJson(meta_path);
    auto result = BuildBoardModel(meta);
    Export3mf(board_path, result.model, result.mesh_cfg);
}

void GenCalibrationBoardFromMeta(const CalibrationBoardMeta& meta, const std::string& board_path,
                                 const std::string& meta_path) {
    if (board_path.empty()) { throw InputError("board_path is empty"); }
    if (meta_path.empty()) { throw InputError("meta_path is empty"); }
    meta.SaveToJson(meta_path);
    auto result = BuildBoardModel(meta);
    Export3mf(board_path, result.model, result.mesh_cfg);
}

CalibrationBoardResult GenCalibrationBoardToBuffer(const CalibrationBoardConfig& cfg) {
    CalibrationBoardMeta meta = BuildCalibrationBoardMeta(cfg);
    auto result               = BuildBoardModel(meta);
    CalibrationBoardResult out;
    out.meta      = std::move(meta);
    out.model_3mf = Export3mfToBuffer(result.model, result.mesh_cfg);
    return out;
}

CalibrationBoardMeshes GenCalibrationBoardMeshes(const CalibrationBoardConfig& cfg) {
    CalibrationBoardMeta meta = BuildCalibrationBoardMeta(cfg);
    auto build                = BuildBoardModel(meta);

    const auto n = static_cast<int>(build.model.voxel_grids.size());
    std::vector<Mesh> meshes(static_cast<std::size_t>(n));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        const VoxelGrid& grid = build.model.voxel_grids[static_cast<std::size_t>(i)];
        if (grid.width <= 0 || grid.height <= 0 || grid.num_layers <= 0) { continue; }
        if (grid.ooc.empty()) { continue; }
        meshes[static_cast<std::size_t>(i)] = Mesh::Build(grid, build.mesh_cfg);
    }

    std::size_t total_verts = 0, total_tris = 0;
    for (const auto& m : meshes) {
        total_verts += m.vertices.size();
        total_tris += m.indices.size();
    }
    spdlog::info("Mesh::Build: {} grids, total vertices={}, triangles={}", n, total_verts, total_tris);

    CalibrationBoardMeshes out;
    out.meta             = std::move(meta);
    out.meshes           = std::move(meshes);
    out.mesh_cfg         = build.mesh_cfg;
    out.base_channel_idx = build.model.base_channel_idx;
    out.base_layers      = build.model.base_layers;
    return out;
}

CalibrationBoardResult BuildResultFromMeshes(const CalibrationBoardMeshes& cached,
                                             const std::vector<Channel>& palette) {
    CalibrationBoardResult out;
    out.meta                 = cached.meta;
    out.meta.config.palette  = palette;
    out.model_3mf = Export3mfFromMeshes(cached.meshes, palette,
                                         cached.base_channel_idx, cached.base_layers);
    return out;
}

ColorDB GenColorDBFromImage(const std::string& image_path, const CalibrationBoardMeta& meta) {
    cv::Mat color_region = LocateCalibrationColorRegion(image_path, meta);
    return BuildColorDBFromColorRegion(color_region, meta);
}

ColorDB GenColorDBFromImage(const std::string& image_path, const std::string& json_path) {
    CalibrationBoardMeta meta = CalibrationBoardMeta::LoadFromJson(json_path);
    return GenColorDBFromImage(image_path, meta);
}

ColorDB GenColorDBFromBuffer(const std::vector<uint8_t>& image_buffer,
                             const CalibrationBoardMeta& meta) {
    if (image_buffer.empty()) {
        throw InputError("Uploaded image data is empty");
    }
    cv::Mat input = cv::imdecode(image_buffer, cv::IMREAD_UNCHANGED);
    if (input.empty()) {
        throw IOError("Failed to decode uploaded image; ensure the file is a valid image format (JPEG/PNG, etc.)");
    }
    cv::Mat color_region = LocateCalibrationColorRegion(input, meta);
    return BuildColorDBFromColorRegion(color_region, meta);
}

} // namespace ChromaPrint3D