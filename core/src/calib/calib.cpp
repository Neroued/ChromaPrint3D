#include "calib.h"
#include "geo.h"
#include "match.h"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ChromaPrint3D {

cv::Mat LocateCalibrationColorRegion(const std::string& image_path,
                                     const CalibrationBoardMeta& meta);

namespace {
using nlohmann::json;

constexpr uint16_t kInvalidRecipeIdx = 0xFFFF;

static std::string ToLayerOrderString(LayerOrder order) {
    switch (order) {
    case LayerOrder::Top2Bottom:
        return "Top2Bottom";
    case LayerOrder::Bottom2Top:
        return "Bottom2Top";
    }
    return "Top2Bottom";
}

static LayerOrder FromLayerOrderString(const std::string& order) {
    if (order == "Top2Bottom") { return LayerOrder::Top2Bottom; }
    if (order == "Bottom2Top") { return LayerOrder::Bottom2Top; }
    throw std::runtime_error("Invalid layer_order string: " + order);
}

static LayerOrder ParseLayerOrder(const json& value) {
    if (value.is_string()) { return FromLayerOrderString(value.get<std::string>()); }
    if (value.is_number_integer()) {
        int v = value.get<int>();
        if (v == 0) { return LayerOrder::Top2Bottom; }
        if (v == 1) { return LayerOrder::Bottom2Top; }
    }
    throw std::runtime_error("Invalid layer_order value");
}

static void ValidateConfig(const CalibrationBoardConfig& cfg) {
    if (!cfg.recipe.IsSupported()) {
        throw std::runtime_error("Calibration recipe is not supported");
    }
    if (!cfg.palette.empty() && static_cast<int>(cfg.palette.size()) != cfg.recipe.num_channels) {
        throw std::runtime_error("Calibration palette size does not match num_channels");
    }
    if (cfg.base_layers < 0) { throw std::runtime_error("base_layers is invalid"); }
    if (cfg.base_layers > 0 &&
        (cfg.base_channel_idx < 0 || cfg.base_channel_idx >= cfg.recipe.num_channels)) {
        throw std::runtime_error("base_channel_idx is out of range");
    }
    if (cfg.layout.line_width_mm <= 0.0f) {
        throw std::runtime_error("line_width_mm must be positive");
    }
    if (cfg.layout.resolution_scale <= 0) {
        throw std::runtime_error("resolution_scale must be positive");
    }
    if (cfg.layout.tile_factor <= 0) { throw std::runtime_error("tile_factor must be positive"); }
    if (cfg.layout.gap_factor < 0) { throw std::runtime_error("gap_factor is invalid"); }
    if (cfg.layout.margin_factor < 0) { throw std::runtime_error("margin_factor is invalid"); }
    if (cfg.layout.fiducial.main_d_factor <= 0) {
        throw std::runtime_error("fiducial main_d_factor must be positive");
    }
}

static int ComputeGridSize(std::size_t recipe_count) {
    if (recipe_count == 0) { throw std::runtime_error("recipe_count is zero"); }
    const double side = std::ceil(std::sqrt(static_cast<double>(recipe_count)));
    const int grid    = static_cast<int>(side);
    if (grid <= 0) { throw std::runtime_error("grid size is invalid"); }
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

static void ApplyHoleMask(std::vector<uint8_t>& mask, int width, int height, float center_x,
                          float center_y, float diameter) {
    if (width <= 0 || height <= 0) { return; }
    if (diameter <= 0.0f) { return; }
    if (mask.size() < static_cast<size_t>(width) * static_cast<size_t>(height)) {
        throw std::runtime_error("mask size mismatch");
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

static cv::Mat EnsureBgr(const cv::Mat& src) {
    if (src.empty()) { return cv::Mat(); }
    if (src.channels() == 3) { return src; }
    if (src.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    if (src.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    throw std::runtime_error("Unsupported image channel count");
}

static cv::Mat BgrToLab(const cv::Mat& bgr) {
    if (bgr.empty()) { return cv::Mat(); }
    cv::Mat bgr_float;
    bgr.convertTo(bgr_float, CV_32F, 1.0 / 255.0);
    cv::Mat lab;
    cv::cvtColor(bgr_float, lab, cv::COLOR_BGR2Lab);
    return lab;
}

static int ResolveColorRegionScale(const CalibrationBoardMeta& meta, int width, int height) {
    if (width <= 0 || height <= 0) { throw std::runtime_error("Color region size is invalid"); }
    const int grid_rows = meta.grid_rows;
    const int grid_cols = meta.grid_cols;
    if (grid_rows <= 0 || grid_cols <= 0) { throw std::runtime_error("Grid size is invalid"); }

    const int tile_factor = meta.config.layout.tile_factor;
    const int gap_factor  = meta.config.layout.gap_factor;
    if (tile_factor <= 0) { throw std::runtime_error("tile_factor is invalid"); }
    if (gap_factor < 0) { throw std::runtime_error("gap_factor is invalid"); }

    const int base_w = grid_cols * tile_factor + (grid_cols - 1) * gap_factor;
    const int base_h = grid_rows * tile_factor + (grid_rows - 1) * gap_factor;
    if (base_w <= 0 || base_h <= 0) { throw std::runtime_error("Color region base size is invalid"); }

    int scale = meta.config.layout.resolution_scale;
    if (scale <= 0) { scale = 1; }

    auto near = [](int a, int b) { return std::abs(a - b) <= 1; };
    if (near(base_w * scale, width) && near(base_h * scale, height)) { return scale; }

    const double scale_x = static_cast<double>(width) / static_cast<double>(base_w);
    const double scale_y = static_cast<double>(height) / static_cast<double>(base_h);
    const int inferred_x = static_cast<int>(std::lround(scale_x));
    const int inferred_y = static_cast<int>(std::lround(scale_y));
    if (inferred_x <= 0 || inferred_y <= 0 || inferred_x != inferred_y) {
        throw std::runtime_error("Color region size does not match meta layout");
    }
    if (!near(base_w * inferred_x, width) || !near(base_h * inferred_y, height)) {
        throw std::runtime_error("Color region size does not match meta layout");
    }
    return inferred_x;
}

static ColorDB BuildColorDBFromColorRegion(const cv::Mat& input,
                                           const CalibrationBoardMeta& meta) {
    if (input.empty()) { throw std::runtime_error("Color region image is empty"); }
    const cv::Mat bgr = EnsureBgr(input);
    if (bgr.empty()) { throw std::runtime_error("Failed to normalize color region image"); }

    const int grid_rows = meta.grid_rows;
    const int grid_cols = meta.grid_cols;
    if (grid_rows <= 0 || grid_cols <= 0) { throw std::runtime_error("Grid size is invalid"); }

    const int scale = ResolveColorRegionScale(meta, bgr.cols, bgr.rows);
    const int tile  = meta.config.layout.tile_factor * scale;
    const int gap   = meta.config.layout.gap_factor * scale;

    const size_t expected_patches =
        static_cast<size_t>(grid_rows) * static_cast<size_t>(grid_cols);
    if (meta.patch_recipe_idx.size() < expected_patches) {
        throw std::runtime_error("patch_recipe_idx size mismatch");
    }

    const size_t recipe_count = meta.config.recipe.NumRecipes();
    if (recipe_count == 0) { throw std::runtime_error("Recipe count is zero"); }

    const cv::Mat lab = BgrToLab(bgr);
    if (lab.empty() || lab.type() != CV_32FC3) {
        throw std::runtime_error("Failed to convert color region to Lab");
    }

    std::vector<Vec3f> sum(recipe_count, Vec3f());
    std::vector<int> counts(recipe_count, 0);

    for (int r = 0; r < grid_rows; ++r) {
        for (int c = 0; c < grid_cols; ++c) {
            const size_t patch_idx =
                static_cast<size_t>(r) * static_cast<size_t>(grid_cols) + static_cast<size_t>(c);
            const uint16_t recipe_idx = meta.patch_recipe_idx[patch_idx];
            if (recipe_idx == kInvalidRecipeIdx) { continue; }
            if (recipe_idx >= recipe_count) { continue; }

            const int x0 = c * (tile + gap);
            const int y0 = r * (tile + gap);
            const int x1 = x0 + tile;
            const int y1 = y0 + tile;
            if (x0 < 0 || y0 < 0 || x1 > lab.cols || y1 > lab.rows) {
                throw std::runtime_error("Patch ROI is out of bounds");
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

            sum[recipe_idx] += Vec3f(static_cast<float>(mean[0]),
                                     static_cast<float>(mean[1]),
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
            throw std::runtime_error("Missing patch for recipe index " + std::to_string(idx));
        }
        Vec3f avg = sum[idx] / static_cast<float>(counts[idx]);
        Entry entry;
        entry.lab = Lab(avg.x, avg.y, avg.z);
        entry.recipe = meta.config.recipe.RecipeAt(idx);
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
    return meta;
}

void CalibrationBoardMeta::SaveToJson(const std::string& path) const {
    json j;
    j["name"]      = name;
    j["grid_rows"] = grid_rows;
    j["grid_cols"] = grid_cols;

    json cfg;
    cfg["base_layers"]      = config.base_layers;
    cfg["base_channel_idx"] = config.base_channel_idx;
    cfg["layer_height_mm"]  = config.layer_height_mm;

    json recipe;
    recipe["num_channels"] = config.recipe.num_channels;
    recipe["color_layers"] = config.recipe.color_layers;
    recipe["layer_order"]  = ToLayerOrderString(config.recipe.layer_order);
    cfg["recipe"]          = recipe;

    json layout;
    layout["line_width_mm"]    = config.layout.line_width_mm;
    layout["resolution_scale"] = config.layout.resolution_scale;
    layout["tile_factor"]      = config.layout.tile_factor;
    layout["gap_factor"]       = config.layout.gap_factor;
    layout["margin_factor"]    = config.layout.margin_factor;

    json fid;
    fid["offset_factor"] = config.layout.fiducial.offset_factor;
    fid["main_d_factor"] = config.layout.fiducial.main_d_factor;
    fid["tag_d_factor"]  = config.layout.fiducial.tag_d_factor;
    fid["tag_dx_factor"] = config.layout.fiducial.tag_dx_factor;
    fid["tag_dy_factor"] = config.layout.fiducial.tag_dy_factor;
    layout["fiducial"]   = fid;

    cfg["layout"] = layout;

    json palette = json::array();
    for (const auto& channel : config.palette) {
        json c;
        c["color"]    = channel.color;
        c["material"] = channel.material;
        palette.push_back(c);
    }
    cfg["palette"] = palette;

    j["config"] = cfg;

    json patches = json::array();
    for (uint16_t v : patch_recipe_idx) { patches.push_back(static_cast<int>(v)); }
    j["patch_recipe_idx"] = patches;

    std::ofstream out(path);
    if (!out.is_open()) { throw std::runtime_error("Failed to open file: " + path); }
    out << j.dump(4);
    if (!out.good()) { throw std::runtime_error("Failed to write json: " + path); }
}

CalibrationBoardMeta CalibrationBoardMeta::LoadFromJson(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) { throw std::runtime_error("Failed to open file: " + path); }

    json j;
    in >> j;

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
                meta.config.recipe.layer_order = ParseLayerOrder(r.at("layer_order"));
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
            if (!p.is_array()) { throw std::runtime_error("palette must be an array"); }
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
        if (!p.is_array()) { throw std::runtime_error("patch_recipe_idx must be an array"); }
        meta.patch_recipe_idx.reserve(p.size());
        for (const auto& v : p) {
            int value = v.get<int>();
            if (value < 0 || value > static_cast<int>(kInvalidRecipeIdx)) {
                throw std::runtime_error("patch_recipe_idx value out of range");
            }
            meta.patch_recipe_idx.push_back(static_cast<uint16_t>(value));
        }
    }

    if (meta.grid_rows > 0 && meta.grid_cols > 0 && !meta.patch_recipe_idx.empty()) {
        const size_t expected =
            static_cast<size_t>(meta.grid_rows) * static_cast<size_t>(meta.grid_cols);
        if (meta.patch_recipe_idx.size() < expected) {
            throw std::runtime_error("patch_recipe_idx size mismatch");
        }
    }

    return meta;
}

void GenCalibrationBoard(const CalibrationBoardConfig& cfg, const std::string& board_path,
                         const std::string& meta_path) {
    if (board_path.empty()) { throw std::runtime_error("board_path is empty"); }
    if (meta_path.empty()) { throw std::runtime_error("meta_path is empty"); }

    ValidateConfig(cfg);
    CalibrationBoardMeta meta = BuildCalibrationBoardMeta(cfg);
    meta.SaveToJson(meta_path);

    const int grid_rows = meta.grid_rows;
    const int grid_cols = meta.grid_cols;
    if (grid_rows <= 0 || grid_cols <= 0) { throw std::runtime_error("grid size is invalid"); }

    const int scale  = cfg.layout.resolution_scale;
    const int tile   = cfg.layout.tile_factor * scale;
    const int gap    = cfg.layout.gap_factor * scale;
    const int margin = cfg.layout.margin_factor * scale;

    const int color_width  = grid_cols * tile + (grid_cols - 1) * gap;
    const int color_height = grid_rows * tile + (grid_rows - 1) * gap;
    const int width        = color_width + 2 * margin;
    const int height       = color_height + 2 * margin;
    if (width <= 0 || height <= 0) { throw std::runtime_error("board size is invalid"); }

    RecipeMap map;
    map.name         = meta.name;
    map.width        = width;
    map.height       = height;
    map.color_layers = cfg.recipe.color_layers;
    map.num_channels = cfg.recipe.num_channels;
    map.layer_order  = cfg.recipe.layer_order;

    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t layers      = static_cast<size_t>(map.color_layers);
    map.recipes.assign(pixel_count * layers, 0);
    map.mask.assign(pixel_count, 255);

    const std::vector<uint8_t> background = BuildBackgroundRecipe(cfg);
    for (size_t i = 0; i < pixel_count; ++i) {
        size_t offset = i * layers;
        for (size_t l = 0; l < layers; ++l) { map.recipes[offset + l] = background[l]; }
    }

    for (int r = 0; r < grid_rows; ++r) {
        for (int c = 0; c < grid_cols; ++c) {
            const size_t patch_idx =
                static_cast<size_t>(r) * static_cast<size_t>(grid_cols) + static_cast<size_t>(c);
            if (patch_idx >= meta.patch_recipe_idx.size()) { continue; }
            const uint16_t recipe_idx = meta.patch_recipe_idx[patch_idx];

            std::vector<uint8_t> recipe;
            if (recipe_idx == kInvalidRecipeIdx) {
                recipe = background;
            } else {
                recipe = cfg.recipe.RecipeAt(static_cast<size_t>(recipe_idx));
                if (recipe.size() < layers) { recipe.resize(layers, background.front()); }
            }

            const int x0 = margin + c * (tile + gap);
            const int y0 = margin + r * (tile + gap);
            const int x1 = x0 + tile;
            const int y1 = y0 + tile;

            for (int y = y0; y < y1; ++y) {
                if (y < 0 || y >= height) { continue; }
                for (int x = x0; x < x1; ++x) {
                    if (x < 0 || x >= width) { continue; }
                    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) +
                                       static_cast<size_t>(x);
                    size_t offset = idx * layers;
                    for (size_t l = 0; l < layers; ++l) { map.recipes[offset + l] = recipe[l]; }
                }
            }
        }
    }

    const float offset = static_cast<float>(cfg.layout.fiducial.offset_factor * scale);
    const float main_d = static_cast<float>(cfg.layout.fiducial.main_d_factor * scale);
    const float tag_d  = static_cast<float>(cfg.layout.fiducial.tag_d_factor * scale);

    ApplyHoleMask(map.mask, width, height, offset, offset, main_d);
    ApplyHoleMask(map.mask, width, height, static_cast<float>(width) - offset, offset, main_d);
    ApplyHoleMask(map.mask, width, height, offset, static_cast<float>(height) - offset, main_d);
    ApplyHoleMask(map.mask, width, height, static_cast<float>(width) - offset,
                  static_cast<float>(height) - offset, main_d);

    const float tag_dx = static_cast<float>(cfg.layout.fiducial.tag_dx_factor * scale);
    const float tag_dy = static_cast<float>(cfg.layout.fiducial.tag_dy_factor * scale);
    ApplyHoleMask(map.mask, width, height, offset + tag_dx, offset + tag_dy, tag_d);

    ColorDB db;
    db.name             = meta.name;
    db.max_color_layers = cfg.recipe.color_layers;
    db.base_layers      = cfg.base_layers;
    db.base_channel_idx = cfg.base_channel_idx;
    db.layer_height_mm  = cfg.layer_height_mm;
    db.line_width_mm    = cfg.layout.line_width_mm;
    db.layer_order      = cfg.recipe.layer_order;
    db.palette          = cfg.palette;

    BuildModelIRConfig build_cfg;
    build_cfg.flip_y      = true;
    build_cfg.base_layers = cfg.base_layers;

    ModelIR model = ModelIR::Build(map, db, build_cfg);

    BuildMeshConfig mesh_cfg;
    mesh_cfg.layer_height_mm = cfg.layer_height_mm;
    mesh_cfg.pixel_mm        = cfg.layout.line_width_mm / static_cast<float>(scale);
    Export3mf(board_path, model, mesh_cfg);
}

ColorDB GenColorDBFromImage(const std::string& image_path, const CalibrationBoardMeta& meta) {
    cv::Mat color_region = LocateCalibrationColorRegion(image_path, meta);
    return BuildColorDBFromColorRegion(color_region, meta);
}

ColorDB GenColorDBFromImage(const std::string& image_path, const std::string& json_path) {
    CalibrationBoardMeta meta = CalibrationBoardMeta::LoadFromJson(json_path);
    return GenColorDBFromImage(image_path, meta);
}

} // namespace ChromaPrint3D