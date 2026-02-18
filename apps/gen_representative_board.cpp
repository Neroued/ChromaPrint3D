#include "chromaprint3d/logging.h"
#include "chromaprint3d/calib.h"
#include "chromaprint3d/color_db.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ChromaPrint3D;
using nlohmann::json;

namespace {

void PrintUsage(const char* exe) {
    std::printf("Usage: %s --recipes recipes.json [--ref-db color_db.json]"
                " [--out board.3mf] [--meta board.json] [--scale N]\n"
                "Defaults: --out representative_board.3mf --meta uses same path/name as --out "
                "with .json extension\n"
                "If recipes.json contains palette/base/layer info, --ref-db is optional.\n",
                exe);
}

bool ParseInt(const char* s, int& out) {
    if (!s) { return false; }
    try {
        size_t idx = 0;
        int value  = std::stoi(s, &idx, 10);
        if (idx != std::string(s).size()) { return false; }
        out = value;
        return true;
    } catch (...) { return false; }
}

LayerOrder ParseLayerOrderString(const std::string& order) {
    if (order == "Top2Bottom") { return LayerOrder::Top2Bottom; }
    if (order == "Bottom2Top") { return LayerOrder::Bottom2Top; }
    throw std::runtime_error("Invalid layer_order string: " + order);
}

std::string BuildDefaultMetaPath(const std::string& out_path) {
    std::filesystem::path path(out_path);
    path.replace_extension(".json");
    return path.string();
}

std::string NormalizeLabel(const std::string& label) {
    std::string out;
    out.reserve(label.size());
    for (unsigned char ch : label) {
        if (std::isalnum(ch)) { out.push_back(static_cast<char>(std::tolower(ch))); }
    }
    return out;
}

int FindWhiteChannelIndex(const std::vector<Channel>& palette) {
    for (size_t i = 0; i < palette.size(); ++i) {
        if (NormalizeLabel(palette[i].color) == "white") { return static_cast<int>(i); }
    }
    return -1;
}

struct RecipePayload {
    std::vector<std::vector<uint8_t>> recipes;
    std::vector<Channel> palette;
    bool has_palette          = false;
    bool has_layer_height     = false;
    bool has_line_width       = false;
    bool has_base_layers      = false;
    bool has_base_channel_idx = false;
    bool has_layer_order      = false;
    float layer_height_mm     = 0.08f;
    float line_width_mm       = 0.42f;
    int base_layers           = 10;
    int base_channel_idx      = 0;
    LayerOrder layer_order    = LayerOrder::Top2Bottom;
};

static RecipePayload LoadRecipes(const std::string& path, int& layers_out) {
    std::ifstream in(path);
    if (!in.is_open()) { throw std::runtime_error("Failed to open recipes: " + path); }
    json j;
    in >> j;

    RecipePayload payload;
    json recipes_json;
    if (j.is_array()) {
        recipes_json = j;
    } else if (j.is_object()) {
        if (!j.contains("recipes")) {
            throw std::runtime_error("recipes json missing recipes field");
        }
        recipes_json = j.at("recipes");
        if (j.contains("palette")) {
            const auto& p = j.at("palette");
            if (!p.is_array()) { throw std::runtime_error("palette must be an array"); }
            for (const auto& item : p) {
                Channel c;
                c.color    = item.value("color", c.color);
                c.material = item.value("material", c.material);
                payload.palette.push_back(c);
            }
            payload.has_palette = !payload.palette.empty();
        }
        if (j.contains("base_layers")) {
            payload.base_layers     = j.at("base_layers").get<int>();
            payload.has_base_layers = true;
        }
        if (j.contains("base_channel_idx")) {
            payload.base_channel_idx     = j.at("base_channel_idx").get<int>();
            payload.has_base_channel_idx = true;
        }
        if (j.contains("layer_height_mm")) {
            payload.layer_height_mm  = j.at("layer_height_mm").get<float>();
            payload.has_layer_height = true;
        }
        if (j.contains("line_width_mm")) {
            payload.line_width_mm  = j.at("line_width_mm").get<float>();
            payload.has_line_width = true;
        }
        if (j.contains("layer_order")) {
            payload.layer_order     = ParseLayerOrderString(j.at("layer_order").get<std::string>());
            payload.has_layer_order = true;
        }
    } else {
        throw std::runtime_error("recipes json must be an array or object");
    }

    if (!recipes_json.is_array()) { throw std::runtime_error("recipes must be an array"); }
    payload.recipes.reserve(recipes_json.size());

    for (const auto& item : recipes_json) {
        json recipe_json;
        if (item.is_array()) {
            recipe_json = item;
        } else if (item.is_object() && item.contains("recipe")) {
            recipe_json = item.at("recipe");
        } else {
            throw std::runtime_error("recipe item must be array or object with recipe field");
        }

        if (!recipe_json.is_array()) { throw std::runtime_error("recipe must be an array"); }

        std::vector<uint8_t> recipe;
        recipe.reserve(recipe_json.size());
        for (const auto& v : recipe_json) {
            int value = v.get<int>();
            if (value < 0 || value > 255) { throw std::runtime_error("recipe value out of range"); }
            recipe.push_back(static_cast<uint8_t>(value));
        }
        if (recipe.empty()) { throw std::runtime_error("recipe is empty"); }
        if (layers_out == 0) { layers_out = static_cast<int>(recipe.size()); }
        if (static_cast<int>(recipe.size()) != layers_out) {
            throw std::runtime_error("inconsistent recipe length");
        }
        payload.recipes.push_back(recipe);
    }
    return payload;
}

} // namespace

int main(int argc, char** argv) {
    std::string recipes_path;
    std::string ref_db_path;
    std::string out_path = "representative_board.3mf";
    std::string meta_path;
    bool meta_path_provided = false;
    int resolution_scale    = 0;
    std::string log_level   = "info";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--recipes" && i + 1 < argc) {
            recipes_path = argv[++i];
            continue;
        }
        if (arg == "--ref-db" && i + 1 < argc) {
            ref_db_path = argv[++i];
            continue;
        }
        if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
            continue;
        }
        if (arg == "--meta" && i + 1 < argc) {
            meta_path          = argv[++i];
            meta_path_provided = true;
            continue;
        }
        if (arg == "--scale" && i + 1 < argc) {
            if (!ParseInt(argv[i + 1], resolution_scale) || resolution_scale <= 0) {
                std::fprintf(stderr, "Invalid --scale value\n");
                return 1;
            }
            i++;
            continue;
        }
        if (arg == "--log-level" && i + 1 < argc) {
            log_level = argv[++i];
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        }
        std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
        PrintUsage(argv[0]);
        return 1;
    }

    InitLogging(ParseLogLevel(log_level));

    if (recipes_path.empty()) {
        PrintUsage(argv[0]);
        return 1;
    }
    if (!meta_path_provided) { meta_path = BuildDefaultMetaPath(out_path); }

    try {
        int layers            = 0;
        RecipePayload payload = LoadRecipes(recipes_path, layers);
        if (payload.recipes.size() < 1024) {
            throw std::runtime_error("recipes count must be >= 1024");
        }

        const bool has_ref_db = !ref_db_path.empty();
        ColorDB db;
        if (has_ref_db) { db = ColorDB::LoadFromJson(ref_db_path); }

        const int num_channels = payload.has_palette ? static_cast<int>(payload.palette.size())
                                                     : static_cast<int>(db.NumChannels());
        if (num_channels <= 0) { throw std::runtime_error("no palette channels available"); }

        for (size_t i = 0; i < payload.recipes.size(); ++i) {
            for (uint8_t v : payload.recipes[i]) {
                if (v >= static_cast<uint8_t>(num_channels)) {
                    throw std::runtime_error("recipe channel index out of range");
                }
            }
        }

        CalibrationBoardConfig cfg = CalibrationBoardConfig::ForChannels(num_channels);
        cfg.recipe.color_layers    = layers;
        if (payload.has_layer_order) {
            cfg.recipe.layer_order = payload.layer_order;
        } else if (has_ref_db) {
            cfg.recipe.layer_order = db.layer_order;
        }

        if (payload.has_base_layers) {
            cfg.base_layers = payload.base_layers;
        } else if (has_ref_db) {
            cfg.base_layers = db.base_layers;
        }

        if (payload.has_base_channel_idx) {
            cfg.base_channel_idx = payload.base_channel_idx;
        } else {
            int white_idx = -1;
            if (payload.has_palette) {
                white_idx = FindWhiteChannelIndex(payload.palette);
            } else if (has_ref_db) {
                white_idx = FindWhiteChannelIndex(db.palette);
            }
            if (white_idx >= 0) {
                cfg.base_channel_idx = white_idx;
            } else if (has_ref_db) {
                cfg.base_channel_idx = db.base_channel_idx;
            }
        }

        if (payload.has_layer_height) {
            cfg.layer_height_mm = payload.layer_height_mm;
        } else if (has_ref_db) {
            cfg.layer_height_mm = db.layer_height_mm;
        } else {
            throw std::runtime_error("layer_height_mm missing (no ref db)");
        }

        if (payload.has_line_width) {
            cfg.layout.line_width_mm = payload.line_width_mm;
        } else if (has_ref_db) {
            cfg.layout.line_width_mm = db.line_width_mm;
        } else {
            throw std::runtime_error("line_width_mm missing (no ref db)");
        }

        if (payload.has_palette) {
            cfg.palette = payload.palette;
        } else if (has_ref_db) {
            cfg.palette = db.palette;
        } else {
            throw std::runtime_error("palette missing (no ref db)");
        }

        if (has_ref_db && payload.has_palette) {
            if (static_cast<int>(db.NumChannels()) != static_cast<int>(payload.palette.size())) {
                throw std::runtime_error("palette channels mismatch with ref db");
            }
        }
        if (cfg.base_channel_idx < 0 ||
            cfg.base_channel_idx >= static_cast<int>(cfg.palette.size())) {
            throw std::runtime_error("base_channel_idx out of range");
        }
        if (resolution_scale > 0) { cfg.layout.resolution_scale = resolution_scale; }

        CalibrationBoardMeta meta;
        meta.name = "RepresentativeBoard_" + std::to_string(num_channels) + "ch_" +
                    std::to_string(layers) + "L";
        meta.config    = cfg;
        meta.grid_rows = 32;
        meta.grid_cols = 32;

        const size_t total = 1024;
        meta.patch_recipe_idx.resize(total);
        meta.patch_recipes.resize(total);
        for (size_t i = 0; i < total; ++i) {
            meta.patch_recipe_idx[i] = static_cast<uint16_t>(i);
            meta.patch_recipes[i]    = payload.recipes[i];
        }

        GenCalibrationBoardFromMeta(meta, out_path, meta_path);

        spdlog::info("Saved board to {}", out_path);
        spdlog::info("Saved meta to {}", meta_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to generate representative board: {}", e.what());
        return 1;
    }
    return 0;
}
