#include "colorDB.h"
#include "geo.h"
#include "imgproc.h"
#include "match.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

using namespace ChromaPrint3D;

namespace {

struct Options {
    std::string image_path;
    std::string db_path;
    std::string out_path;

    float request_scale = 1.0f;
    int max_width       = 0;
    int max_height      = 0;

    ColorSpace color_space = ColorSpace::Lab;
    int k_candidates       = 1;
    bool flip_y            = true;

    float pixel_mm        = 0.0f;
    float layer_height_mm = 0.0f;
};

void PrintUsage(const char* exe) {
    std::cout << "Usage: " << exe << " --image input.png --db color_db.json --out output.3mf\n"
              << "Options:\n"
              << "  --scale S           ImgProc request scale (default 1.0)\n"
              << "  --max-width N       Max width for resize (0 = no limit)\n"
              << "  --max-height N      Max height for resize (0 = no limit)\n"
              << "  --color-space lab|rgb   Match in Lab or RGB (default lab)\n"
              << "  --k N               Top-k candidates (default 1)\n"
              << "  --flip-y 0|1        Flip Y axis when building model (default 1)\n"
              << "  --pixel-mm X        Pixel size in mm (default: db.line_width_mm)\n"
              << "  --layer-mm X        Layer height in mm (default: db.layer_height_mm)\n";
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

bool ParseFloat(const char* s, float& out) {
    if (!s) { return false; }
    try {
        size_t idx  = 0;
        float value = std::stof(s, &idx);
        if (idx != std::string(s).size()) { return false; }
        out = value;
        return true;
    } catch (...) { return false; }
}

bool ParseBool(const char* s, bool& out) {
    if (!s) { return false; }
    std::string v(s);
    if (v == "1" || v == "true" || v == "TRUE" || v == "True") {
        out = true;
        return true;
    }
    if (v == "0" || v == "false" || v == "FALSE" || v == "False") {
        out = false;
        return true;
    }
    return false;
}

ColorSpace ParseColorSpace(const std::string& value) {
    if (value == "lab" || value == "Lab" || value == "LAB") { return ColorSpace::Lab; }
    if (value == "rgb" || value == "Rgb" || value == "RGB") { return ColorSpace::Rgb; }
    throw std::runtime_error("Invalid color-space: " + value);
}

std::string DefaultOutPath(const std::string& image_path) {
    if (image_path.empty()) { return "output.3mf"; }
    std::filesystem::path p(image_path);
    std::string stem = p.stem().string();
    if (stem.empty()) { return "output.3mf"; }
    return stem + ".3mf";
}

} // namespace

int main(int argc, char** argv) {
    Options opt;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) {
            opt.image_path = argv[++i];
            continue;
        }
        if (arg == "--db" && i + 1 < argc) {
            opt.db_path = argv[++i];
            continue;
        }
        if (arg == "--out" && i + 1 < argc) {
            opt.out_path = argv[++i];
            continue;
        }
        if (arg == "--scale" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.request_scale) || opt.request_scale <= 0.0f) {
                std::cerr << "Invalid --scale value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--max-width" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.max_width) || opt.max_width < 0) {
                std::cerr << "Invalid --max-width value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--max-height" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.max_height) || opt.max_height < 0) {
                std::cerr << "Invalid --max-height value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--color-space" && i + 1 < argc) {
            opt.color_space = ParseColorSpace(argv[++i]);
            continue;
        }
        if (arg == "--k" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.k_candidates) || opt.k_candidates < 1) {
                std::cerr << "Invalid --k value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--flip-y" && i + 1 < argc) {
            if (!ParseBool(argv[++i], opt.flip_y)) {
                std::cerr << "Invalid --flip-y value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--pixel-mm" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.pixel_mm) || opt.pixel_mm <= 0.0f) {
                std::cerr << "Invalid --pixel-mm value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--layer-mm" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.layer_height_mm) || opt.layer_height_mm <= 0.0f) {
                std::cerr << "Invalid --layer-mm value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        }
        std::cerr << "Unknown argument: " << arg << "\n";
        PrintUsage(argv[0]);
        return 1;
    }

    if (opt.image_path.empty() || opt.db_path.empty()) {
        PrintUsage(argv[0]);
        return 1;
    }
    if (opt.out_path.empty()) { opt.out_path = DefaultOutPath(opt.image_path); }

    try {
        ColorDB db = ColorDB::LoadFromJson(opt.db_path);

        ImgProc imgproc;
        imgproc.request_scale = opt.request_scale;
        imgproc.max_width     = opt.max_width;
        imgproc.max_height    = opt.max_height;

        ImgProcResult img = imgproc.Run(opt.image_path);

        MatchConfig match_cfg;
        match_cfg.color_space  = opt.color_space;
        match_cfg.k_candidates = opt.k_candidates;

        RecipeMap recipe_map = RecipeMap::MatchFromImage(img, db, match_cfg);

        BuildModelIRConfig build_cfg;
        build_cfg.flip_y = opt.flip_y;

        ModelIR model = ModelIR::Build(recipe_map, db, build_cfg);

        BuildMeshConfig mesh_cfg;
        mesh_cfg.pixel_mm        = (opt.pixel_mm > 0.0f)
                                       ? opt.pixel_mm
                                       : (db.line_width_mm > 0.0f ? db.line_width_mm : 1.0f);
        mesh_cfg.layer_height_mm = (opt.layer_height_mm > 0.0f)
                                       ? opt.layer_height_mm
                                       : (db.layer_height_mm > 0.0f ? db.layer_height_mm : 0.08f);

        Export3mf(opt.out_path, model, mesh_cfg);

        std::cout << "Saved 3MF to " << opt.out_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
