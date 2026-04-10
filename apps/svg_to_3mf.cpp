#include "chromaprint3d/logging.h"
#include "chromaprint3d/pipeline.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace ChromaPrint3D;

namespace {

struct Options {
    std::string svg_path;
    std::vector<std::string> db_paths;
    std::string out_path;

    float target_width_mm  = 0.0f;
    float target_height_mm = 0.0f;

    ColorSpace color_space           = ColorSpace::Lab;
    int k_candidates                 = 1;
    int color_layers                 = 5;
    float color_layer_height_mm      = 0.0f;
    bool flip_y                      = false;
    NozzleSize nozzle_size           = NozzleSize::N04;
    FaceOrientation face_orientation = FaceOrientation::FaceUp;

    float layer_height_mm           = 0.0f;
    float tessellation_tolerance_mm = 0.03f;
    float gradient_pixel_mm         = 0.0f;
    int base_layers                 = -1;
    bool double_sided               = false;

    std::string log_level = "info";
};

void PrintUsage(const char* exe) {
    std::printf(
        "Usage: %s --svg input.svg --db color_db.json [--db more.json] --out output.3mf\n"
        "Options:\n"
        "  --svg PATH          Input SVG file\n"
        "  --db PATH           ColorDB file or directory (can be repeated)\n"
        "  --out PATH          Output 3MF path (default: <svg_stem>.3mf)\n"
        "  --width-mm X        Target width in mm (0 = SVG original)\n"
        "  --height-mm X       Target height in mm (0 = SVG original)\n"
        "  --mode 0.08x5|0.04x10  Print mode (default 0.08x5)\n"
        "  --color-space lab|rgb   Match in Lab or RGB (default lab)\n"
        "  --k N               Top-k candidates (default 1)\n"
        "  --flip-y 0|1        Flip Y axis (default 1)\n"
        "  --nozzle-size V     Nozzle size: n02|n04|0.2|0.4 (default n04)\n"
        "  --face-orientation V  Viewing face: faceup|facedown (default faceup)\n"
        "  --layer-mm X        Layer height in mm (default: from profile)\n"
        "  --base-layers N     Base layers override (-1 inherit, >=0 explicit, default -1)\n"
        "  --double-sided 0|1  Generate mirrored color layers on both sides (default 0)\n"
        "  --tess-tol X        Bezier tessellation tolerance in mm (default 0.03)\n"
        "  --gradient-px-mm X  Gradient rasterization resolution in mm/px (0=auto)\n"
        "  --log-level LEVEL   trace/debug/info/warn/error/off (default: info)\n",
        exe);
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

bool ParseBool(const char* s, bool& out) {
    if (!s) { return false; }
    std::string v(s);
    if (v == "1" || v == "true") {
        out = true;
        return true;
    }
    if (v == "0" || v == "false") {
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

void ParseMode(const std::string& value, int& color_layers, float& layer_height_mm) {
    if (value == "0.08x5" || value == "0p08x5") {
        color_layers    = 5;
        layer_height_mm = 0.08f;
        return;
    }
    if (value == "0.04x10" || value == "0p04x10") {
        color_layers    = 10;
        layer_height_mm = 0.04f;
        return;
    }
    throw std::runtime_error("Invalid --mode value: " + value);
}

bool ParseNozzleSizeArg(const std::string& value, NozzleSize& out) {
    std::string s = value;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "n02" || s == "0.2" || s == "0.2mm") {
        out = NozzleSize::N02;
        return true;
    }
    if (s == "n04" || s == "0.4" || s == "0.4mm") {
        out = NozzleSize::N04;
        return true;
    }
    return false;
}

bool ParseFaceOrientationArg(const std::string& value, FaceOrientation& out) {
    std::string s = value;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "faceup" || s == "face_up" || s == "up") {
        out = FaceOrientation::FaceUp;
        return true;
    }
    if (s == "facedown" || s == "face_down" || s == "down") {
        out = FaceOrientation::FaceDown;
        return true;
    }
    return false;
}

std::string DefaultOutPath(const std::string& svg_path) {
    if (svg_path.empty()) { return "output.3mf"; }
    return std::filesystem::path(svg_path).stem().string() + ".3mf";
}

bool ParseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--svg" && i + 1 < argc) {
            opt.svg_path = argv[++i];
            continue;
        }
        if (arg == "--db" && i + 1 < argc) {
            opt.db_paths.push_back(argv[++i]);
            continue;
        }
        if (arg == "--out" && i + 1 < argc) {
            opt.out_path = argv[++i];
            continue;
        }
        if (arg == "--width-mm" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.target_width_mm)) {
                std::fprintf(stderr, "Invalid --width-mm\n");
                return false;
            }
            continue;
        }
        if (arg == "--height-mm" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.target_height_mm)) {
                std::fprintf(stderr, "Invalid --height-mm\n");
                return false;
            }
            continue;
        }
        if (arg == "--mode" && i + 1 < argc) {
            ParseMode(argv[++i], opt.color_layers, opt.color_layer_height_mm);
            continue;
        }
        if (arg == "--color-space" && i + 1 < argc) {
            opt.color_space = ParseColorSpace(argv[++i]);
            continue;
        }
        if (arg == "--k" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.k_candidates) || opt.k_candidates < 1) {
                std::fprintf(stderr, "Invalid --k\n");
                return false;
            }
            continue;
        }
        if (arg == "--flip-y" && i + 1 < argc) {
            if (!ParseBool(argv[++i], opt.flip_y)) {
                std::fprintf(stderr, "Invalid --flip-y\n");
                return false;
            }
            continue;
        }
        if (arg == "--nozzle-size" && i + 1 < argc) {
            if (!ParseNozzleSizeArg(argv[++i], opt.nozzle_size)) {
                std::fprintf(stderr, "Invalid --nozzle-size\n");
                return false;
            }
            continue;
        }
        if (arg == "--face-orientation" && i + 1 < argc) {
            if (!ParseFaceOrientationArg(argv[++i], opt.face_orientation)) {
                std::fprintf(stderr, "Invalid --face-orientation\n");
                return false;
            }
            continue;
        }
        if (arg == "--layer-mm" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.layer_height_mm) || opt.layer_height_mm <= 0.0f) {
                std::fprintf(stderr, "Invalid --layer-mm\n");
                return false;
            }
            continue;
        }
        if (arg == "--base-layers" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.base_layers) || opt.base_layers < -1) {
                std::fprintf(stderr, "Invalid --base-layers\n");
                return false;
            }
            continue;
        }
        if (arg == "--double-sided" && i + 1 < argc) {
            if (!ParseBool(argv[++i], opt.double_sided)) {
                std::fprintf(stderr, "Invalid --double-sided\n");
                return false;
            }
            continue;
        }
        if (arg == "--tess-tol" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.tessellation_tolerance_mm) ||
                opt.tessellation_tolerance_mm <= 0.0f) {
                std::fprintf(stderr, "Invalid --tess-tol\n");
                return false;
            }
            continue;
        }
        if (arg == "--gradient-px-mm" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.gradient_pixel_mm) || opt.gradient_pixel_mm < 0.0f) {
                std::fprintf(stderr, "Invalid --gradient-px-mm\n");
                return false;
            }
            continue;
        }
        if (arg == "--log-level" && i + 1 < argc) {
            opt.log_level = argv[++i];
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return false;
        }
        std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
        PrintUsage(argv[0]);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!ParseArgs(argc, argv, opt)) { return 1; }

    InitLogging(ParseLogLevel(opt.log_level));

    if (opt.svg_path.empty() || opt.db_paths.empty()) {
        PrintUsage(argv[0]);
        return 1;
    }
    if (opt.out_path.empty()) { opt.out_path = DefaultOutPath(opt.svg_path); }

    try {
        ConvertVectorRequest req;
        req.svg_path         = opt.svg_path;
        req.db_paths         = opt.db_paths;
        req.target_width_mm  = opt.target_width_mm;
        req.target_height_mm = opt.target_height_mm;
        req.color_layers     = opt.color_layers;
        if (opt.color_layer_height_mm > 0.0f) req.layer_height_mm = opt.color_layer_height_mm;
        req.color_space               = opt.color_space;
        req.k_candidates              = opt.k_candidates;
        req.flip_y                    = opt.flip_y;
        req.nozzle_size               = opt.nozzle_size;
        req.face_orientation          = opt.face_orientation;
        req.layer_height_mm           = opt.layer_height_mm;
        req.base_layers               = opt.base_layers;
        req.double_sided              = opt.double_sided;
        req.tessellation_tolerance_mm = opt.tessellation_tolerance_mm;
        req.gradient_pixel_mm         = opt.gradient_pixel_mm;
        req.output_3mf_path           = opt.out_path;
        req.generate_preview          = false;

        ConvertResult result = ConvertVector(req);

        spdlog::info("Match stats: {} unique colors, avg_de={:.2f}", result.stats.clusters_total,
                     result.stats.avg_db_de);
        spdlog::info("Saved 3MF to {} ({} bytes)", opt.out_path, result.model_3mf.size());
    } catch (const std::exception& e) {
        spdlog::error("Failed: {}", e.what());
        return 1;
    }

    return 0;
}
