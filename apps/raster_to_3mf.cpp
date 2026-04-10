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
    std::string image_path;
    std::vector<std::string> db_paths;
    std::string out_path;
    std::string preview_path;
    std::string source_mask_path;

    float request_scale = 1.0f;
    int max_width       = 512;
    int max_height      = 512;

    ColorSpace color_space           = ColorSpace::Lab;
    int k_candidates                 = 1;
    int cluster_count                = 0;
    int color_layers                 = 5;
    float color_layer_height_mm      = 0.0f;
    bool flip_y                      = true;
    NozzleSize nozzle_size           = NozzleSize::N04;
    FaceOrientation face_orientation = FaceOrientation::FaceUp;

    std::string model_pack_path;
    bool model_enable        = true;
    bool model_only          = false;
    bool model_threshold_set = false;
    float model_threshold    = 0.0f;
    bool model_margin_set    = false;
    float model_margin       = 0.0f;

    float pixel_mm        = 0.0f;
    float layer_height_mm = 0.0f;
    int base_layers       = -1;
    bool double_sided     = false;

    std::string log_level = "info";
};

void PrintUsage(const char* exe) {
    std::printf(
        "Usage: %s --image input.png --db color_db.json [--db more.json] --out output.3mf\n"
        "Options:\n"
        "  --db PATH          ColorDB file or directory (can be repeated)\n"
        "  --scale S           Image scale factor (default 1.0)\n"
        "  --max-width N       Max width for resize (default 512, 0 = no limit)\n"
        "  --max-height N      Max height for resize (default 512, 0 = no limit)\n"
        "  --mode 0.08x5|0.04x10   Target print mode (default 0.08x5)\n"
        "  --color-space lab|rgb   Match in Lab or RGB (default lab)\n"
        "  --k N               Top-k candidates (default 1)\n"
        "  --clusters N        Cluster count (0=auto-detect, 1=per-pixel, >=2=manual; default 0)\n"
        "  --model-pack PATH   Phase A model package json\n"
        "  --model-enable 0|1  Enable model fallback when model pack is provided (default 1)\n"
        "  --model-only 0|1    Match by model only (requires --model-pack, default 0)\n"
        "  --model-threshold X Override model fallback threshold (DeltaE76)\n"
        "  --model-margin X    Override model fallback margin (DeltaE76)\n"
        "  --flip-y 0|1        Flip Y axis when building model (default 1)\n"
        "  --nozzle-size V     Nozzle size: n02|n04|0.2|0.4 (default n04)\n"
        "  --face-orientation V  Viewing face: faceup|facedown (default faceup)\n"
        "  --pixel-mm X        Pixel size in mm (default: db.line_width_mm)\n"
        "  --layer-mm X        Layer height in mm (default: db.layer_height_mm)\n"
        "  --base-layers N     Base layers override (-1 inherit, >=0 explicit, default -1)\n"
        "  --double-sided 0|1  Generate mirrored color layers on both sides (default 0)\n"
        "  --preview PATH      Save preview image (default: <out>_preview.png)\n"
        "  --source-mask PATH  Save source mask image (default: <out>_source_mask.png)\n"
        "  --log-level LEVEL   Log level: trace/debug/info/warn/error/off (default: info)\n",
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

std::string DefaultOutPath(const std::string& image_path) {
    if (image_path.empty()) { return "output.3mf"; }
    std::filesystem::path p(image_path);
    std::string stem = p.stem().string();
    if (stem.empty()) { return "output.3mf"; }
    return stem + ".3mf";
}

std::string DefaultPreviewPath(const std::string& out_path) {
    std::filesystem::path base(out_path);
    if (base.empty()) { return "preview.png"; }
    std::string stem = base.stem().string();
    if (stem.empty()) { stem = "preview"; }
    return (base.parent_path() / (stem + "_preview.png")).string();
}

std::string DefaultSourceMaskPath(const std::string& out_path) {
    std::filesystem::path base(out_path);
    if (base.empty()) { return "source_mask.png"; }
    std::string stem = base.stem().string();
    if (stem.empty()) { stem = "output"; }
    return (base.parent_path() / (stem + "_source_mask.png")).string();
}

bool ParseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) {
            opt.image_path = argv[++i];
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
        if (arg == "--preview" && i + 1 < argc) {
            opt.preview_path = argv[++i];
            continue;
        }
        if (arg == "--source-mask" && i + 1 < argc) {
            opt.source_mask_path = argv[++i];
            continue;
        }
        if (arg == "--scale" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.request_scale) || opt.request_scale <= 0.0f) {
                std::fprintf(stderr, "Invalid --scale value\n");
                return false;
            }
            continue;
        }
        if (arg == "--max-width" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.max_width) || opt.max_width < 0) {
                std::fprintf(stderr, "Invalid --max-width value\n");
                return false;
            }
            continue;
        }
        if (arg == "--max-height" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.max_height) || opt.max_height < 0) {
                std::fprintf(stderr, "Invalid --max-height value\n");
                return false;
            }
            continue;
        }
        if (arg == "--color-space" && i + 1 < argc) {
            opt.color_space = ParseColorSpace(argv[++i]);
            continue;
        }
        if (arg == "--mode" && i + 1 < argc) {
            ParseMode(argv[++i], opt.color_layers, opt.color_layer_height_mm);
            continue;
        }
        if (arg == "--k" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.k_candidates) || opt.k_candidates < 1) {
                std::fprintf(stderr, "Invalid --k value\n");
                return false;
            }
            continue;
        }
        if (arg == "--clusters" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.cluster_count) || opt.cluster_count < 0) {
                std::fprintf(stderr, "Invalid --clusters value\n");
                return false;
            }
            continue;
        }
        if (arg == "--model-pack" && i + 1 < argc) {
            opt.model_pack_path = argv[++i];
            continue;
        }
        if (arg == "--model-enable" && i + 1 < argc) {
            if (!ParseBool(argv[++i], opt.model_enable)) {
                std::fprintf(stderr, "Invalid --model-enable value\n");
                return false;
            }
            continue;
        }
        if (arg == "--model-only" && i + 1 < argc) {
            if (!ParseBool(argv[++i], opt.model_only)) {
                std::fprintf(stderr, "Invalid --model-only value\n");
                return false;
            }
            continue;
        }
        if (arg == "--model-threshold" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.model_threshold) || opt.model_threshold < 0.0f) {
                std::fprintf(stderr, "Invalid --model-threshold value\n");
                return false;
            }
            opt.model_threshold_set = true;
            continue;
        }
        if (arg == "--model-margin" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.model_margin) || opt.model_margin < 0.0f) {
                std::fprintf(stderr, "Invalid --model-margin value\n");
                return false;
            }
            opt.model_margin_set = true;
            continue;
        }
        if (arg == "--flip-y" && i + 1 < argc) {
            if (!ParseBool(argv[++i], opt.flip_y)) {
                std::fprintf(stderr, "Invalid --flip-y value\n");
                return false;
            }
            continue;
        }
        if (arg == "--nozzle-size" && i + 1 < argc) {
            if (!ParseNozzleSizeArg(argv[++i], opt.nozzle_size)) {
                std::fprintf(stderr, "Invalid --nozzle-size value\n");
                return false;
            }
            continue;
        }
        if (arg == "--face-orientation" && i + 1 < argc) {
            if (!ParseFaceOrientationArg(argv[++i], opt.face_orientation)) {
                std::fprintf(stderr, "Invalid --face-orientation value\n");
                return false;
            }
            continue;
        }
        if (arg == "--pixel-mm" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.pixel_mm) || opt.pixel_mm <= 0.0f) {
                std::fprintf(stderr, "Invalid --pixel-mm value\n");
                return false;
            }
            continue;
        }
        if (arg == "--layer-mm" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.layer_height_mm) || opt.layer_height_mm <= 0.0f) {
                std::fprintf(stderr, "Invalid --layer-mm value\n");
                return false;
            }
            continue;
        }
        if (arg == "--base-layers" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.base_layers) || opt.base_layers < -1) {
                std::fprintf(stderr, "Invalid --base-layers value\n");
                return false;
            }
            continue;
        }
        if (arg == "--double-sided" && i + 1 < argc) {
            if (!ParseBool(argv[++i], opt.double_sided)) {
                std::fprintf(stderr, "Invalid --double-sided value\n");
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

    if (opt.image_path.empty() || opt.db_paths.empty()) {
        PrintUsage(argv[0]);
        return 1;
    }
    if (opt.model_only && opt.model_pack_path.empty()) {
        spdlog::error("--model-only requires --model-pack");
        return 1;
    }
    if (opt.out_path.empty()) { opt.out_path = DefaultOutPath(opt.image_path); }
    if (opt.preview_path.empty()) { opt.preview_path = DefaultPreviewPath(opt.out_path); }
    if (opt.source_mask_path.empty()) {
        opt.source_mask_path = DefaultSourceMaskPath(opt.out_path);
    }

    try {
        ConvertRasterRequest req;
        req.image_path      = opt.image_path;
        req.db_paths        = opt.db_paths;
        req.model_pack_path = opt.model_pack_path;
        req.scale           = opt.request_scale;
        req.max_width       = opt.max_width;
        req.max_height      = opt.max_height;
        req.color_layers    = opt.color_layers;
        if (opt.color_layer_height_mm > 0.0f) req.layer_height_mm = opt.color_layer_height_mm;
        req.color_space      = opt.color_space;
        req.k_candidates     = opt.k_candidates;
        req.cluster_count    = opt.cluster_count;
        req.model_enable     = opt.model_enable;
        req.model_only       = opt.model_only;
        req.model_threshold  = opt.model_threshold_set ? opt.model_threshold : -1.0f;
        req.model_margin     = opt.model_margin_set ? opt.model_margin : -1.0f;
        req.flip_y           = opt.flip_y;
        req.nozzle_size      = opt.nozzle_size;
        req.face_orientation = opt.face_orientation;
        req.pixel_mm         = opt.pixel_mm;
        req.layer_height_mm  = opt.layer_height_mm;
        req.base_layers      = opt.base_layers;
        req.double_sided     = opt.double_sided;

        req.generate_preview     = true;
        req.generate_source_mask = true;
        req.output_3mf_path      = opt.out_path;
        req.preview_path         = opt.preview_path;
        req.source_mask_path     = opt.source_mask_path;

        ConvertResult result = ConvertRaster(req);

        spdlog::info("Match stats: clusters_total={}, db_only={}, model_fallback={}, "
                     "model_queries={}, avg_db_de={:.2f}, avg_model_de={:.2f}",
                     result.stats.clusters_total, result.stats.db_only, result.stats.model_fallback,
                     result.stats.model_queries, result.stats.avg_db_de, result.stats.avg_model_de);

        if (!opt.preview_path.empty()) { spdlog::info("Saved preview to {}", opt.preview_path); }
        if (!opt.source_mask_path.empty()) {
            spdlog::info("Saved source mask to {}", opt.source_mask_path);
        }
        spdlog::info("Saved 3MF to {}", opt.out_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed: {}", e.what());
        return 1;
    }

    return 0;
}
