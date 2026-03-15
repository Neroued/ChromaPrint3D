#include "chromaprint3d/logging.h"
#include "chromaprint3d/vectorizer.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace ChromaPrint3D;

namespace {

struct Options {
    std::string image_path;
    std::string out_path;
    int colors                      = 0;
    int min_region_area             = 50;
    float curve_fit_error           = 0.8f;
    float corner_angle              = 135.0f;
    float smoothing_spatial         = 15.0f;
    float smoothing_color           = 25.0f;
    int upscale_short_edge          = 600;
    int max_working_pixels          = 3000000;
    int slic_region_size            = 20;
    float slic_compactness          = 6.0f;
    float thin_line_radius          = 2.5f;
    float min_contour               = 10.0f;
    float min_hole_area             = 4.0f;
    float contour_simplify          = 0.45f;
    float edge_sensitivity          = 0.8f;
    int refine_passes               = 6;
    float max_merge_color_dist      = 200.0f;
    bool enable_subpixel_refine     = true;
    float subpixel_max_displacement = 0.7f;
    bool enable_coverage_fix        = true;
    float min_coverage_ratio        = 0.998f;
    float smoothness                = 0.5f;
    float detail_level              = -1.0f;
    float merge_segment_tolerance   = 0.05f;
    bool enable_antialias_detect    = false;
    float aa_tolerance              = 10.0f;
    bool svg_stroke                 = true;
    float svg_stroke_w              = 0.5f;
    std::string log_level           = "info";
};

void PrintUsage(const char* exe) {
    std::printf("Usage: %s --image input.png [--out output.svg] [options]\n"
                "Options:\n"
                "  --colors N          Number of quantization colors, 0 = auto (default 0)\n"
                "  --min-region N      Min region area in pixels (default 50)\n"
                "  --curve-fit-error F Curve fitting error threshold (default 0.8)\n"
                "  --corner-angle F    Corner angle threshold in degrees (default 135)\n"
                "  --smoothing-spatial F Mean Shift spatial radius (default 15)\n"
                "  --smoothing-color F Mean Shift color radius (default 25)\n"
                "  --upscale-short-edge N Auto-upscale short-edge threshold, 0 disables "
                "(default 600)\n"
                "  --max-working-pixels N Auto-downscale pixel count threshold, 0 disables "
                "(default 3000000)\n"
                "  --slic-region-size N SLIC region size in pixels, 0 uses auto (default 20)\n"
                "  --slic-compactness F SLIC compactness, lower follows edges more (default 6)\n"
                "  --thin-line-radius F Thin-line max radius in pixels (default 2.5)\n"
                "  --min-contour F     Min contour area in pixels (default 10)\n"
                "  --min-hole-area F   Minimum kept hole area in pixels^2 (default 4.0)\n"
                "  --contour-simplify F  Contour simplification strength (default 0.45)\n"
                "  --edge-sensitivity F  Edge-aware SLIC sensitivity [0,1] (default 0.8)\n"
                "  --refine-passes N     Boundary label refinement iterations (default 6)\n"
                "  --max-merge-color-dist F Max LAB dE^2 for small-region merging (default 200)\n"
                "  --disable-subpixel-refine Disable sub-pixel boundary refinement\n"
                "  --subpixel-max-displacement F Sub-pixel max displacement (default 0.7)\n"
                "  --disable-coverage-fix Disable coverage patching\n"
                "  --min-coverage-ratio F Coverage fix trigger ratio (default 0.998)\n"
                "  --smoothness F      Contour smoothness [0,1] (default 0.5)\n"
                "  --detail-level F    Unified detail control [0,1], -1 disables (default -1)\n"
                "  --merge-tolerance F Near-linear segment merge tolerance (default 0.05)\n"
                "  --enable-antialias  Enable AA mixed-edge detection\n"
                "  --aa-tolerance F    AA blend detection LAB tolerance (default 10)\n"
                "  --no-svg-stroke     Disable SVG stroke output (default on)\n"
                "  --svg-stroke-w F    SVG stroke width when enabled (default 0.5)\n"
                "  --log-level LEVEL   Log level: trace/debug/info/warn/error/off (default info)\n",
                exe);
}

bool ParseInt(const char* s, int& out) {
    if (!s) return false;
    try {
        size_t idx = 0;
        out        = std::stoi(s, &idx, 10);
        return idx == std::string(s).size();
    } catch (...) { return false; }
}

bool ParseFloat(const char* s, float& out) {
    if (!s) return false;
    try {
        size_t idx = 0;
        out        = std::stof(s, &idx);
        return idx == std::string(s).size();
    } catch (...) { return false; }
}

bool ParseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return false;
        }
        if (arg == "--image" && i + 1 < argc) {
            opt.image_path = argv[++i];
            continue;
        }
        if (arg == "--out" && i + 1 < argc) {
            opt.out_path = argv[++i];
            continue;
        }
        if (arg == "--colors" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.colors) || (opt.colors != 0 && opt.colors < 2)) {
                std::fprintf(stderr, "Invalid --colors (0=auto, or >=2)\n");
                return false;
            }
            continue;
        }
        if (arg == "--min-region" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.min_region_area) || opt.min_region_area < 0) {
                std::fprintf(stderr, "Invalid --min-region\n");
                return false;
            }
            continue;
        }
        if (arg == "--curve-fit-error" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.curve_fit_error) || opt.curve_fit_error <= 0.0f) {
                std::fprintf(stderr, "Invalid --curve-fit-error\n");
                return false;
            }
            continue;
        }
        if (arg == "--corner-angle" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.corner_angle) || opt.corner_angle <= 0.0f ||
                opt.corner_angle >= 180.0f) {
                std::fprintf(stderr, "Invalid --corner-angle\n");
                return false;
            }
            continue;
        }
        if (arg == "--smoothing-spatial" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.smoothing_spatial) || opt.smoothing_spatial < 0.0f) {
                std::fprintf(stderr, "Invalid --smoothing-spatial\n");
                return false;
            }
            continue;
        }
        if (arg == "--smoothing-color" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.smoothing_color) || opt.smoothing_color < 0.0f) {
                std::fprintf(stderr, "Invalid --smoothing-color\n");
                return false;
            }
            continue;
        }
        if (arg == "--upscale-short-edge" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.upscale_short_edge) || opt.upscale_short_edge < 0) {
                std::fprintf(stderr, "Invalid --upscale-short-edge\n");
                return false;
            }
            continue;
        }
        if (arg == "--max-working-pixels" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.max_working_pixels) || opt.max_working_pixels < 0) {
                std::fprintf(stderr, "Invalid --max-working-pixels\n");
                return false;
            }
            continue;
        }
        if (arg == "--slic-region-size" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.slic_region_size) || opt.slic_region_size < 0) {
                std::fprintf(stderr, "Invalid --slic-region-size\n");
                return false;
            }
            continue;
        }
        if (arg == "--slic-compactness" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.slic_compactness) || opt.slic_compactness < 0.0f) {
                std::fprintf(stderr, "Invalid --slic-compactness\n");
                return false;
            }
            continue;
        }
        if (arg == "--thin-line-radius" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.thin_line_radius) || opt.thin_line_radius <= 0.0f) {
                std::fprintf(stderr, "Invalid --thin-line-radius\n");
                return false;
            }
            continue;
        }
        if (arg == "--min-contour" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.min_contour)) {
                std::fprintf(stderr, "Invalid --min-contour\n");
                return false;
            }
            continue;
        }
        if (arg == "--min-hole-area" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.min_hole_area) || opt.min_hole_area < 0.0f) {
                std::fprintf(stderr, "Invalid --min-hole-area\n");
                return false;
            }
            continue;
        }
        if (arg == "--contour-simplify" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.contour_simplify) || opt.contour_simplify < 0.0f) {
                std::fprintf(stderr, "Invalid --contour-simplify\n");
                return false;
            }
            continue;
        }
        if (arg == "--edge-sensitivity" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.edge_sensitivity) || opt.edge_sensitivity < 0.0f ||
                opt.edge_sensitivity > 1.0f) {
                std::fprintf(stderr, "Invalid --edge-sensitivity\n");
                return false;
            }
            continue;
        }
        if (arg == "--refine-passes" && i + 1 < argc) {
            if (!ParseInt(argv[++i], opt.refine_passes) || opt.refine_passes < 0) {
                std::fprintf(stderr, "Invalid --refine-passes\n");
                return false;
            }
            continue;
        }
        if (arg == "--max-merge-color-dist" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.max_merge_color_dist) ||
                opt.max_merge_color_dist < 0.0f) {
                std::fprintf(stderr, "Invalid --max-merge-color-dist\n");
                return false;
            }
            continue;
        }
        if (arg == "--disable-subpixel-refine") {
            opt.enable_subpixel_refine = false;
            continue;
        }
        if (arg == "--subpixel-max-displacement" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.subpixel_max_displacement) ||
                opt.subpixel_max_displacement < 0.0f) {
                std::fprintf(stderr, "Invalid --subpixel-max-displacement\n");
                return false;
            }
            continue;
        }
        if (arg == "--disable-coverage-fix") {
            opt.enable_coverage_fix = false;
            continue;
        }
        if (arg == "--min-coverage-ratio" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.min_coverage_ratio) || opt.min_coverage_ratio < 0.0f ||
                opt.min_coverage_ratio > 1.0f) {
                std::fprintf(stderr, "Invalid --min-coverage-ratio\n");
                return false;
            }
            continue;
        }
        if (arg == "--smoothness" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.smoothness)) {
                std::fprintf(stderr, "Invalid --smoothness\n");
                return false;
            }
            continue;
        }
        if (arg == "--detail-level" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.detail_level)) {
                std::fprintf(stderr, "Invalid --detail-level\n");
                return false;
            }
            continue;
        }
        if (arg == "--merge-tolerance" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.merge_segment_tolerance)) {
                std::fprintf(stderr, "Invalid --merge-tolerance\n");
                return false;
            }
            continue;
        }
        if (arg == "--enable-antialias") {
            opt.enable_antialias_detect = true;
            continue;
        }
        if (arg == "--aa-tolerance" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.aa_tolerance)) {
                std::fprintf(stderr, "Invalid --aa-tolerance\n");
                return false;
            }
            continue;
        }
        if (arg == "--no-svg-stroke") {
            opt.svg_stroke = false;
            continue;
        }
        if (arg == "--svg-stroke-w" && i + 1 < argc) {
            if (!ParseFloat(argv[++i], opt.svg_stroke_w) || opt.svg_stroke_w < 0.0f) {
                std::fprintf(stderr, "Invalid --svg-stroke-w\n");
                return false;
            }
            continue;
        }
        if (arg == "--log-level" && i + 1 < argc) {
            opt.log_level = argv[++i];
            continue;
        }
        std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
        PrintUsage(argv[0]);
        return false;
    }
    return true;
}

std::string DefaultOutPath(const std::string& image_path) {
    std::filesystem::path p(image_path);
    return (p.parent_path() / (p.stem().string() + ".svg")).string();
}

} // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!ParseArgs(argc, argv, opt)) return 1;

    InitLogging(ParseLogLevel(opt.log_level));

    if (opt.image_path.empty()) {
        std::fprintf(stderr, "Error: --image is required\n");
        PrintUsage(argv[0]);
        return 1;
    }
    if (opt.out_path.empty()) opt.out_path = DefaultOutPath(opt.image_path);

    try {
        VectorizerConfig cfg;
        cfg.num_colors                = opt.colors;
        cfg.min_region_area           = opt.min_region_area;
        cfg.curve_fit_error           = opt.curve_fit_error;
        cfg.corner_angle_threshold    = opt.corner_angle;
        cfg.smoothing_spatial         = opt.smoothing_spatial;
        cfg.smoothing_color           = opt.smoothing_color;
        cfg.upscale_short_edge        = opt.upscale_short_edge;
        cfg.max_working_pixels        = opt.max_working_pixels;
        cfg.slic_region_size          = opt.slic_region_size;
        cfg.slic_compactness          = opt.slic_compactness;
        cfg.thin_line_max_radius      = opt.thin_line_radius;
        cfg.min_contour_area          = opt.min_contour;
        cfg.min_hole_area             = opt.min_hole_area;
        cfg.contour_simplify          = opt.contour_simplify;
        cfg.edge_sensitivity          = opt.edge_sensitivity;
        cfg.refine_passes             = opt.refine_passes;
        cfg.max_merge_color_dist      = opt.max_merge_color_dist;
        cfg.enable_subpixel_refine    = opt.enable_subpixel_refine;
        cfg.subpixel_max_displacement = opt.subpixel_max_displacement;
        cfg.enable_coverage_fix       = opt.enable_coverage_fix;
        cfg.min_coverage_ratio        = opt.min_coverage_ratio;
        cfg.smoothness                = opt.smoothness;
        cfg.detail_level              = opt.detail_level;
        cfg.merge_segment_tolerance   = opt.merge_segment_tolerance;
        cfg.enable_antialias_detect   = opt.enable_antialias_detect;
        cfg.aa_tolerance              = opt.aa_tolerance;
        cfg.svg_enable_stroke         = opt.svg_stroke;
        cfg.svg_stroke_width          = opt.svg_stroke_w;

        spdlog::info("Vectorizing {} -> {}", opt.image_path, opt.out_path);
        spdlog::info("Colors={}, contour_simplify={:.2f}, edge_sensitivity={:.2f}, "
                     "refine_passes={}, max_merge_color_dist={:.1f}",
                     cfg.num_colors, cfg.contour_simplify, cfg.edge_sensitivity, cfg.refine_passes,
                     cfg.max_merge_color_dist);

        auto result = Vectorize(opt.image_path, cfg);

        std::ofstream ofs(opt.out_path);
        if (!ofs) throw std::runtime_error("Cannot open output file: " + opt.out_path);
        ofs << result.svg_content;
        ofs.close();

        spdlog::info("Done: {}x{}, {} shapes, palette size {}", result.width, result.height,
                     result.num_shapes, static_cast<int>(result.palette.size()));
        spdlog::info("Saved SVG to {}", opt.out_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed: {}", e.what());
        return 1;
    }

    return 0;
}
