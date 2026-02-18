#include "chromaprint3d/logging.h"
#include "chromaprint3d/calib.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ChromaPrint3D;

namespace {

void PrintUsage(const char* exe) {
    std::printf("Usage: %s [--channels N] [--out board.3mf] [--meta board.json] [--scale N]\n"
                "Defaults: --channels 4 --out calibration_board.3mf --meta calibration_board.json\n"
                "          --scale uses CalibrationBoardLayout.resolution_scale\n",
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

void ApplyDefaultPalette(CalibrationBoardConfig& cfg) {
    const std::string material = "PLA Basic";
    for (auto& channel : cfg.palette) { channel.material = material; }

    cfg.palette.resize(static_cast<size_t>(cfg.recipe.num_channels));
    if (cfg.recipe.num_channels == 4) {
        const char* names[] = {"White", "Yellow", "Red", "Blue"};
        for (int i = 0; i < 4; ++i) { cfg.palette[static_cast<size_t>(i)].color = names[i]; }
        return;
    }
    if (cfg.recipe.num_channels == 3) {
        const char* names[] = {"White", "Gray", "Black"};
        for (int i = 0; i < 3; ++i) { cfg.palette[static_cast<size_t>(i)].color = names[i]; }
        return;
    }
    if (cfg.recipe.num_channels == 2) {
        cfg.palette[0].color = "Color A";
        cfg.palette[1].color = "Color B";
        return;
    }
    for (int i = 0; i < cfg.recipe.num_channels; ++i) {
        cfg.palette[static_cast<size_t>(i)].color = "Channel " + std::to_string(i);
    }
}

} // namespace

int main(int argc, char** argv) {
    int num_channels      = 4;
    std::string out_path  = "calibration_board.3mf";
    std::string meta_path = "calibration_board.json";
    int resolution_scale  = 0;
    std::string log_level = "info";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--channels" && i + 1 < argc) {
            if (!ParseInt(argv[i + 1], num_channels)) {
                std::fprintf(stderr, "Invalid --channels value\n");
                return 1;
            }
            i++;
            continue;
        }
        if (arg == "--out" && i + 1 < argc) {
            out_path = argv[i + 1];
            i++;
            continue;
        }
        if (arg == "--meta" && i + 1 < argc) {
            meta_path = argv[i + 1];
            i++;
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
            log_level = argv[i + 1];
            i++;
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

    try {
        CalibrationBoardConfig cfg = CalibrationBoardConfig::ForChannels(num_channels);
        ApplyDefaultPalette(cfg);
        if (resolution_scale > 0) { cfg.layout.resolution_scale = resolution_scale; }

        GenCalibrationBoard(cfg, out_path, meta_path);

        spdlog::info("Saved board to {}", out_path);
        spdlog::info("Saved meta to {}", meta_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to generate board: {}", e.what());
        return 1;
    }
    return 0;
}
