#include "chromaprint3d/logging.h"
#include "chromaprint3d/calib.h"
#include "chromaprint3d/color_db.h"

#include <cstdio>
#include <filesystem>
#include <string>

using namespace ChromaPrint3D;

namespace {

void PrintUsage(const char* exe) {
    std::printf("Usage: %s --image calib.png --meta calib.json --out color_db.json\n", exe);
}

std::string DefaultOutPath(const std::string& image_path) {
    if (image_path.empty()) { return "color_db.json"; }
    std::filesystem::path p(image_path);
    std::string stem = p.stem().string();
    if (stem.empty()) { return "color_db.json"; }
    return stem + "_colordb.json";
}

} // namespace

int main(int argc, char** argv) {
    std::string image_path;
    std::string meta_path;
    std::string out_path;
    std::string log_level = "info";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
            continue;
        }
        if (arg == "--meta" && i + 1 < argc) {
            meta_path = argv[++i];
            continue;
        }
        if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
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

    if (image_path.empty() || meta_path.empty()) {
        PrintUsage(argv[0]);
        return 1;
    }
    if (out_path.empty()) { out_path = DefaultOutPath(image_path); }

    try {
        ColorDB db = GenColorDBFromImage(image_path, meta_path);
        db.SaveToJson(out_path);
        spdlog::info("Saved ColorDB to {}", out_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to build ColorDB: {}", e.what());
        return 1;
    }

    return 0;
}
