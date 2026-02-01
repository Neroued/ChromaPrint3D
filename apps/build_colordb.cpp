#include "calib.h"
#include "colorDB.h"

#include <filesystem>
#include <iostream>
#include <string>

using namespace ChromaPrint3D;

namespace {

void PrintUsage(const char* exe) {
    std::cout << "Usage: " << exe << " --image calib.png --meta calib.json --out color_db.json\n";
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
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        }
        std::cerr << "Unknown argument: " << arg << "\n";
        PrintUsage(argv[0]);
        return 1;
    }

    if (image_path.empty() || meta_path.empty()) {
        PrintUsage(argv[0]);
        return 1;
    }
    if (out_path.empty()) { out_path = DefaultOutPath(image_path); }

    try {
        ColorDB db = GenColorDBFromImage(image_path, meta_path);
        db.SaveToJson(out_path);
        std::cout << "Saved ColorDB to " << out_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Failed to build ColorDB: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
