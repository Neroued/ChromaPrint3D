#include "chromaprint3d/logging.h"
#include "chromaprint3d/voxel.h"
#include "chromaprint3d/mesh.h"
#include "chromaprint3d/export_3mf.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ChromaPrint3D;

namespace {

struct StageConfig {
    int grid_cols = 6;
    int grid_rows = 3;
    int block_mm  = 10;
    int gap_mm    = 1;
    int margin_mm = 0;

    float pixel_mm        = 1.0f;
    float layer_height_mm = 0.04f;
    float base_mm         = 1.0f;

    int max_step_layers = 25;

    std::vector<int> step_layers;

    StageConfig() {
        margin_mm = gap_mm;
        step_layers.reserve(18);
        for (int i = 0; i <= 16; ++i) { step_layers.push_back(i); }
        step_layers.push_back(25);
    }
};

void PrintUsage(const char* exe) { std::printf("Usage: %s [--out stage.3mf]\n", exe); }

size_t GridIndex(int x, int y, int z, int width, int height, int layers) {
    return (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) *
               static_cast<size_t>(layers) +
           static_cast<size_t>(z);
}

void FillBox(VoxelGrid& grid, int x0, int y0, int x1, int y1, int z0, int z1) {
    if (grid.width <= 0 || grid.height <= 0 || grid.num_layers <= 0) { return; }
    const int width  = grid.width;
    const int height = grid.height;
    const int layers = grid.num_layers;

    if (grid.ooc.size() <
        static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(layers)) {
        throw std::runtime_error("VoxelGrid ooc size mismatch");
    }

    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    z0 = std::max(0, z0);
    x1 = std::min(x1, width);
    y1 = std::min(y1, height);
    z1 = std::min(z1, layers);
    if (x0 >= x1 || y0 >= y1 || z0 >= z1) { return; }

    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            const size_t base = GridIndex(x, y, 0, width, height, layers);
            for (int z = z0; z < z1; ++z) { grid.ooc[base + static_cast<size_t>(z)] = 1; }
        }
    }
}

ModelIR BuildStageModel(const StageConfig& cfg) {
    if (cfg.grid_cols <= 0 || cfg.grid_rows <= 0) {
        throw std::runtime_error("Grid size must be positive");
    }
    if (cfg.pixel_mm <= 0.0f || cfg.layer_height_mm <= 0.0f) {
        throw std::runtime_error("pixel_mm and layer_height_mm must be positive");
    }
    if (cfg.max_step_layers <= 0) { throw std::runtime_error("max_step_layers is invalid"); }

    const int block_px  = static_cast<int>(std::lround(cfg.block_mm / cfg.pixel_mm));
    const int gap_px    = static_cast<int>(std::lround(cfg.gap_mm / cfg.pixel_mm));
    const int margin_px = static_cast<int>(std::lround(cfg.margin_mm / cfg.pixel_mm));
    if (block_px <= 0 || gap_px < 0 || margin_px < 0) {
        throw std::runtime_error("block/gap/margin sizes are invalid");
    }

    const int width  = cfg.grid_cols * block_px + (cfg.grid_cols - 1) * gap_px + 2 * margin_px;
    const int height = cfg.grid_rows * block_px + (cfg.grid_rows - 1) * gap_px + 2 * margin_px;
    if (width <= 0 || height <= 0) { throw std::runtime_error("Model size is invalid"); }

    const int base_layers = static_cast<int>(std::lround(cfg.base_mm / cfg.layer_height_mm));
    if (base_layers <= 0) { throw std::runtime_error("base_layers is invalid"); }

    const int total_layers = base_layers + cfg.max_step_layers;
    if (total_layers <= 0) { throw std::runtime_error("total_layers is invalid"); }

    if (static_cast<int>(cfg.step_layers.size()) != cfg.grid_cols * cfg.grid_rows) {
        throw std::runtime_error("step_layers size does not match grid size");
    }

    ModelIR model;
    model.name             = "stage";
    model.width            = width;
    model.height           = height;
    model.color_layers     = cfg.max_step_layers;
    model.base_layers      = base_layers;
    model.base_channel_idx = 0;

    model.palette.resize(2);
    model.palette[0].color    = "Base";
    model.palette[0].material = "base";
    model.palette[1].color    = "Step";
    model.palette[1].material = "color";

    model.voxel_grids.resize(2);
    const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height) *
                            static_cast<size_t>(total_layers);
    for (int ch = 0; ch < 2; ++ch) {
        VoxelGrid& grid  = model.voxel_grids[static_cast<size_t>(ch)];
        grid.width       = width;
        grid.height      = height;
        grid.num_layers  = total_layers;
        grid.channel_idx = ch;
        grid.ooc.assign(expected, 0);
    }

    VoxelGrid& base_grid = model.voxel_grids[0];
    VoxelGrid& step_grid = model.voxel_grids[1];

    FillBox(base_grid, 0, 0, width, height, 0, base_layers);

    for (size_t i = 0; i < cfg.step_layers.size(); ++i) {
        const int step_layers = std::min(cfg.step_layers[i], cfg.max_step_layers);
        if (step_layers <= 0) { continue; }

        const int row = static_cast<int>(i) / cfg.grid_cols;
        const int col = static_cast<int>(i) % cfg.grid_cols;

        const int x0 = margin_px + col * (block_px + gap_px);
        const int y0 = margin_px + row * (block_px + gap_px);
        const int x1 = x0 + block_px;
        const int y1 = y0 + block_px;
        FillBox(step_grid, x0, y0, x1, y1, base_layers, base_layers + step_layers);
    }

    return model;
}

} // namespace

int main(int argc, char** argv) {
    StageConfig cfg;
    std::string out_path  = "stage.3mf";
    std::string log_level = "info";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
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

    try {
        ModelIR model = BuildStageModel(cfg);

        BuildMeshConfig mesh_cfg;
        mesh_cfg.layer_height_mm = cfg.layer_height_mm;
        mesh_cfg.pixel_mm        = cfg.pixel_mm;

        Export3mf(out_path, model, mesh_cfg);
        spdlog::info("Saved stage model to {}", out_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed: {}", e.what());
        return 1;
    }

    return 0;
}
