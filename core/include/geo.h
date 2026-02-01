#pragma once

#include "vec3.h"
#include "colorDB.h"
#include "match.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ChromaPrint3D {

struct VoxelGrid {
    int width       = 0;
    int height      = 0;
    int num_layers  = 0; // 层序：0 为最底层，向上递增
    int channel_idx = 0;

    // VoxelGrid 层序为自底向上，layer 0 是最底层（含 base）。
    std::vector<uint8_t> ooc; // H * W * L，与 num_layers 层序一致

    bool Get(int w, int h, int l) const;
    bool Set(int w, int h, int l, bool v);
};

struct BuildModelIRConfig {
    bool flip_y     = true;
    int base_layers = 0;       // 0 表示使用 ColorDB 的数据
    bool double_sided = false; // 双面模型：两侧各有 color_layers，中间共享 base_layers
};

struct ModelIR {
    int width  = 0;
    int height = 0;

    int color_layers     = 0;
    int base_layers      = 0;
    int base_channel_idx = 0;

    std::vector<Channel> palette;
    std::vector<VoxelGrid> voxel_grids; // 1 channel per grid

    size_t NumChannels() const { return palette.size(); }

    static ModelIR Build(const RecipeMap& recipe_map, const ColorDB& db,
                         const BuildModelIRConfig& cfg = BuildModelIRConfig{});
};

struct BuildMeshConfig {
    float layer_height_mm = 0.08f;
    float pixel_mm        = 1.0f;
};

struct Mesh {
    std::vector<Vec3f> vertices;
    std::vector<Vec3i> indices;

    static Mesh Build(const VoxelGrid& voxel_grid, const BuildMeshConfig& cfg = BuildMeshConfig{});
};

void Export3mf(const std::string& path, const ModelIR& ModelIR);

} // namespace ChromaPrint3D