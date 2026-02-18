#pragma once

/// \file voxel.h
/// \brief VoxelGrid and ModelIR — intermediate 3D representations.

#include "common.h"
#include "color_db.h"
#include "recipe_map.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ChromaPrint3D {

/// Dense boolean occupancy grid for one material channel.
struct VoxelGrid {
    int width       = 0;
    int height      = 0;
    int num_layers  = 0;
    int channel_idx = 0;

    /// Flat occupancy array (H * W * L), layer 0 is the bottom.
    std::vector<uint8_t> ooc;

    bool Get(int w, int h, int l) const;
    bool Set(int w, int h, int l, bool v);
};

/// Configuration for ModelIR::Build().
struct BuildModelIRConfig {
    bool flip_y       = true;
    int base_layers   = 0;
    bool double_sided = false;
    std::vector<uint8_t> base_only_mask;
};

/// Intermediate representation of a multi-material 3D model.
struct ModelIR {
    std::string name;

    int width  = 0;
    int height = 0;

    int color_layers     = 0;
    int base_layers      = 0;
    int base_channel_idx = 0;

    std::vector<Channel> palette;
    std::vector<VoxelGrid> voxel_grids;

    size_t NumChannels() const { return palette.size(); }

    /// Build from a RecipeMap and its associated ColorDB.
    static ModelIR Build(const RecipeMap& recipe_map, const ColorDB& db,
                         const BuildModelIRConfig& cfg = BuildModelIRConfig{});
};

} // namespace ChromaPrint3D
