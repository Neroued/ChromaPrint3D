#pragma once

/// \file raster_region_map.h
/// \brief Connected-component region map for raster recipe editing.

#include "color.h"
#include "recipe_map.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ChromaPrint3D {

struct RasterRegionInfo {
    int region_id = 0;
    std::vector<uint8_t> recipe;
    Lab mapped_color;
    int pixel_count = 0;
    bool from_model = false;
};

struct RasterRegionMap {
    int width  = 0;
    int height = 0;
    std::vector<uint32_t> region_ids;
    std::vector<RasterRegionInfo> regions;

    static constexpr uint32_t kMaskedRegion = 0xFFFFFFFF;

    static RasterRegionMap Build(const RecipeMap& recipe_map);

    std::vector<uint8_t> SerializeRegionIds() const;

    std::size_t EstimateBytes() const;
};

} // namespace ChromaPrint3D
