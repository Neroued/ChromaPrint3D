#include "chromaprint3d/raster_region_map.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <queue>
#include <utility>

namespace ChromaPrint3D {

RasterRegionMap RasterRegionMap::Build(const RecipeMap& recipe_map) {
    RasterRegionMap result;
    result.width  = recipe_map.width;
    result.height = recipe_map.height;

    const int w              = result.width;
    const int h              = result.height;
    const std::size_t total  = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    const std::size_t layers = static_cast<std::size_t>(recipe_map.color_layers);

    result.region_ids.assign(total, kMaskedRegion);

    uint32_t next_region_id = 0;
    std::queue<std::pair<int, int>> bfs;

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            const std::size_t idx = static_cast<std::size_t>(r) * static_cast<std::size_t>(w) +
                                    static_cast<std::size_t>(c);

            if (recipe_map.mask[idx] == 0) { continue; }
            if (result.region_ids[idx] != kMaskedRegion) { continue; }

            const uint32_t region_id = next_region_id++;
            result.region_ids[idx]   = region_id;

            const uint8_t* ref_recipe = &recipe_map.recipes[idx * layers];
            int pixel_count           = 1;

            bfs.push({r, c});

            while (!bfs.empty()) {
                auto [cr, cc] = bfs.front();
                bfs.pop();

                static constexpr int dr[] = {-1, 1, 0, 0};
                static constexpr int dc[] = {0, 0, -1, 1};

                for (int d = 0; d < 4; ++d) {
                    const int nr = cr + dr[d];
                    const int nc = cc + dc[d];
                    if (nr < 0 || nr >= h || nc < 0 || nc >= w) { continue; }

                    const std::size_t nidx =
                        static_cast<std::size_t>(nr) * static_cast<std::size_t>(w) +
                        static_cast<std::size_t>(nc);
                    if (result.region_ids[nidx] != kMaskedRegion) { continue; }
                    if (recipe_map.mask[nidx] == 0) { continue; }

                    const uint8_t* neighbor_recipe = &recipe_map.recipes[nidx * layers];
                    if (!std::equal(ref_recipe, ref_recipe + layers, neighbor_recipe)) { continue; }

                    result.region_ids[nidx] = region_id;
                    ++pixel_count;
                    bfs.push({nr, nc});
                }
            }

            RasterRegionInfo info;
            info.region_id = static_cast<int>(region_id);
            info.recipe.assign(ref_recipe, ref_recipe + layers);
            info.mapped_color = recipe_map.mapped_color[idx];
            info.pixel_count  = pixel_count;
            info.from_model   = (recipe_map.source_mask[idx] != 0);
            result.regions.push_back(std::move(info));
        }
    }

    return result;
}

std::vector<uint8_t> RasterRegionMap::SerializeRegionIds() const {
    const std::size_t total = region_ids.size();
    std::vector<uint8_t> buf(total * sizeof(uint32_t));
    std::memcpy(buf.data(), region_ids.data(), buf.size());
    return buf;
}

std::size_t RasterRegionMap::EstimateBytes() const {
    std::size_t bytes = sizeof(*this);
    bytes += region_ids.capacity() * sizeof(uint32_t);
    for (const auto& info : regions) { bytes += sizeof(RasterRegionInfo) + info.recipe.capacity(); }
    return bytes;
}

} // namespace ChromaPrint3D
