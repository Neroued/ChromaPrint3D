#pragma once

#include "common.h"
#include "vec3.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ChromaPrint3D {

struct Entry {
    Vec3f lab;
    std::vector<uint8_t> recipe; // size() == color_layers

    size_t ColorLayers() const { return recipe.size(); }
};

struct Channel {
    std::string color    = "Default Color";    // e.g. "Cyan"
    std::string material = "Default Material"; // e.g. "PLA Basic"
};

class ColorDB {
public:
    std::string name = "Default ColorDB";

    int max_color_layers = 0; // 叠色区配方最大层数

    int base_layers      = 0; // 底板层数
    int base_channel_idx = 0; // 底板使用的通道

    float layer_height_mm = 0.08f; // 层高
    float line_width_mm   = 0.42f; // 线宽

    LayerOrder layer_order = LayerOrder::Top2Bottom;

    std::vector<Channel> palette; // size() == num_channels
    std::vector<Entry> entries;

    ColorDB()  = default;
    ~ColorDB() = default;

    size_t NumChannels() const { return palette.size(); }

    static ColorDB LoadFromJson(const std::string& path);

    void SaveToJson(const std::string& path) const;

    const Entry& NearestEntry(const Vec3f& target, ColorSpace space = ColorSpace::Lab) const;

    std::vector<const Entry*> NearestEntries(const Vec3f& target, std::size_t k,
                                             ColorSpace space = ColorSpace::Lab) const;
};

} // namespace ChromaPrint3D