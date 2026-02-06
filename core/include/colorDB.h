#pragma once

#include "common.h"
#include "vec3.h"

#include "kdtree/kdtree.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ChromaPrint3D {

struct Entry {
    Lab lab;
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

    const Entry& NearestEntry(const Lab& target) const;

    const Entry& NearestEntry(const Rgb& target) const;

    std::vector<const Entry*> NearestEntries(const Lab& target, std::size_t k) const;

    std::vector<const Entry*> NearestEntries(const Rgb& target, std::size_t k) const;

private:
    struct LabProj {
        const Lab& operator()(const Entry& entry) const { return entry.lab; }
    };

    struct RgbProj {
        Rgb operator()(const Entry& entry) const { return entry.lab.ToRgb(); }
    };

    using KdIndex = std::size_t;
    using LabTree = kdt::KDTree<Entry, 3, LabProj, KdIndex, float>;
    using RgbTree = kdt::KDTree<Entry, 3, RgbProj, KdIndex, float>;

    void BuildKDTree() const;
    void EnsureKDTree() const;

    mutable std::vector<KdIndex> kd_indices_;
    mutable LabTree lab_tree_;
    mutable RgbTree rgb_tree_;
    mutable std::size_t kd_entries_size_ = 0;
};

} // namespace ChromaPrint3D