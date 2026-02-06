#include "colorDB.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>

namespace ChromaPrint3D {
namespace {
using nlohmann::json;

static std::string ToLayerOrderString(LayerOrder order) {
    switch (order) {
    case LayerOrder::Top2Bottom:
        return "Top2Bottom";
    case LayerOrder::Bottom2Top:
        return "Bottom2Top";
    }
    return "Top2Bottom";
}

static LayerOrder FromLayerOrderString(const std::string& order) {
    if (order == "Top2Bottom") { return LayerOrder::Top2Bottom; }
    if (order == "Bottom2Top") { return LayerOrder::Bottom2Top; }
    throw std::runtime_error("Invalid layer_order string: " + order);
}

static LayerOrder ParseLayerOrder(const json& value) {
    if (value.is_string()) { return FromLayerOrderString(value.get<std::string>()); }
    if (value.is_number_integer()) {
        int v = value.get<int>();
        if (v == 0) { return LayerOrder::Top2Bottom; }
        if (v == 1) { return LayerOrder::Bottom2Top; }
    }
    throw std::runtime_error("Invalid layer_order value");
}

} // namespace

ColorDB ColorDB::LoadFromJson(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) { throw std::runtime_error("Failed to open file: " + path); }

    json j;
    in >> j;

    ColorDB db;
    db.name             = j.value("name", db.name);
    db.max_color_layers = j.value("max_color_layers", db.max_color_layers);
    db.base_layers      = j.value("base_layers", db.base_layers);
    db.base_channel_idx = j.value("base_channel_idx", db.base_channel_idx);
    db.layer_height_mm  = j.value("layer_height_mm", db.layer_height_mm);
    db.line_width_mm    = j.value("line_width_mm", db.line_width_mm);

    if (j.contains("layer_order")) { db.layer_order = ParseLayerOrder(j.at("layer_order")); }

    db.palette.clear();
    if (j.contains("palette")) {
        const auto& p = j.at("palette");
        if (!p.is_array()) { throw std::runtime_error("palette must be an array"); }
        for (const auto& item : p) {
            Channel c;
            c.color    = item.value("color", c.color);
            c.material = item.value("material", c.material);
            db.palette.push_back(c);
        }
    }

    db.entries.clear();
    if (j.contains("entries")) {
        const auto& e = j.at("entries");
        if (!e.is_array()) { throw std::runtime_error("entries must be an array"); }
        for (const auto& item : e) {
            if (!item.contains("lab")) { throw std::runtime_error("entry missing lab"); }
            if (!item.contains("recipe")) { throw std::runtime_error("entry missing recipe"); }

            const auto& lab = item.at("lab");
            if (!lab.is_array() || lab.size() != 3) {
                throw std::runtime_error("lab must be an array of size 3");
            }

            Entry entry;
            entry.lab = Lab::FromLab(lab.at(0).get<float>(), lab.at(1).get<float>(),
                                     lab.at(2).get<float>());

            const auto& recipe = item.at("recipe");
            if (!recipe.is_array()) { throw std::runtime_error("recipe must be an array"); }
            entry.recipe.reserve(recipe.size());
            for (const auto& v : recipe) {
                int value = v.get<int>();
                if (value < 0 || value > 255) {
                    throw std::runtime_error("recipe value out of range: " + std::to_string(value));
                }
                entry.recipe.push_back(static_cast<uint8_t>(value));
            }
            db.entries.push_back(entry);
        }
    }

    db.BuildKDTree();
    return db;
}

void ColorDB::SaveToJson(const std::string& path) const {
    json j;
    j["name"]             = name;
    j["max_color_layers"] = max_color_layers;
    j["base_layers"]      = base_layers;
    j["base_channel_idx"] = base_channel_idx;
    j["layer_height_mm"]  = layer_height_mm;
    j["line_width_mm"]    = line_width_mm;
    j["layer_order"]      = ToLayerOrderString(layer_order);

    j["palette"] = json::array();
    for (const auto& channel : palette) {
        json c;
        c["color"]    = channel.color;
        c["material"] = channel.material;
        j["palette"].push_back(c);
    }

    j["entries"] = json::array();
    for (const auto& entry : entries) {
        json e;
        e["lab"]    = json::array({entry.lab.l(), entry.lab.a(), entry.lab.b()});
        e["recipe"] = json::array();
        for (uint8_t v : entry.recipe) { e["recipe"].push_back(v); }
        j["entries"].push_back(e);
    }

    std::ofstream out(path);
    if (!out.is_open()) { throw std::runtime_error("Failed to open file: " + path); }
    out << j.dump(4);
    if (!out.good()) { throw std::runtime_error("Failed to write json: " + path); }
}

const Entry& ColorDB::NearestEntry(const Lab& target) const {
    if (entries.empty()) { throw std::runtime_error("ColorDB entries is empty"); }
    EnsureKDTree();
    const auto neighbor = lab_tree_.Nearest(target);
    return entries[static_cast<std::size_t>(neighbor.index)];
}

const Entry& ColorDB::NearestEntry(const Rgb& target) const {
    if (entries.empty()) { throw std::runtime_error("ColorDB entries is empty"); }
    EnsureKDTree();
    const auto neighbor = rgb_tree_.Nearest(target);
    return entries[static_cast<std::size_t>(neighbor.index)];
}

std::vector<const Entry*> ColorDB::NearestEntries(const Lab& target, std::size_t k) const {
    if (k == 0) { return {}; }
    if (entries.empty()) { throw std::runtime_error("ColorDB entries is empty"); }

    EnsureKDTree();
    k = std::min(k, entries.size());

    std::vector<LabTree::NeighborT> neighbors;
    lab_tree_.KNearest(target, k, neighbors);

    std::vector<const Entry*> result;
    result.reserve(neighbors.size());
    for (const auto& n : neighbors) {
        result.push_back(&entries[static_cast<std::size_t>(n.index)]);
    }
    return result;
}

std::vector<const Entry*> ColorDB::NearestEntries(const Rgb& target, std::size_t k) const {
    if (k == 0) { return {}; }
    if (entries.empty()) { throw std::runtime_error("ColorDB entries is empty"); }

    EnsureKDTree();
    k = std::min(k, entries.size());

    std::vector<RgbTree::NeighborT> neighbors;
    rgb_tree_.KNearest(target, k, neighbors);

    std::vector<const Entry*> result;
    result.reserve(neighbors.size());
    for (const auto& n : neighbors) {
        result.push_back(&entries[static_cast<std::size_t>(n.index)]);
    }
    return result;
}

void ColorDB::BuildKDTree() const {
    kd_indices_.clear();
    kd_indices_.reserve(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        kd_indices_.push_back(i);
    }

    const auto points  = std::span<const Entry>(entries);
    const auto indices = std::span<const KdIndex>(kd_indices_);
    lab_tree_.Reset(points, indices, LabProj{});
    rgb_tree_.Reset(points, indices, RgbProj{});
    kd_entries_size_ = entries.size();
}

void ColorDB::EnsureKDTree() const {
    if (entries.size() != kd_entries_size_) {
        BuildKDTree();
    }
}

} // namespace ChromaPrint3D
