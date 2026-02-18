#include "chromaprint3d/color_db.h"
#include "detail/json_utils.h"
#include "chromaprint3d/error.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <fstream>
#include <span>
#include <string>
#include <utility>

namespace ChromaPrint3D {

using nlohmann::json;

ColorDB::ColorDB(const ColorDB& other)
    : name(other.name), max_color_layers(other.max_color_layers), base_layers(other.base_layers),
      base_channel_idx(other.base_channel_idx), layer_height_mm(other.layer_height_mm),
      line_width_mm(other.line_width_mm), layer_order(other.layer_order), palette(other.palette),
      entries(other.entries) {
    if (!entries.empty()) { BuildKDTree(); }
}

ColorDB::ColorDB(ColorDB&& other) noexcept
    : name(std::move(other.name)), max_color_layers(other.max_color_layers),
      base_layers(other.base_layers), base_channel_idx(other.base_channel_idx),
      layer_height_mm(other.layer_height_mm), line_width_mm(other.line_width_mm),
      layer_order(other.layer_order), palette(std::move(other.palette)),
      entries(std::move(other.entries)) {
    if (!entries.empty()) { BuildKDTree(); }
    other.ResetKDTreeCache();
}

ColorDB& ColorDB::operator=(const ColorDB& other) {
    if (this == &other) { return *this; }
    name             = other.name;
    max_color_layers = other.max_color_layers;
    base_layers      = other.base_layers;
    base_channel_idx = other.base_channel_idx;
    layer_height_mm  = other.layer_height_mm;
    line_width_mm    = other.line_width_mm;
    layer_order      = other.layer_order;
    palette          = other.palette;
    entries          = other.entries;
    ResetKDTreeCache();
    if (!entries.empty()) { BuildKDTree(); }
    return *this;
}

ColorDB& ColorDB::operator=(ColorDB&& other) noexcept {
    if (this == &other) { return *this; }
    name             = std::move(other.name);
    max_color_layers = other.max_color_layers;
    base_layers      = other.base_layers;
    base_channel_idx = other.base_channel_idx;
    layer_height_mm  = other.layer_height_mm;
    line_width_mm    = other.line_width_mm;
    layer_order      = other.layer_order;
    palette          = std::move(other.palette);
    entries          = std::move(other.entries);
    ResetKDTreeCache();
    if (!entries.empty()) { BuildKDTree(); }
    other.ResetKDTreeCache();
    return *this;
}

static ColorDB DBFromJson(const json& j) {
    ColorDB db;
    db.name             = j.value("name", db.name);
    db.max_color_layers = j.value("max_color_layers", db.max_color_layers);
    db.base_layers      = j.value("base_layers", db.base_layers);
    db.base_channel_idx = j.value("base_channel_idx", db.base_channel_idx);
    db.layer_height_mm  = j.value("layer_height_mm", db.layer_height_mm);
    db.line_width_mm    = j.value("line_width_mm", db.line_width_mm);

    if (j.contains("layer_order")) { db.layer_order = detail::ParseLayerOrder(j.at("layer_order")); }

    db.palette.clear();
    if (j.contains("palette")) {
        const auto& p = j.at("palette");
        if (!p.is_array()) { throw FormatError("palette must be an array"); }
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
        if (!e.is_array()) { throw FormatError("entries must be an array"); }
        for (const auto& item : e) {
            if (!item.contains("lab")) { throw FormatError("entry missing lab"); }
            if (!item.contains("recipe")) { throw FormatError("entry missing recipe"); }

            const auto& lab = item.at("lab");
            if (!lab.is_array() || lab.size() != 3) {
                throw FormatError("lab must be an array of size 3");
            }

            Entry entry;
            entry.lab = Lab::FromLab(lab.at(0).get<float>(), lab.at(1).get<float>(),
                                     lab.at(2).get<float>());

            const auto& recipe = item.at("recipe");
            if (!recipe.is_array()) { throw FormatError("recipe must be an array"); }
            entry.recipe.reserve(recipe.size());
            for (const auto& v : recipe) {
                int value = v.get<int>();
                if (value < 0 || value > 255) {
                    throw FormatError("recipe value out of range: " + std::to_string(value));
                }
                entry.recipe.push_back(static_cast<uint8_t>(value));
            }
            db.entries.push_back(entry);
        }
    }

    return db;
}

static json DBToJson(const ColorDB& db) {
    json j;
    j["name"]             = db.name;
    j["max_color_layers"] = db.max_color_layers;
    j["base_layers"]      = db.base_layers;
    j["base_channel_idx"] = db.base_channel_idx;
    j["layer_height_mm"]  = db.layer_height_mm;
    j["line_width_mm"]    = db.line_width_mm;
    j["layer_order"]      = ToLayerOrderString(db.layer_order);

    j["palette"] = json::array();
    for (const auto& channel : db.palette) {
        json c;
        c["color"]    = channel.color;
        c["material"] = channel.material;
        j["palette"].push_back(c);
    }

    j["entries"] = json::array();
    for (const auto& entry : db.entries) {
        json e;
        e["lab"]    = json::array({entry.lab.l(), entry.lab.a(), entry.lab.b()});
        e["recipe"] = json::array();
        for (uint8_t v : entry.recipe) { e["recipe"].push_back(v); }
        j["entries"].push_back(e);
    }
    return j;
}

ColorDB ColorDB::LoadFromJson(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) { throw IOError("Failed to open file: " + path); }
    json j;
    in >> j;
    ColorDB db = DBFromJson(j);
    db.BuildKDTree();
    spdlog::info("ColorDB loaded: name={}, entries={}, channels={}", db.name, db.entries.size(),
                 db.NumChannels());
    return db;
}

ColorDB ColorDB::FromJsonString(const std::string& json_str) {
    json j     = json::parse(json_str);
    ColorDB db = DBFromJson(j);
    db.BuildKDTree();
    spdlog::info("ColorDB parsed: name={}, entries={}, channels={}", db.name, db.entries.size(),
                 db.NumChannels());
    return db;
}

void ColorDB::SaveToJson(const std::string& path) const {
    json j = DBToJson(*this);
    std::ofstream out(path);
    if (!out.is_open()) { throw IOError("Failed to open file: " + path); }
    out << j.dump(4);
    if (!out.good()) { throw IOError("Failed to write json: " + path); }
    spdlog::info("ColorDB saved: name={}, entries={}, path={}", name, entries.size(), path);
}

std::string ColorDB::ToJsonString() const { return DBToJson(*this).dump(4); }

const Entry& ColorDB::NearestEntry(const Lab& target) const {
    if (entries.empty()) { throw InputError("ColorDB entries is empty"); }
    EnsureKDTree();
    const auto neighbor = lab_tree_.Nearest(target);
    return entries[static_cast<std::size_t>(neighbor.index)];
}

const Entry& ColorDB::NearestEntry(const Rgb& target) const {
    if (entries.empty()) { throw InputError("ColorDB entries is empty"); }
    EnsureKDTree();
    const auto neighbor = rgb_tree_.Nearest(target);
    return entries[static_cast<std::size_t>(neighbor.index)];
}

std::vector<const Entry*> ColorDB::NearestEntries(const Lab& target, std::size_t k) const {
    if (k == 0) { return {}; }
    if (entries.empty()) { throw InputError("ColorDB entries is empty"); }

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
    if (entries.empty()) { throw InputError("ColorDB entries is empty"); }

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
    for (std::size_t i = 0; i < entries.size(); ++i) { kd_indices_.push_back(i); }

    const auto points  = std::span<const Entry>(entries);
    const auto indices = std::span<const KdIndex>(kd_indices_);
    lab_tree_.Reset(points, indices, LabProj{});
    rgb_tree_.Reset(points, indices, RgbProj{});
    kd_entries_size_ = entries.size();
    spdlog::debug("ColorDB::BuildKDTree: {} entries indexed", entries.size());
}

void ColorDB::EnsureKDTree() const {
    if (entries.size() != kd_entries_size_) { BuildKDTree(); }
}

void ColorDB::ResetKDTreeCache() const {
    kd_indices_.clear();
    lab_tree_        = LabTree();
    rgb_tree_        = RgbTree();
    kd_entries_size_ = 0;
}

} // namespace ChromaPrint3D
