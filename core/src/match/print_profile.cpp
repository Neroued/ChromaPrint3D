#include "chromaprint3d/print_profile.h"
#include "chromaprint3d/color_db.h"
#include "chromaprint3d/error.h"
#include "detail/match_utils.h"

#include <spdlog/spdlog.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace ChromaPrint3D {

using detail::BuildChannelKey;
using detail::kTargetColorThicknessMm;
using detail::ModeSpec;
using detail::NearlyEqual;
using detail::NormalizeChannelKeyString;

void PrintProfile::Validate() const {
    if (palette.empty()) { throw ConfigError("PrintProfile palette is empty"); }
    if (palette.size() > static_cast<std::size_t>(std::numeric_limits<uint8_t>::max())) {
        throw ConfigError("PrintProfile palette exceeds uint8 recipe range");
    }
    if (color_layers <= 0) {
        throw ConfigError("PrintProfile color_layers must be positive");
    }
    if (layer_height_mm <= 0.0f) {
        throw ConfigError("PrintProfile layer_height_mm must be positive");
    }
    if (!NearlyEqual(max_color_thickness_mm, kTargetColorThicknessMm)) {
        throw ConfigError("PrintProfile max_color_thickness_mm must be 0.4");
    }
    if (!NearlyEqual(layer_height_mm * static_cast<float>(color_layers), max_color_thickness_mm)) {
        throw ConfigError("PrintProfile layer_height_mm * color_layers must equal 0.4");
    }
    const auto [mode_layer_height, mode_layers] = ModeSpec(mode);
    if (!NearlyEqual(layer_height_mm, mode_layer_height) || color_layers != mode_layers) {
        throw ConfigError("PrintProfile mode does not match layer_height_mm/color_layers");
    }
    if (line_width_mm <= 0.0f) {
        throw ConfigError("PrintProfile line_width_mm must be positive");
    }
    if (base_layers < 0) { throw ConfigError("PrintProfile base_layers must be >= 0"); }
    if (base_channel_idx < 0 || static_cast<std::size_t>(base_channel_idx) >= palette.size()) {
        throw ConfigError("PrintProfile base_channel_idx out of range");
    }
}

ColorDB PrintProfile::ToColorDB(const std::string& name) const {
    Validate();
    ColorDB out;
    out.name             = name;
    out.max_color_layers = color_layers;
    out.base_layers      = base_layers;
    out.base_channel_idx = base_channel_idx;
    out.layer_height_mm  = layer_height_mm;
    out.line_width_mm    = line_width_mm;
    out.layer_order      = layer_order;
    out.palette          = palette;
    return out;
}

PrintProfile PrintProfile::BuildFromColorDBs(std::span<const ColorDB> dbs, PrintMode mode) {
    if (dbs.empty()) {
        throw InputError("BuildFromColorDBs requires at least one ColorDB");
    }

    PrintProfile profile;
    profile.mode                                            = mode;
    profile.max_color_thickness_mm                          = kTargetColorThicknessMm;
    std::tie(profile.layer_height_mm, profile.color_layers) = ModeSpec(mode);

    profile.base_layers   = dbs.front().base_layers;
    profile.line_width_mm = (dbs.front().line_width_mm > 0.0f) ? dbs.front().line_width_mm : 0.42f;
    profile.layer_order   = dbs.front().layer_order;

    std::optional<std::string> base_channel_key;
    std::map<std::string, Channel> sorted_palette;

    for (const ColorDB& db : dbs) {
        if (db.palette.empty()) { throw ConfigError("ColorDB palette is empty"); }
        if (db.base_channel_idx < 0 ||
            static_cast<std::size_t>(db.base_channel_idx) >= db.palette.size()) {
            throw ConfigError("ColorDB base_channel_idx out of range");
        }
        if (db.base_layers != profile.base_layers) {
            throw ConfigError("base_layers mismatch across ColorDBs");
        }
        if (db.line_width_mm > 0.0f && !NearlyEqual(db.line_width_mm, profile.line_width_mm)) {
            throw ConfigError("line_width_mm mismatch across ColorDBs");
        }

        for (const Channel& channel : db.palette) {
            const std::string key = BuildChannelKey(channel);
            sorted_palette.try_emplace(key, channel);
        }

        const std::string db_base_key =
            BuildChannelKey(db.palette[static_cast<std::size_t>(db.base_channel_idx)]);
        if (!base_channel_key.has_value()) { base_channel_key = db_base_key; }
    }

    if (sorted_palette.empty()) { throw ConfigError("Merged palette is empty"); }

    profile.palette.clear();
    profile.palette.reserve(sorted_palette.size());
    profile.base_channel_idx = -1;
    int idx                  = 0;
    for (const auto& [key, channel] : sorted_palette) {
        profile.palette.push_back(channel);
        if (base_channel_key.has_value() && key == *base_channel_key) {
            profile.base_channel_idx = idx;
        }
        ++idx;
    }
    if (profile.base_channel_idx < 0) {
        throw ConfigError("Failed to resolve base_channel_idx in merged palette");
    }

    profile.Validate();
    return profile;
}

void PrintProfile::FilterChannels(const std::vector<std::string>& allowed_keys) {
    if (allowed_keys.empty()) { return; }

    std::set<std::string> allowed_set;
    for (const std::string& key : allowed_keys) {
        allowed_set.insert(NormalizeChannelKeyString(key));
    }

    if (base_channel_idx >= 0 && static_cast<std::size_t>(base_channel_idx) < palette.size()) {
        allowed_set.insert(BuildChannelKey(palette[static_cast<std::size_t>(base_channel_idx)]));
    }

    std::vector<Channel> filtered;
    int new_base_idx = -1;
    for (std::size_t i = 0; i < palette.size(); ++i) {
        const std::string key = BuildChannelKey(palette[i]);
        if (allowed_set.count(key)) {
            if (static_cast<int>(i) == base_channel_idx) {
                new_base_idx = static_cast<int>(filtered.size());
            }
            filtered.push_back(palette[i]);
        }
    }

    if (filtered.empty()) {
        throw ConfigError("FilterChannels: no channels remaining after filtering");
    }

    palette          = std::move(filtered);
    base_channel_idx = new_base_idx;
    Validate();
}

} // namespace ChromaPrint3D
