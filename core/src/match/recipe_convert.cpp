#include "detail/recipe_convert.h"
#include "detail/match_utils.h"
#include "chromaprint3d/error.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ChromaPrint3D {
namespace detail {

using detail::BuildChannelKey;
using detail::NearlyEqual;

bool ConvertRecipeToProfile(const Entry& entry, const PreparedDB& prepared_db,
                            const PrintProfile& profile, std::vector<uint8_t>& out_recipe) {
    if (!prepared_db.db) { return false; }
    if (entry.recipe.empty()) { return false; }
    const ColorDB& db = *prepared_db.db;

    std::vector<int> ordered_recipe(entry.recipe.begin(), entry.recipe.end());
    if (db.layer_order != profile.layer_order) {
        std::reverse(ordered_recipe.begin(), ordered_recipe.end());
    }

    if (db.layer_height_mm <= 0.0f || profile.layer_height_mm <= 0.0f) { return false; }

    std::vector<int> profile_layers_recipe;
    profile_layers_recipe.reserve(static_cast<std::size_t>(profile.color_layers));

    if (NearlyEqual(db.layer_height_mm, profile.layer_height_mm)) {
        if (static_cast<int>(ordered_recipe.size()) != profile.color_layers) { return false; }
        profile_layers_recipe = std::move(ordered_recipe);
    } else if (db.layer_height_mm > profile.layer_height_mm) {
        const float ratio_f = db.layer_height_mm / profile.layer_height_mm;
        const int ratio     = static_cast<int>(std::lround(ratio_f));
        if (ratio <= 0 || !NearlyEqual(ratio_f, static_cast<float>(ratio))) { return false; }
        if (static_cast<int>(ordered_recipe.size()) * ratio != profile.color_layers) {
            return false;
        }
        for (int channel_idx : ordered_recipe) {
            for (int i = 0; i < ratio; ++i) { profile_layers_recipe.push_back(channel_idx); }
        }
    } else {
        const float ratio_f = profile.layer_height_mm / db.layer_height_mm;
        const int ratio     = static_cast<int>(std::lround(ratio_f));
        if (ratio <= 0 || !NearlyEqual(ratio_f, static_cast<float>(ratio))) { return false; }
        if (static_cast<int>(ordered_recipe.size()) != profile.color_layers * ratio) {
            return false;
        }

        for (int i = 0; i < profile.color_layers; ++i) {
            const int begin = i * ratio;
            const int ref   = ordered_recipe[static_cast<std::size_t>(begin)];
            for (int j = 1; j < ratio; ++j) {
                if (ordered_recipe[static_cast<std::size_t>(begin + j)] != ref) { return false; }
            }
            profile_layers_recipe.push_back(ref);
        }
    }

    out_recipe.resize(static_cast<std::size_t>(profile.color_layers), 0);
    const std::size_t src_channels_size = prepared_db.source_to_target_channel.size();
    const std::size_t target_channels   = profile.NumChannels();

    for (int i = 0; i < profile.color_layers; ++i) {
        const int src_channel = profile_layers_recipe[static_cast<std::size_t>(i)];
        if (src_channel < 0 || static_cast<std::size_t>(src_channel) >= src_channels_size) {
            return false;
        }
        const int mapped_channel =
            prepared_db.source_to_target_channel[static_cast<std::size_t>(src_channel)];
        if (mapped_channel < 0 || static_cast<std::size_t>(mapped_channel) >= target_channels) {
            return false;
        }
        out_recipe[static_cast<std::size_t>(i)] = static_cast<uint8_t>(mapped_channel);
    }
    return true;
}

std::vector<PreparedDB> PrepareDBs(std::span<const ColorDB> dbs, const PrintProfile& profile) {
    std::unordered_map<std::string, int> key_to_target_channel;
    key_to_target_channel.reserve(profile.palette.size());
    for (std::size_t i = 0; i < profile.palette.size(); ++i) {
        const std::string key = BuildChannelKey(profile.palette[i]);
        auto [it, inserted]   = key_to_target_channel.emplace(key, static_cast<int>(i));
        if (!inserted) {
            throw ConfigError("PrintProfile palette contains duplicated normalized channel");
        }
    }

    std::vector<PreparedDB> prepared;
    prepared.reserve(dbs.size());
    for (const ColorDB& db : dbs) {
        if (db.entries.empty() || db.NumChannels() == 0) { continue; }
        if (db.palette.size() != db.NumChannels()) {
            throw ConfigError("ColorDB palette size does not match NumChannels");
        }
        PreparedDB item;
        item.db = &db;
        item.source_to_target_channel.assign(db.NumChannels(), -1);
        bool has_unmapped = false;
        for (std::size_t i = 0; i < db.palette.size(); ++i) {
            const std::string key = BuildChannelKey(db.palette[i]);
            auto it               = key_to_target_channel.find(key);
            if (it != key_to_target_channel.end()) {
                item.source_to_target_channel[i] = it->second;
            } else {
                has_unmapped = true;
            }
        }

        if (has_unmapped) {
            auto fdb              = std::make_unique<ColorDB>();
            fdb->name             = db.name + "_filtered";
            fdb->max_color_layers = db.max_color_layers;
            fdb->base_layers      = db.base_layers;
            fdb->base_channel_idx = db.base_channel_idx;
            fdb->layer_height_mm  = db.layer_height_mm;
            fdb->line_width_mm    = db.line_width_mm;
            fdb->layer_order      = db.layer_order;
            fdb->palette          = db.palette;

            fdb->entries.reserve(db.entries.size());
            std::vector<uint8_t> dummy_recipe;
            for (const Entry& entry : db.entries) {
                if (ConvertRecipeToProfile(entry, item, profile, dummy_recipe)) {
                    fdb->entries.push_back(entry);
                }
            }

            if (!fdb->entries.empty()) {
                spdlog::info("PrepareDBs: filtered '{}' from {} to {} entries", db.name,
                             db.entries.size(), fdb->entries.size());
                item.filtered_db = std::move(fdb);
            } else {
                spdlog::warn("PrepareDBs: '{}' has 0 compatible entries after channel filtering",
                             db.name);
            }
        }

        prepared.push_back(std::move(item));
    }
    if (prepared.empty()) { throw ConfigError("No usable ColorDB after preparation"); }
    return prepared;
}

} // namespace detail
} // namespace ChromaPrint3D
