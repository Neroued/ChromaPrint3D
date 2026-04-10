#include "chromaprint3d/model_package.h"
#include "chromaprint3d/error.h"
#include "detail/match_utils.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

namespace ChromaPrint3D {

using detail::NormalizeChannelKeyString;
using detail::ParseLayerOrderValue;
using nlohmann::json;

size_t ModelLayerPackage::NumCandidates() const { return pred_lab.size(); }

const uint8_t* ModelLayerPackage::RecipeAt(size_t idx) const {
    if (idx >= NumCandidates() || color_layers <= 0) { return nullptr; }
    const size_t offset = idx * static_cast<size_t>(color_layers);
    if (offset + static_cast<size_t>(color_layers) > candidate_recipes.size()) { return nullptr; }
    return &candidate_recipes[offset];
}

const ModelLayerPackage* ModelPackage::FindByColorLayers(int target_color_layers) const {
    for (const ModelLayerPackage& item : layer_packages) {
        if (item.color_layers == target_color_layers) { return &item; }
    }
    return nullptr;
}

ModelPackage ModelPackage::LoadFromJson(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) { throw IOError("Failed to open model package: " + path); }

    json j;
    in >> j;

    ModelPackage pkg;
    pkg.name = j.value("name", std::string("PhaseA_ModelPackage"));

    const json defaults   = j.value("defaults", json::object());
    pkg.default_threshold = defaults.value("threshold", pkg.default_threshold);
    pkg.default_margin    = defaults.value("margin", pkg.default_margin);

    const json channel_keys = j.value("channel_keys", json::array());
    if (!channel_keys.is_array() || channel_keys.empty()) {
        throw FormatError("Model package missing channel_keys array");
    }
    pkg.channel_keys.reserve(channel_keys.size());
    for (const auto& item : channel_keys) {
        if (!item.is_string()) { throw FormatError("channel_keys item must be string"); }
        pkg.channel_keys.push_back(NormalizeChannelKeyString(item.get<std::string>()));
    }

    const json modes_j = j.value("modes", json::object());
    if (!modes_j.is_object() || modes_j.empty()) {
        throw FormatError("Model package missing modes object");
    }

    for (const auto& [mode_key, mode_value] : modes_j.items()) {
        if (!mode_value.is_object()) {
            throw FormatError("Model package mode value must be object");
        }
        ModelLayerPackage lpkg;
        lpkg.layer_height_mm = mode_value.value("layer_height_mm", 0.0f);
        lpkg.color_layers    = mode_value.value("color_layers", 0);
        lpkg.layer_order = ParseLayerOrderValue(mode_value.value("layer_order", json("Top2Bottom")),
                                                LayerOrder::Top2Bottom);
        lpkg.base_channel_key =
            NormalizeChannelKeyString(mode_value.value("base_channel_key", std::string("")));

        // Backward compatibility: derive from legacy mode_key if fields missing.
        if (lpkg.layer_height_mm <= 0.0f || lpkg.color_layers <= 0) {
            if (mode_key == "0.08x5" || mode_key == "0p08x5") {
                lpkg.layer_height_mm = 0.08f;
                lpkg.color_layers    = 5;
            } else if (mode_key == "0.04x10" || mode_key == "0p04x10") {
                lpkg.layer_height_mm = 0.04f;
                lpkg.color_layers    = 10;
            } else {
                throw FormatError("Model package mode has invalid layer config: " + mode_key);
            }
        }

        const json recipes_j = mode_value.value("candidate_recipes", json::array());
        const json labs_j    = mode_value.value("pred_lab", json::array());
        if (!recipes_j.is_array() || !labs_j.is_array()) {
            throw FormatError("Model package mode requires candidate_recipes and pred_lab arrays");
        }
        if (recipes_j.size() != labs_j.size()) {
            throw FormatError("Model package candidate_recipes size != pred_lab size");
        }

        lpkg.candidate_recipes.reserve(recipes_j.size() * static_cast<size_t>(lpkg.color_layers));
        lpkg.pred_lab.reserve(labs_j.size());

        for (size_t i = 0; i < recipes_j.size(); ++i) {
            const json& recipe_j = recipes_j[i];
            const json& lab_j    = labs_j[i];
            if (!recipe_j.is_array() || recipe_j.size() != static_cast<size_t>(lpkg.color_layers)) {
                throw FormatError("Model package recipe length mismatch for mode: " + mode_key);
            }
            if (!lab_j.is_array() || lab_j.size() != 3) {
                throw FormatError("Model package pred_lab item must be [L,a,b]");
            }

            for (const auto& v : recipe_j) {
                const int ch = v.get<int>();
                if (ch < 0 || ch > 255) {
                    throw FormatError("Model package recipe channel out of uint8 range");
                }
                lpkg.candidate_recipes.push_back(static_cast<uint8_t>(ch));
            }
            lpkg.pred_lab.emplace_back(lab_j.at(0).get<float>(), lab_j.at(1).get<float>(),
                                       lab_j.at(2).get<float>());
        }

        if (lpkg.NumCandidates() == 0) {
            throw FormatError("Model package mode has zero candidates: " + mode_key);
        }
        pkg.layer_packages.push_back(std::move(lpkg));
    }

    return pkg;
}

} // namespace ChromaPrint3D
