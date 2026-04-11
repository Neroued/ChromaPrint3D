/// \file forward_color_model.cpp
/// \brief Beer-Lambert forward color model implementation.

#include "chromaprint3d/forward_color_model.h"
#include "chromaprint3d/error.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace ChromaPrint3D {

using nlohmann::json;

// ── Helpers ─────────────────────────────────────────────────────────────────

namespace {

std::string NormalizeLabel(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isalnum(c)) { out.push_back(static_cast<char>(std::tolower(c))); }
    }
    return out;
}

std::string NormalizeChannelKey(const std::string& value) {
    const auto delim = value.find('|');
    if (delim == std::string::npos) { return NormalizeLabel(value); }
    return NormalizeLabel(value.substr(0, delim)) + "|" + NormalizeLabel(value.substr(delim + 1));
}

std::array<float, 3> ParseFloat3(const json& arr, const char* context) {
    if (!arr.is_array() || arr.size() != 3) {
        throw FormatError(std::string("Expected 3-element array for ") + context);
    }
    return {arr[0].get<float>(), arr[1].get<float>(), arr[2].get<float>()};
}

} // namespace

// ── ForwardColorModel ───────────────────────────────────────────────────────

ForwardColorModel ForwardColorModel::LoadFromJson(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) { throw std::runtime_error("Cannot open forward model file: " + path); }

    json j;
    try {
        j = json::parse(ifs);
    } catch (const json::parse_error& e) {
        throw FormatError("JSON parse error in " + path + ": " + e.what());
    }

    ForwardColorModel model;
    model.name       = j.value("name", std::filesystem::path(path).stem().string());
    model.param_type = j.value("param_type", std::string("stage_B"));

    if (j.contains("scope") && j["scope"].is_object()) {
        model.scope.vendor        = j["scope"].value("vendor", std::string());
        model.scope.material_type = j["scope"].value("material_type", std::string());
    }
    if (j.contains("channel_keys") && j["channel_keys"].is_array()) {
        for (const auto& item : j["channel_keys"]) {
            if (item.is_string()) {
                model.channel_keys.push_back(NormalizeChannelKey(item.get<std::string>()));
            }
        }
    }

    // Meta
    const json meta             = j.value("meta", json::object());
    model.micro_layer_height_mm = meta.value("micro_layer_height_mm", model.micro_layer_height_mm);
    model.height_ref_mm         = meta.value("height_ref_mm", model.height_ref_mm);
    model.enable_height_scale   = meta.value("enable_height_scale", model.enable_height_scale);
    if (model.micro_layer_height_mm <= 0) { model.micro_layer_height_mm = 0.04f; }
    if (model.height_ref_mm <= 0) { model.height_ref_mm = 0.04f; }

    // Channels
    const json params   = j.value("parameters", json::object());
    json channels_array = params.value("channels", json::array());
    if (!channels_array.is_array() || channels_array.empty()) {
        throw FormatError("Forward model missing parameters.channels");
    }

    std::vector<json> sorted_channels(channels_array.begin(), channels_array.end());
    std::sort(sorted_channels.begin(), sorted_channels.end(), [](const json& a, const json& b) {
        return a.value("channel_index", 0) < b.value("channel_index", 0);
    });

    model.num_channels = static_cast<int>(sorted_channels.size());
    model.channel_names.reserve(model.num_channels);
    model.E.reserve(model.num_channels);
    model.k.reserve(model.num_channels);
    model.gamma.reserve(model.num_channels);

    for (int idx = 0; idx < model.num_channels; ++idx) {
        const auto& ch = sorted_channels[idx];
        model.channel_names.push_back(ch.value("color_name", "channel_" + std::to_string(idx)));
        model.E.push_back(ParseFloat3(ch.value("E", json::array({0.5, 0.5, 0.5})), "E"));
        model.k.push_back(ParseFloat3(ch.value("k", json::array({1.0, 1.0, 1.0})), "k"));
        model.gamma.push_back(ch.value("k_height_scale_gamma", 0.0f));
    }

    // Substrates
    const json substrates_array = params.value("substrates", json::array());
    if (substrates_array.is_array() && !substrates_array.empty()) {
        int max_idx = 0;
        for (const auto& item : substrates_array) {
            max_idx = std::max(max_idx, item.value("substrate_idx", 0));
        }
        model.C0.resize(max_idx + 1, {0.0f, 0.0f, 0.0f});
        model.substrate_base_channel_idx.resize(max_idx + 1, 0);

        for (const auto& item : substrates_array) {
            int sidx = item.value("substrate_idx", 0);
            model.C0[sidx] =
                ParseFloat3(item.value("C0_fitted", json::array({0.0, 0.0, 0.0})), "C0_fitted");
            model.substrate_base_channel_idx[sidx] = item.value("base_channel_stage_idx", 0);
        }
    } else {
        model.C0.push_back(model.E.empty() ? std::array<float, 3>{0.5f, 0.5f, 0.5f} : model.E[0]);
        model.substrate_base_channel_idx.push_back(0);
    }

    // Neighbor delta
    const json neighbor    = params.value("neighbor_delta", json::object());
    const json delta_array = neighbor.value("delta_k", json::array());
    const int nc           = model.num_channels;
    model.delta_k.assign(nc, std::vector<std::array<float, 3>>(nc, {0.0f, 0.0f, 0.0f}));

    if (delta_array.is_array() && !delta_array.empty()) {
        if (static_cast<int>(delta_array.size()) != nc) {
            spdlog::warn("Forward model delta_k outer dimension {} != num_channels {}",
                         delta_array.size(), nc);
        }
        for (int i = 0; i < nc && i < static_cast<int>(delta_array.size()); ++i) {
            const auto& row = delta_array[i];
            if (!row.is_array()) { continue; }
            for (int j = 0; j < nc && j < static_cast<int>(row.size()); ++j) {
                model.delta_k[i][j] = ParseFloat3(row[j], "delta_k element");
            }
        }
    }

    return model;
}

bool ForwardColorModel::MatchesScope(const std::string& vendor,
                                     const std::string& material_type) const {
    return scope.vendor == vendor && scope.material_type == material_type;
}

int ForwardColorModel::MapChannelByName(const std::string& color_name) const {
    const std::string norm = NormalizeLabel(color_name);
    for (int i = 0; i < num_channels; ++i) {
        if (NormalizeLabel(channel_names[i]) == norm) { return i; }
    }
    return -1;
}

int ForwardColorModel::ResolveSubstrateIdx(int base_channel_stage_idx) const {
    for (int i = 0; i < static_cast<int>(substrate_base_channel_idx.size()); ++i) {
        if (substrate_base_channel_idx[i] == base_channel_stage_idx) { return i; }
    }
    return 0;
}

Lab ForwardColorModel::PredictLab(std::span<const int> recipe, const PredictionConfig& cfg) const {
    if (recipe.empty()) { return Lab(0.0f, 0.0f, 0.0f); }

    const float ratio_f = cfg.layer_height_mm / micro_layer_height_mm;
    const int n_u       = static_cast<int>(std::round(ratio_f));
    if (n_u <= 0) {
        spdlog::warn("Forward model: invalid n_u={} (layer_height={}mm, micro={}mm)", n_u,
                     cfg.layer_height_mm, micro_layer_height_mm);
        return Lab(0.0f, 0.0f, 0.0f);
    }

    // Build micro-layer sequence (bottom-to-top order for the iterative model).
    const int color_layers = static_cast<int>(recipe.size());
    const int total_micro  = color_layers * n_u;
    std::vector<int> micro_bt(total_micro);

    if (cfg.layer_order == LayerOrder::Top2Bottom) {
        for (int l = 0; l < color_layers; ++l) {
            int reversed = color_layers - 1 - l;
            for (int u = 0; u < n_u; ++u) { micro_bt[l * n_u + u] = recipe[reversed]; }
        }
    } else {
        for (int l = 0; l < color_layers; ++l) {
            for (int u = 0; u < n_u; ++u) { micro_bt[l * n_u + u] = recipe[l]; }
        }
    }

    // Height scale factor
    const float h_ref   = std::max(height_ref_mm, 1e-6f);
    const float h_delta = cfg.layer_height_mm / h_ref - 1.0f;
    const float t       = micro_layer_height_mm;

    // Initial boundary color
    std::array<float, 3> C;
    if (cfg.substrate_idx >= 0 && cfg.substrate_idx < static_cast<int>(C0.size())) {
        C = C0[cfg.substrate_idx];
    } else if (cfg.base_channel_idx >= 0 && cfg.base_channel_idx < num_channels) {
        C = E[cfg.base_channel_idx];
    } else {
        C = {0.5f, 0.5f, 0.5f};
    }

    int prev = cfg.base_channel_idx;

    for (int layer = 0; layer < total_micro; ++layer) {
        int ch = micro_bt[layer];
        if (ch < 0 || ch >= num_channels) {
            prev = ch;
            continue;
        }

        float scale = 1.0f;
        if (enable_height_scale) {
            float exponent = std::clamp(gamma[ch] * h_delta, -20.0f, 20.0f);
            scale          = std::exp(exponent);
        }

        for (int c = 0; c < 3; ++c) {
            float k_scaled = k[ch][c] * scale;
            float dk       = 0.0f;
            if (prev >= 0 && prev < num_channels) { dk = delta_k[ch][prev][c]; }
            float k_eff = k_scaled + dk;
            float T     = std::exp(-std::min(k_eff * t, 50.0f));
            C[c]        = (1.0f - T) * E[ch][c] + T * C[c];
        }

        prev = ch;
    }

    // Clamp to [0,1] then convert linear RGB -> sRGB -> XYZ -> Lab
    Rgb linear_rgb(std::clamp(C[0], 0.0f, 1.0f), std::clamp(C[1], 0.0f, 1.0f),
                   std::clamp(C[2], 0.0f, 1.0f));
    return linear_rgb.ToLab();
}

// ── ForwardModelRegistry ────────────────────────────────────────────────────

void ForwardModelRegistry::LoadFromDirectory(const std::string& dir) {
    namespace fs = std::filesystem;

    if (!fs::is_directory(dir)) {
        spdlog::info("Forward model directory does not exist: {}", dir);
        return;
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto& file : files) {
        try {
            auto model = ForwardColorModel::LoadFromJson(file.string());
            if (model.scope.vendor.empty() || model.scope.material_type.empty()) {
                spdlog::warn("Forward model {} has no scope, skipping", file.string());
                continue;
            }
            spdlog::info("Loaded forward model: {} [{}::{}] type={}", model.name,
                         model.scope.vendor, model.scope.material_type, model.param_type);
            models_.push_back(std::move(model));
        } catch (const std::exception& e) {
            spdlog::warn("Skipping invalid forward model {}: {}", file.string(), e.what());
        }
    }

    if (files.empty()) { spdlog::info("No .json forward models found in {}", dir); }
}

const ForwardColorModel*
ForwardModelRegistry::Select(const std::string& vendor, const std::string& material_type,
                             const std::vector<std::string>& profile_channel_keys) const {
    if (vendor.empty() || material_type.empty()) { return nullptr; }

    const std::unordered_set<std::string> profile_keys(profile_channel_keys.begin(),
                                                       profile_channel_keys.end());

    const ForwardColorModel* best = nullptr;
    size_t best_channel_count     = 0;

    for (const auto& model : models_) {
        if (!model.MatchesScope(vendor, material_type)) { continue; }

        bool all_keys_found = true;
        for (const auto& key : model.channel_keys) {
            if (profile_keys.find(key) == profile_keys.end()) {
                all_keys_found = false;
                break;
            }
        }
        if (!all_keys_found) { continue; }

        if (model.channel_keys.size() > best_channel_count) {
            best               = &model;
            best_channel_count = model.channel_keys.size();
        }
    }

    return best;
}

} // namespace ChromaPrint3D
