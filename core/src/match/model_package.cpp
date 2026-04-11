#include "chromaprint3d/model_package.h"
#include "chromaprint3d/error.h"
#include "detail/match_utils.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>

static_assert(std::endian::native == std::endian::little,
              "ModelPackage binary format requires a little-endian platform");

namespace ChromaPrint3D {

using detail::NormalizeChannelKeyString;
using detail::ParseLayerOrderValue;
using nlohmann::json;

// ── ModelLayerPackage ────────────────────────────────────────────────────────

size_t ModelLayerPackage::NumCandidates() const { return pred_lab.size(); }

const uint8_t* ModelLayerPackage::RecipeAt(size_t idx) const {
    if (idx >= NumCandidates() || color_layers <= 0) { return nullptr; }
    const size_t offset = idx * static_cast<size_t>(color_layers);
    if (offset + static_cast<size_t>(color_layers) > candidate_recipes.size()) { return nullptr; }
    return &candidate_recipes[offset];
}

// ── ModelPackage ─────────────────────────────────────────────────────────────

const ModelLayerPackage* ModelPackage::FindByColorLayers(int target_color_layers) const {
    auto it = mode_index_.find(target_color_layers);
    if (it == mode_index_.end()) { return nullptr; }
    return &layer_packages[it->second];
}

bool ModelPackage::MatchesScope(const std::string& vendor, const std::string& material_type) const {
    return scope.vendor == vendor && scope.material_type == material_type;
}

void ModelPackage::BuildIndex() {
    mode_index_.clear();
    for (size_t i = 0; i < layer_packages.size(); ++i) {
        mode_index_.emplace(layer_packages[i].color_layers, i);
    }
}

ModelPackage ModelPackage::Load(const std::string& path) {
    namespace fs = std::filesystem;

    const fs::path file_path(path);
    if (file_path.extension() == ".json") {
        throw FormatError("JSON model package format is no longer supported; "
                          "regenerate as .msgpack with step5_build_model_package.py");
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) { throw IOError("Failed to open model package: " + path); }

    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());
    json j = json::from_msgpack(bytes);

    const int schema_ver = j.value("schema_version", 0);
    if (schema_ver != kSchemaVersion) {
        throw FormatError("Unsupported model package schema version (" +
                          std::to_string(schema_ver) + "); regenerate with step5");
    }

    ModelPackage pkg;
    pkg.name = j.value("name", std::string("ModelPackage"));

    const json scope_j      = j.value("scope", json::object());
    pkg.scope.vendor        = scope_j.value("vendor", std::string());
    pkg.scope.material_type = scope_j.value("material_type", std::string());
    if (pkg.scope.vendor.empty() || pkg.scope.material_type.empty()) {
        throw FormatError("Model package scope requires non-empty vendor and material_type");
    }

    const json defaults   = j.value("defaults", json::object());
    pkg.default_threshold = defaults.value("threshold", pkg.default_threshold);
    pkg.default_margin    = defaults.value("margin", pkg.default_margin);

    const json channel_keys_j = j.value("channel_keys", json::array());
    if (!channel_keys_j.is_array() || channel_keys_j.empty()) {
        throw FormatError("Model package missing channel_keys array");
    }
    pkg.channel_keys.reserve(channel_keys_j.size());
    for (const auto& item : channel_keys_j) {
        if (!item.is_string()) { throw FormatError("channel_keys item must be string"); }
        pkg.channel_keys.push_back(NormalizeChannelKeyString(item.get<std::string>()));
    }

    const json modes_j = j.value("modes", json::array());
    if (!modes_j.is_array() || modes_j.empty()) {
        throw FormatError("Model package missing modes array");
    }

    for (size_t mi = 0; mi < modes_j.size(); ++mi) {
        const json& mode_value = modes_j[mi];
        if (!mode_value.is_object()) {
            throw FormatError("Model package mode[" + std::to_string(mi) + "] must be object");
        }

        ModelLayerPackage lpkg;
        lpkg.color_layers    = mode_value.value("color_layers", 0);
        lpkg.layer_height_mm = mode_value.value("layer_height_mm", 0.0f);
        if (lpkg.color_layers <= 0 || lpkg.layer_height_mm <= 0.0f) {
            throw FormatError("Model package mode[" + std::to_string(mi) +
                              "] requires positive color_layers and layer_height_mm");
        }

        lpkg.layer_order = ParseLayerOrderValue(mode_value.value("layer_order", json("Top2Bottom")),
                                                LayerOrder::Top2Bottom);
        lpkg.base_channel_key =
            NormalizeChannelKeyString(mode_value.value("base_channel_key", std::string()));

        const int num_candidates = mode_value.value("num_candidates", 0);
        if (num_candidates <= 0) {
            throw FormatError("Model package mode[" + std::to_string(mi) +
                              "] requires positive num_candidates");
        }

        // Deserialize candidate_recipes from binary blob
        if (!mode_value.contains("candidate_recipes") ||
            !mode_value["candidate_recipes"].is_binary()) {
            throw FormatError("Model package mode[" + std::to_string(mi) +
                              "] candidate_recipes must be a binary blob");
        }
        const auto& recipes_bin = mode_value["candidate_recipes"].get_binary();
        const size_t expected_recipes_size =
            static_cast<size_t>(num_candidates) * static_cast<size_t>(lpkg.color_layers);
        if (recipes_bin.size() != expected_recipes_size) {
            throw FormatError("Model package mode[" + std::to_string(mi) +
                              "] candidate_recipes blob size mismatch: got " +
                              std::to_string(recipes_bin.size()) + ", expected " +
                              std::to_string(expected_recipes_size));
        }
        lpkg.candidate_recipes.assign(recipes_bin.begin(), recipes_bin.end());

        // Deserialize pred_lab from binary blob via memcpy
        if (!mode_value.contains("pred_lab") || !mode_value["pred_lab"].is_binary()) {
            throw FormatError("Model package mode[" + std::to_string(mi) +
                              "] pred_lab must be a binary blob");
        }
        const auto& lab_bin            = mode_value["pred_lab"].get_binary();
        const size_t expected_lab_size = static_cast<size_t>(num_candidates) * 3 * sizeof(float);
        if (lab_bin.size() != expected_lab_size) {
            throw FormatError(
                "Model package mode[" + std::to_string(mi) + "] pred_lab blob size mismatch: got " +
                std::to_string(lab_bin.size()) + ", expected " + std::to_string(expected_lab_size));
        }
        lpkg.pred_lab.resize(static_cast<size_t>(num_candidates));
        std::memcpy(lpkg.pred_lab.data(), lab_bin.data(), lab_bin.size());

        // Uniqueness check
        const size_t idx   = pkg.layer_packages.size();
        auto [_, inserted] = pkg.mode_index_.emplace(lpkg.color_layers, idx);
        if (!inserted) {
            throw FormatError("Duplicate color_layers=" + std::to_string(lpkg.color_layers) +
                              " in model package");
        }

        pkg.layer_packages.push_back(std::move(lpkg));
    }

    return pkg;
}

// ── ModelPackageRegistry ─────────────────────────────────────────────────────

void ModelPackageRegistry::LoadFromDirectory(const std::string& dir) {
    namespace fs = std::filesystem;

    if (!fs::is_directory(dir)) {
        spdlog::info("Model pack directory does not exist: {}", dir);
        return;
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".msgpack") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto& file : files) {
        try {
            packages_.push_back(ModelPackage::Load(file.string()));
            spdlog::info("Loaded model package: {} [{}::{}]", packages_.back().name,
                         packages_.back().scope.vendor, packages_.back().scope.material_type);
        } catch (const std::exception& e) {
            spdlog::warn("Skipping invalid model package {}: {}", file.string(), e.what());
        }
    }

    if (files.empty()) { spdlog::info("No .msgpack model packs found in {}", dir); }
}

void ModelPackageRegistry::LoadSingle(const std::string& path) {
    packages_.push_back(ModelPackage::Load(path));
    spdlog::info("Loaded model package: {} [{}::{}]", packages_.back().name,
                 packages_.back().scope.vendor, packages_.back().scope.material_type);
}

const ModelPackage*
ModelPackageRegistry::Select(const std::string& vendor, const std::string& material_type,
                             const std::vector<std::string>& profile_channel_keys) const {
    if (vendor.empty() || material_type.empty()) { return nullptr; }

    const std::unordered_set<std::string> profile_keys(profile_channel_keys.begin(),
                                                       profile_channel_keys.end());

    const ModelPackage* best  = nullptr;
    size_t best_channel_count = 0;

    for (const auto& pack : packages_) {
        if (!pack.MatchesScope(vendor, material_type)) { continue; }

        bool all_keys_found = true;
        for (const auto& key : pack.channel_keys) {
            if (profile_keys.find(key) == profile_keys.end()) {
                all_keys_found = false;
                break;
            }
        }
        if (!all_keys_found) { continue; }

        if (pack.channel_keys.size() > best_channel_count) {
            best               = &pack;
            best_channel_count = pack.channel_keys.size();
        }
    }

    return best;
}

} // namespace ChromaPrint3D
