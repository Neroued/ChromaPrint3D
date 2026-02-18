#pragma once

#include "color_db_cache.h"
#include "session.h"

#include "chromaprint3d/pipeline.h"
#include "chromaprint3d/print_profile.h"
#include "chromaprint3d/model_package.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

using namespace ChromaPrint3D;
using json = nlohmann::json;

inline PrintMode ParsePrintMode(const std::string& value) {
    if (value == "0.08x5" || value == "0p08x5") { return PrintMode::Mode0p08x5; }
    if (value == "0.04x10" || value == "0p04x10") { return PrintMode::Mode0p04x10; }
    throw std::runtime_error("Invalid print_mode: " + value);
}

inline ColorSpace ParseColorSpace(const std::string& value) {
    if (value == "lab" || value == "Lab" || value == "LAB") { return ColorSpace::Lab; }
    if (value == "rgb" || value == "Rgb" || value == "RGB") { return ColorSpace::Rgb; }
    throw std::runtime_error("Invalid color_space: " + value);
}

inline ConvertRequest BuildConvertRequest(const json& params,
                                          const std::vector<uint8_t>& image_buffer,
                                          const std::string& image_name,
                                          const ColorDBCache& db_cache,
                                          const ModelPackage* model_pack,
                                          UserSession* session) {
    ConvertRequest req;

    // Image
    req.image_buffer = image_buffer;
    req.image_name   = image_name;

    // ColorDB selection
    if (params.contains("db_names") && params["db_names"].is_array()) {
        std::vector<const ColorDB*> selected;
        for (const auto& name_val : params["db_names"]) {
            std::string name  = name_val.get<std::string>();
            const ColorDB* db = db_cache.FindByName(name);
            if (!db && session) {
                auto it = session->color_dbs.find(name);
                if (it != session->color_dbs.end()) { db = &it->second; }
            }
            if (!db) { throw std::runtime_error("ColorDB not found: " + name); }
            selected.push_back(db);
        }
        if (selected.empty()) { throw std::runtime_error("No valid ColorDB names provided"); }
        req.preloaded_dbs = std::move(selected);
    } else {
        req.preloaded_dbs = db_cache.GetAll();
    }

    // Model package
    if (model_pack) { req.preloaded_model_pack = model_pack; }

    // Image processing params
    if (params.contains("scale")) {
        req.scale = params["scale"].get<float>();
        if (req.scale <= 0.0f) { throw std::runtime_error("scale must be > 0"); }
    }
    if (params.contains("max_width")) {
        req.max_width = params["max_width"].get<int>();
        if (req.max_width < 0) { throw std::runtime_error("max_width must be >= 0"); }
    }
    if (params.contains("max_height")) {
        req.max_height = params["max_height"].get<int>();
        if (req.max_height < 0) { throw std::runtime_error("max_height must be >= 0"); }
    }

    // Matching params
    if (params.contains("print_mode")) {
        req.print_mode = ParsePrintMode(params["print_mode"].get<std::string>());
    }
    if (params.contains("color_space")) {
        req.color_space = ParseColorSpace(params["color_space"].get<std::string>());
    }
    if (params.contains("k_candidates")) {
        req.k_candidates = params["k_candidates"].get<int>();
        if (req.k_candidates < 1) { throw std::runtime_error("k_candidates must be >= 1"); }
    }
    if (params.contains("cluster_count")) {
        req.cluster_count = params["cluster_count"].get<int>();
        if (req.cluster_count < 0) { throw std::runtime_error("cluster_count must be >= 0"); }
    }
    if (params.contains("allowed_channels") && params["allowed_channels"].is_array()) {
        for (const auto& ch : params["allowed_channels"]) {
            std::string color    = ch.value("color", "");
            std::string material = ch.value("material", "");
            if (!color.empty()) {
                req.allowed_channel_keys.push_back(color + "|" + material);
            }
        }
    }

    // Model gate
    if (params.contains("model_enable")) { req.model_enable = params["model_enable"].get<bool>(); }
    if (params.contains("model_only")) { req.model_only = params["model_only"].get<bool>(); }
    if (params.contains("model_threshold")) {
        req.model_threshold = params["model_threshold"].get<float>();
    }
    if (params.contains("model_margin")) { req.model_margin = params["model_margin"].get<float>(); }

    // Geometry
    if (params.contains("flip_y")) { req.flip_y = params["flip_y"].get<bool>(); }
    if (params.contains("pixel_mm")) { req.pixel_mm = params["pixel_mm"].get<float>(); }
    if (params.contains("layer_height_mm")) {
        req.layer_height_mm = params["layer_height_mm"].get<float>();
    }

    // Output flags
    if (params.contains("generate_preview")) {
        req.generate_preview = params["generate_preview"].get<bool>();
    }
    if (params.contains("generate_source_mask")) {
        req.generate_source_mask = params["generate_source_mask"].get<bool>();
    }

    return req;
}
