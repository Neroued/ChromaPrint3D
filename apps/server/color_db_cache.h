#pragma once

#include "chromaprint3d/color_db.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ChromaPrint3D;

struct ColorDBCache {
    std::vector<ColorDB> databases;
    std::unordered_map<std::string, size_t> name_to_index;

    void LoadFromDirectory(const std::string& dir) {
        if (!std::filesystem::is_directory(dir)) {
            throw std::runtime_error("Data directory does not exist: " + dir);
        }
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file()) { continue; }
            if (entry.path().extension() != ".json") { continue; }
            try {
                ColorDB db       = ColorDB::LoadFromJson(entry.path().string());
                std::string name = db.name;
                size_t idx       = databases.size();
                databases.push_back(std::move(db));
                name_to_index[name] = idx;
                spdlog::info("  Loaded ColorDB: {} ({} entries, {} channels)", name,
                             databases.back().entries.size(), databases.back().NumChannels());
            } catch (const std::exception& e) {
                spdlog::warn("  Failed to load {}: {}", entry.path().string(), e.what());
            }
        }
        if (databases.empty()) { throw std::runtime_error("No ColorDB files found in: " + dir); }
    }

    const ColorDB* FindByName(const std::string& name) const {
        auto it = name_to_index.find(name);
        if (it == name_to_index.end()) { return nullptr; }
        return &databases[it->second];
    }

    std::vector<const ColorDB*> GetAll() const {
        std::vector<const ColorDB*> result;
        result.reserve(databases.size());
        for (const auto& db : databases) { result.push_back(&db); }
        return result;
    }
};
