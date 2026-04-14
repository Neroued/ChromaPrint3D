#pragma once

#include "infrastructure/color_db_bytes.h"

#include "chromaprint3d/color_db.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ChromaPrint3D;

struct ColorDBEntry {
    std::shared_ptr<const ColorDB> db;
    std::string material_type; // "PLA", "PETG", ... (from directory structure)
    std::string vendor;        // "BambuLab", "Aliz", ... (from directory structure)
};

struct ColorDBCache {
    std::vector<ColorDBEntry> databases;
    std::unordered_map<std::string, size_t> name_to_index;
    std::size_t estimated_total_bytes = 0;

    void LoadFromDirectory(const std::string& dir) {
        namespace fs = std::filesystem;

        if (!fs::is_directory(dir)) {
            throw std::runtime_error("Data directory does not exist: " + dir);
        }

        const fs::path root(dir);
        databases.clear();
        name_to_index.clear();
        estimated_total_bytes = 0;

        for (const auto& entry : fs::recursive_directory_iterator(root)) {
            if (!entry.is_regular_file()) { continue; }
            if (entry.path().extension() != ".json") { continue; }

            auto [material, vendor] = InferMaterialVendor(root, entry.path());

            try {
                auto db =
                    std::make_shared<const ColorDB>(ColorDB::LoadFromJson(entry.path().string()));
                std::string name = db->name;
                size_t idx       = databases.size();

                databases.push_back({db, std::move(material), std::move(vendor)});
                name_to_index[name] = idx;
                estimated_total_bytes += chromaprint3d::backend::detail::EstimateColorDbBytes(*db);

                const auto& e = databases.back();
                spdlog::info("  Loaded ColorDB: {} ({} entries, {} ch) [{}::{}]", name,
                             e.db->entries.size(), e.db->NumChannels(), e.material_type, e.vendor);
            } catch (const std::exception& e) {
                spdlog::warn("  Failed to load {}: {}", entry.path().string(), e.what());
            }
        }
        if (databases.empty()) { throw std::runtime_error("No ColorDB files found in: " + dir); }
    }

    const ColorDBEntry* FindEntryByName(const std::string& name) const {
        auto it = name_to_index.find(name);
        if (it == name_to_index.end()) { return nullptr; }
        return &databases[it->second];
    }

    std::shared_ptr<const ColorDB> FindByName(const std::string& name) const {
        const auto* entry = FindEntryByName(name);
        return entry ? entry->db : nullptr;
    }

    std::vector<std::shared_ptr<const ColorDB>> GetAll() const {
        std::vector<std::shared_ptr<const ColorDB>> result;
        result.reserve(databases.size());
        for (const auto& e : databases) { result.push_back(e.db); }
        return result;
    }

    std::size_t EstimateTotalBytes() const { return estimated_total_bytes; }

private:
    static std::pair<std::string, std::string>
    InferMaterialVendor(const std::filesystem::path& root, const std::filesystem::path& file) {
        auto rel = std::filesystem::relative(file.parent_path(), root);
        std::vector<std::string> parts;
        for (const auto& seg : rel) {
            std::string s = seg.string();
            if (s == "." || s.empty()) { continue; }
            parts.push_back(s);
        }
        if (parts.size() >= 2) { return {parts[0], parts[1]}; }
        if (parts.size() == 1) { return {parts[0], "Unknown"}; }
        return {"Unknown", "Unknown"};
    }
};
