#pragma once

#include "chromaprint3d/color_db.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ChromaPrint3D;

struct ColorDBEntry {
    ColorDB db;
    std::string material_type; // "PLA", "PETG", ... (from directory structure)
    std::string vendor;        // "BambuLab", "Aliz", ... (from directory structure)
};

struct ColorDBCache {
    std::vector<ColorDBEntry> databases;
    std::unordered_map<std::string, size_t> name_to_index;

    void LoadFromDirectory(const std::string& dir) {
        namespace fs = std::filesystem;

        if (!fs::is_directory(dir)) {
            throw std::runtime_error("Data directory does not exist: " + dir);
        }

        const fs::path root(dir);

        for (const auto& entry : fs::recursive_directory_iterator(root)) {
            if (!entry.is_regular_file()) { continue; }
            if (entry.path().extension() != ".json") { continue; }

            auto [material, vendor] = InferMaterialVendor(root, entry.path());

            try {
                ColorDB db       = ColorDB::LoadFromJson(entry.path().string());
                std::string name = db.name;
                size_t idx       = databases.size();

                databases.push_back({std::move(db), std::move(material), std::move(vendor)});
                name_to_index[name] = idx;

                const auto& e = databases.back();
                spdlog::info("  Loaded ColorDB: {} ({} entries, {} ch) [{}::{}]", name,
                             e.db.entries.size(), e.db.NumChannels(), e.material_type, e.vendor);
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

    const ColorDB* FindByName(const std::string& name) const {
        const auto* entry = FindEntryByName(name);
        return entry ? &entry->db : nullptr;
    }

    std::vector<const ColorDB*> GetAll() const {
        std::vector<const ColorDB*> result;
        result.reserve(databases.size());
        for (const auto& e : databases) { result.push_back(&e.db); }
        return result;
    }

private:
    /// Derive material_type and vendor from directory structure.
    /// Expected layout: root / Material / Vendor / file.json
    /// Falls back to "Unknown" when the hierarchy is flat.
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
