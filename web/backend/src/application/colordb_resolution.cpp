#include "application/colordb_resolution.h"

#include <cctype>
#include <unordered_set>

namespace chromaprint3d::backend {

using json = nlohmann::json;
using namespace ChromaPrint3D;

namespace {

std::string NormalizeLabel(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isalnum(c)) { out.push_back(static_cast<char>(std::tolower(c))); }
    }
    return out;
}

} // namespace

std::string NormalizeChannelKey(const Channel& ch) {
    return NormalizeLabel(ch.color) + "|" + NormalizeLabel(ch.material);
}

std::vector<std::string> BuildProfileChannelKeys(const std::vector<const ColorDB*>& dbs) {
    std::unordered_set<std::string> keys;
    for (const auto* db : dbs) {
        for (const auto& ch : db->palette) { keys.insert(NormalizeChannelKey(ch)); }
    }
    return {keys.begin(), keys.end()};
}

std::vector<std::string>
BuildProfileChannelKeys(const std::vector<std::shared_ptr<const ColorDB>>& dbs) {
    std::unordered_set<std::string> keys;
    for (const auto& db : dbs) {
        for (const auto& ch : db->palette) { keys.insert(NormalizeChannelKey(ch)); }
    }
    return {keys.begin(), keys.end()};
}

ServiceResult ResolveSelectedColorDbs(const json& params,
                                      const std::optional<SessionSnapshot>& session,
                                      const DataRepository& data,
                                      std::vector<std::shared_ptr<const ColorDB>>& out_dbs,
                                      std::string& common_vendor, std::string& common_material) {
    out_dbs.clear();
    common_vendor.clear();
    common_material.clear();

    struct DbWithMeta {
        std::shared_ptr<const ColorDB> db;
        std::string vendor;
        std::string material_type;
    };

    std::vector<DbWithMeta> selected;

    if (params.contains("db_names") && params["db_names"].is_array()) {
        selected.reserve(params["db_names"].size());
        for (const auto& name_val : params["db_names"]) {
            if (!name_val.is_string()) {
                return ServiceResult::Error(400, "invalid_params",
                                            "db_names must be an array of strings");
            }
            const std::string name = name_val.get<std::string>();
            std::shared_ptr<const ColorDB> db;
            std::string vendor, material;
            if (const auto* entry = data.ColorDbCache().FindEntryByName(name)) {
                db       = entry->db;
                vendor   = entry->vendor;
                material = entry->material_type;
            }
            if (!db && session) {
                auto it = session->color_dbs.find(name);
                if (it != session->color_dbs.end()) { db = it->second; }
            }
            if (!db) {
                return ServiceResult::Error(400, "invalid_params", "ColorDB not found: " + name);
            }
            selected.push_back({std::move(db), vendor, material});
        }
        if (selected.empty()) {
            return ServiceResult::Error(400, "invalid_params", "No valid ColorDB names provided");
        }
    } else {
        selected.reserve(data.ColorDbCache().databases.size());
        for (const auto& entry : data.ColorDbCache().databases) {
            selected.push_back({entry.db, entry.vendor, entry.material_type});
        }
    }

    out_dbs.reserve(selected.size());
    for (auto& s : selected) { out_dbs.push_back(std::move(s.db)); }

    if (!selected.empty()) {
        const auto& first_v = selected.front().vendor;
        const auto& first_m = selected.front().material_type;
        bool all_same       = !first_v.empty() && !first_m.empty();
        for (size_t i = 1; i < selected.size() && all_same; ++i) {
            if (selected[i].vendor != first_v || selected[i].material_type != first_m) {
                all_same = false;
            }
        }
        if (all_same) {
            common_vendor   = first_v;
            common_material = first_m;
        }
    }

    return ServiceResult::Success(200, json::object());
}

void ResolveTaskVendorMaterial(const std::vector<std::string>& db_names, const DataRepository& data,
                               std::string& common_vendor, std::string& common_material) {
    common_vendor.clear();
    common_material.clear();

    bool all_same = true;
    for (const auto& name : db_names) {
        const auto* entry = data.ColorDbCache().FindEntryByName(name);
        if (!entry) continue;
        if (common_vendor.empty()) {
            common_vendor   = entry->vendor;
            common_material = entry->material_type;
        } else if (entry->vendor != common_vendor || entry->material_type != common_material) {
            all_same = false;
            break;
        }
    }
    if (!all_same) {
        common_vendor.clear();
        common_material.clear();
    }
}

} // namespace chromaprint3d::backend
