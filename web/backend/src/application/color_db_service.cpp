#include "application/color_db_service.h"

#include "application/json_serialization.h"

#include "chromaprint3d/color_db.h"

#include <nlohmann/json.hpp>

namespace chromaprint3d::backend {

using json = nlohmann::json;

ColorDbService::ColorDbService(DataRepository& data, SessionStore& sessions)
    : data_(data), sessions_(sessions) {}

ServiceResult ColorDbService::ListDatabases(const std::optional<std::string>& session_token) {
    json arr = json::array();
    for (const auto& entry : data_.ColorDbCache().databases) {
        auto j      = ColorDbInfoToJson(entry.db, entry.material_type, entry.vendor);
        j["source"] = "global";
        arr.push_back(std::move(j));
    }
    if (session_token && !session_token->empty()) {
        auto dbs = sessions_.ListSessionDbs(*session_token);
        for (const auto& db : dbs) {
            auto j      = ColorDbInfoToJson(db);
            j["source"] = "session";
            arr.push_back(std::move(j));
        }
    }
    return ServiceResult::Success(200, {{"databases", std::move(arr)}});
}

ServiceResult ColorDbService::ListSessionDatabases(const std::string& token) {
    if (token.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    json arr = json::array();
    for (const auto& db : sessions_.ListSessionDbs(token)) {
        auto j      = ColorDbInfoToJson(db);
        j["source"] = "session";
        arr.push_back(std::move(j));
    }
    return ServiceResult::Success(200, {{"databases", std::move(arr)}});
}

ServiceResult
ColorDbService::UploadSessionDatabase(const std::string& token, const std::string& json_content,
                                      const std::optional<std::string>& override_name) {
    if (token.empty()) return ServiceResult::Error(401, "unauthorized", "No session");

    ChromaPrint3D::ColorDB db;
    try {
        db = ChromaPrint3D::ColorDB::FromJsonString(json_content);
    } catch (const std::exception& e) {
        return ServiceResult::Error(400, "invalid_colordb",
                                    "Invalid ColorDB JSON: " + std::string(e.what()));
    }
    if (override_name && !override_name->empty()) db.name = *override_name;
    if (!SessionStore::IsValidDbName(db.name)) {
        return ServiceResult::Error(400, "invalid_name",
                                    "ColorDB name must be [a-zA-Z0-9_], length 1-64");
    }
    if (data_.ColorDbCache().FindByName(db.name)) {
        return ServiceResult::Error(409, "name_conflict", "Name conflicts with global database");
    }
    std::string err;
    if (!sessions_.PutSessionDb(token, db, &err)) {
        return ServiceResult::Error(429, "session_limit", err);
    }
    auto out      = ColorDbInfoToJson(db);
    out["source"] = "session";
    return ServiceResult::Success(200, std::move(out));
}

ServiceResult ColorDbService::DeleteSessionDatabase(const std::string& token,
                                                    const std::string& name) {
    if (token.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto existing = sessions_.GetSessionDb(token, name);
    if (!existing) return ServiceResult::Error(404, "not_found", "ColorDB not found in session");
    sessions_.RemoveSessionDb(token, name);
    return ServiceResult::Success(200, {{"deleted", true}});
}

ServiceResult ColorDbService::DownloadSessionDatabase(const std::string& token,
                                                      const std::string& name,
                                                      std::string& json_out,
                                                      std::string& filename_out) {
    if (token.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto existing = sessions_.GetSessionDb(token, name);
    if (!existing) return ServiceResult::Error(404, "not_found", "ColorDB not found in session");
    json_out     = existing->ToJsonString();
    filename_out = name + ".json";
    return ServiceResult::Success(200, json::object());
}

} // namespace chromaprint3d::backend
