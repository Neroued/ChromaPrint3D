#pragma once

#include "chromaprint3d/color_db.h"

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace chromaprint3d::backend {

struct SessionSnapshot {
    std::string token;
    std::unordered_map<std::string, std::shared_ptr<const ChromaPrint3D::ColorDB>> color_dbs;
    std::chrono::steady_clock::time_point last_access;
    std::size_t estimated_db_bytes = 0;
};

class SessionStore {
public:
    SessionStore(std::int64_t ttl_seconds, std::int64_t max_colordbs);

    std::string EnsureSession(const std::optional<std::string>& existing_token, bool* created);
    std::optional<SessionSnapshot> Snapshot(const std::string& token);
    bool WithSession(const std::string& token,
                     const std::function<void(SessionSnapshot&)>& mutator);
    bool RemoveSessionDb(const std::string& token, const std::string& db_name);
    std::shared_ptr<const ChromaPrint3D::ColorDB> GetSessionDb(const std::string& token,
                                                               const std::string& db_name);
    std::vector<std::shared_ptr<const ChromaPrint3D::ColorDB>>
    ListSessionDbs(const std::string& token);

    bool PutSessionDb(const std::string& token, ChromaPrint3D::ColorDB db, std::string* error);
    void CleanupExpired();

    std::size_t EstimateSessionDbBytes(const std::string& token) const;
    std::size_t EstimateAllSessionDbBytes() const;

    static bool IsValidDbName(const std::string& name);
    static std::string NewSecureToken();

private:
    std::int64_t ttl_seconds_;
    std::int64_t max_colordbs_;

    mutable std::mutex mtx_;
    std::unordered_map<std::string, SessionSnapshot> sessions_;
    std::size_t total_session_db_bytes_ = 0;

    SessionSnapshot& GetOrCreateLocked(const std::string& token);
};

} // namespace chromaprint3d::backend
