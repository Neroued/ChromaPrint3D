#include "infrastructure/session_store.h"
#include "infrastructure/color_db_bytes.h"
#include "infrastructure/random_utils.h"

#include <regex>

namespace chromaprint3d::backend {

SessionStore::SessionStore(std::int64_t ttl_seconds, std::int64_t max_colordbs)
    : ttl_seconds_(ttl_seconds), max_colordbs_(max_colordbs) {}

std::string SessionStore::EnsureSession(const std::optional<std::string>& existing_token,
                                        bool* created) {
    CleanupExpired();

    std::string token = existing_token.value_or("");
    if (token.empty()) token = NewSecureToken();

    std::lock_guard<std::mutex> lock(mtx_);
    auto& s       = GetOrCreateLocked(token);
    s.last_access = std::chrono::steady_clock::now();
    if (created) *created = !existing_token.has_value() || existing_token->empty();
    return token;
}

std::optional<SessionSnapshot> SessionStore::Snapshot(const std::string& token) {
    if (token.empty()) return std::nullopt;
    CleanupExpired();

    std::lock_guard<std::mutex> lock(mtx_);
    auto it = sessions_.find(token);
    if (it == sessions_.end()) return std::nullopt;
    it->second.last_access = std::chrono::steady_clock::now();
    return it->second;
}

bool SessionStore::WithSession(const std::string& token,
                               const std::function<void(SessionSnapshot&)>& mutator) {
    if (token.empty()) return false;
    CleanupExpired();
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = sessions_.find(token);
    if (it == sessions_.end()) return false;
    const auto before_bytes = it->second.estimated_db_bytes;
    it->second.last_access  = std::chrono::steady_clock::now();
    mutator(it->second);
    it->second.estimated_db_bytes = detail::EstimateColorDbMapBytes(it->second.color_dbs);
    total_session_db_bytes_       = total_session_db_bytes_ -
                              std::min(total_session_db_bytes_, before_bytes) +
                              it->second.estimated_db_bytes;
    return true;
}

bool SessionStore::RemoveSessionDb(const std::string& token, const std::string& db_name) {
    return WithSession(token, [&](SessionSnapshot& s) { s.color_dbs.erase(db_name); });
}

std::shared_ptr<const ChromaPrint3D::ColorDB>
SessionStore::GetSessionDb(const std::string& token, const std::string& db_name) {
    if (token.empty()) return nullptr;
    CleanupExpired();
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = sessions_.find(token);
    if (it == sessions_.end()) return nullptr;
    it->second.last_access = std::chrono::steady_clock::now();
    auto db_it             = it->second.color_dbs.find(db_name);
    if (db_it == it->second.color_dbs.end()) return nullptr;
    return db_it->second;
}

std::vector<std::shared_ptr<const ChromaPrint3D::ColorDB>>
SessionStore::ListSessionDbs(const std::string& token) {
    std::vector<std::shared_ptr<const ChromaPrint3D::ColorDB>> out;
    if (token.empty()) return out;
    CleanupExpired();
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = sessions_.find(token);
    if (it == sessions_.end()) return out;
    it->second.last_access = std::chrono::steady_clock::now();
    out.reserve(it->second.color_dbs.size());
    for (const auto& [_, db] : it->second.color_dbs) out.push_back(db);
    return out;
}

bool SessionStore::PutSessionDb(const std::string& token, ChromaPrint3D::ColorDB db,
                                std::string* error) {
    if (!IsValidDbName(db.name)) {
        if (error) *error = "Invalid ColorDB name";
        return false;
    }
    const std::string db_name = db.name;
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = sessions_.find(token);
    if (it == sessions_.end()) {
        if (error) *error = "Session not found";
        return false;
    }
    auto& s       = it->second;
    s.last_access = std::chrono::steady_clock::now();
    auto existing = s.color_dbs.find(db_name);
    if (existing == s.color_dbs.end() &&
        static_cast<std::int64_t>(s.color_dbs.size()) >= max_colordbs_) {
        if (error) *error = "Session ColorDB limit reached";
        return false;
    }
    const std::size_t before_bytes = existing != s.color_dbs.end() && existing->second
                                         ? detail::EstimateColorDbBytes(*existing->second)
                                         : 0;
    auto db_ptr                    = std::make_shared<const ChromaPrint3D::ColorDB>(std::move(db));
    const std::size_t after_bytes  = detail::EstimateColorDbBytes(*db_ptr);
    s.color_dbs[db_name]           = std::move(db_ptr);
    s.estimated_db_bytes =
        s.estimated_db_bytes - std::min(s.estimated_db_bytes, before_bytes) + after_bytes;
    total_session_db_bytes_ =
        total_session_db_bytes_ - std::min(total_session_db_bytes_, before_bytes) + after_bytes;
    return true;
}

void SessionStore::CleanupExpired() {
    std::lock_guard<std::mutex> lock(mtx_);
    auto now = std::chrono::steady_clock::now();
    for (auto it = sessions_.begin(); it != sessions_.end();) {
        auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(now - it->second.last_access).count();
        if (elapsed > ttl_seconds_) {
            total_session_db_bytes_ -=
                std::min(total_session_db_bytes_, it->second.estimated_db_bytes);
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
}

std::size_t SessionStore::EstimateSessionDbBytes(const std::string& token) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = sessions_.find(token);
    if (it == sessions_.end()) return 0;
    return it->second.estimated_db_bytes;
}

std::size_t SessionStore::EstimateAllSessionDbBytes() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return total_session_db_bytes_;
}

bool SessionStore::IsValidDbName(const std::string& name) {
    if (name.empty() || name.size() > 64) return false;
    static const std::regex kPattern("^[a-zA-Z0-9_]+$");
    return std::regex_match(name, kPattern);
}

std::string SessionStore::NewSecureToken() { return detail::RandomHex(32); }

SessionSnapshot& SessionStore::GetOrCreateLocked(const std::string& token) {
    auto& s       = sessions_[token];
    s.token       = token;
    s.last_access = std::chrono::steady_clock::now();
    return s;
}

} // namespace chromaprint3d::backend
