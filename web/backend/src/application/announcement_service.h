#pragma once

#include "application/service_result.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

namespace chromaprint3d::backend {

// A user-facing announcement rendered by the frontend as a banner.
//
// Intentionally plain data + strings: every time field is the raw
// ISO8601-UTC string supplied by the operator. All comparisons are done
// as strings against a current-time string formatted by the service,
// which gives total ordering without needing a UTC parser.
struct Announcement {
    std::string id;
    std::string type;     // "info" | "warning" | "maintenance"
    std::string severity; // "info" | "warning" | "error"

    // Titles / bodies are bilingual. Both languages are optional; at
    // least one of the two languages must be non-empty for title,
    // and at least one for body.
    std::optional<std::string> title_zh;
    std::optional<std::string> title_en;
    std::optional<std::string> body_zh;
    std::optional<std::string> body_en;

    std::optional<std::string> starts_at;
    std::string ends_at;
    std::optional<std::string> scheduled_update_at;
    bool dismissible = true;

    std::string created_at;
    std::string updated_at;
};

// Thread-safe in-memory store backed by `${data_dir}/announcements.json`.
// - Reads are lock-free-ish (shared mutex) and very fast.
// - Writes are atomic: serialize → temp file → rename.
// - Version hash is a stable 64-bit FNV-1a over the serialized JSON,
//   surfaced as the first 8 hex chars; used by health to let the
//   frontend detect changes.
class AnnouncementService {
public:
    explicit AnnouncementService(std::filesystem::path data_dir);

    AnnouncementService(const AnnouncementService&)            = delete;
    AnnouncementService& operator=(const AnnouncementService&) = delete;

    // Returns all announcements whose [starts_at, ends_at] window covers
    // `now_utc`, sorted by severity (error > warning > info) then by
    // updated_at ascending.
    std::vector<Announcement> ListActive(std::chrono::system_clock::time_point now_utc) const;

    // Convenience overload using the system clock.
    std::vector<Announcement> ListActive() const;

    // Insert (by id) or update (when an entry with the same id already
    // exists). On update, created_at is preserved and updated_at is
    // refreshed to "now". Returns 400 on schema validation errors.
    ServiceResult Upsert(const nlohmann::json& body);

    // Removes by id. Returns 404 when the id is not present.
    ServiceResult Remove(const std::string& id);

    // Stable 8-char hex content hash (FNV-1a over canonicalized JSON).
    // Non-empty even when there are no announcements.
    std::string VersionHash() const;

    // Count of announcements active at the given time (used by Health()).
    std::size_t ActiveCount(std::chrono::system_clock::time_point now_utc) const;
    std::size_t ActiveCount() const;

    // Returns the ISO8601-UTC string representation of `tp`.
    static std::string FormatIso8601Utc(std::chrono::system_clock::time_point tp);

    // Strict ISO8601-UTC validator: `YYYY-MM-DDTHH:MM:SS[.fraction]Z`.
    static bool IsValidIso8601Utc(const std::string& s);

    // `^[a-zA-Z0-9_-]{1,64}$`
    static bool IsValidId(const std::string& id);

    // Converts one announcement to the public JSON wire format.
    static nlohmann::json ToWireJson(const Announcement& a);

private:
    struct State {
        std::vector<Announcement> items;
        std::string version_hash;
    };

    std::filesystem::path data_dir_;
    std::filesystem::path data_file_;
    mutable std::shared_mutex mtx_;
    State state_;

    // Re-serializes current items in canonical order and recomputes the
    // FNV-1a hash. Caller must hold unique_lock.
    void RefreshVersionHashLocked();

    // Atomic disk persistence. Must be called under unique_lock.
    void PersistLocked() const;

    // Loads announcements from disk at construction time. Failures
    // (missing file, parse errors, or per-entry validation errors) are
    // logged and result in an empty list — the service must never
    // prevent server startup.
    void LoadFromDiskLocked();

    // Validates `body` against the full schema. On success, returns a
    // constructed Announcement with created_at/updated_at filled in. On
    // failure, returns the error message in `error_out`.
    std::optional<Announcement> ValidateAndBuild(const nlohmann::json& body,
                                                 std::string& error_out) const;

    // Merges created_at from an existing entry on upsert, so modifying
    // the content preserves the original creation timestamp.
    static Announcement MergeForUpsert(const std::optional<Announcement>& existing,
                                       Announcement incoming, const std::string& now_iso);
};

} // namespace chromaprint3d::backend
