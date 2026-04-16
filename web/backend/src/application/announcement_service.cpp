#include "application/announcement_service.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <system_error>

namespace chromaprint3d::backend {
namespace {

using json       = nlohmann::json;
using clock_type = std::chrono::system_clock;

constexpr std::size_t kMaxFileBytes      = 1024 * 1024; // 1 MB
constexpr std::size_t kMaxTextFieldBytes = 2000;
constexpr const char* kAnnouncementsFile = "announcements.json";

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

bool IsAllowedType(const std::string& s) {
    return s == "info" || s == "warning" || s == "maintenance";
}

bool IsAllowedSeverity(const std::string& s) {
    return s == "info" || s == "warning" || s == "error";
}

int SeverityRank(const std::string& s) {
    if (s == "error") return 3;
    if (s == "warning") return 2;
    return 1; // info
}

std::optional<std::string> OptionalString(const json& obj, const std::string& key) {
    if (!obj.contains(key) || obj[key].is_null()) return std::nullopt;
    if (!obj[key].is_string()) return std::nullopt;
    auto value = obj[key].get<std::string>();
    if (value.empty()) return std::nullopt;
    return value;
}

bool StringLengthWithinLimit(const std::optional<std::string>& s, std::size_t limit) {
    if (!s) return true;
    return s->size() <= limit;
}

// ---------------------------------------------------------------------------
// Content hash (FNV-1a 64, printed as 8-hex prefix)
//
// Choice rationale: the hash is a *change detector* used by health() so
// the frontend can refetch. It does not need cryptographic strength. FNV-1a
// is stable, dependency-free and cheap.
// ---------------------------------------------------------------------------

std::string Fnv1a64Hex(const std::string& data) {
    constexpr std::uint64_t kOffset = 1469598103934665603ULL;
    constexpr std::uint64_t kPrime  = 1099511628211ULL;

    std::uint64_t h = kOffset;
    for (char c : data) {
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
        h *= kPrime;
    }

    static constexpr char kHex[] = "0123456789abcdef";
    std::array<char, 8> out{};
    // Take the top 32 bits of the 64-bit hash and render as 8 hex chars.
    const std::uint32_t top = static_cast<std::uint32_t>(h >> 32);
    for (int i = 0; i < 8; ++i) { out[7 - i] = kHex[(top >> (i * 4)) & 0x0F]; }
    return std::string(out.data(), out.size());
}

// ---------------------------------------------------------------------------
// Wire format helpers
// ---------------------------------------------------------------------------

void WriteOptional(json& out, const std::string& key, const std::optional<std::string>& value) {
    if (value)
        out[key] = *value;
    else
        out[key] = nullptr;
}

// Canonicalize ordering for hashing so hash only changes when content changes.
// Sort by id, then emit each record with the same key order.
std::string CanonicalJsonForHash(const std::vector<Announcement>& items) {
    std::vector<Announcement> sorted = items;
    std::sort(sorted.begin(), sorted.end(),
              [](const Announcement& a, const Announcement& b) { return a.id < b.id; });

    json arr = json::array();
    for (const auto& a : sorted) arr.push_back(AnnouncementService::ToWireJson(a));
    // nlohmann's dump keeps insertion order for objects; we pass the same
    // ToWireJson helper used by the wire format to keep both paths in sync.
    return arr.dump();
}

// ---------------------------------------------------------------------------
// Disk I/O helpers
// ---------------------------------------------------------------------------

std::optional<json> ReadJsonFile(const std::filesystem::path& path, std::size_t max_bytes) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) return std::nullopt;
    if (size > max_bytes) {
        spdlog::warn("announcements: {} exceeds size limit ({} > {}); treating as empty",
                     path.string(), size, max_bytes);
        return std::nullopt;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::ostringstream buf;
    buf << in.rdbuf();
    try {
        return json::parse(buf.str());
    } catch (const std::exception& e) {
        spdlog::warn("announcements: parse failed for {}: {}", path.string(), e.what());
        return std::nullopt;
    }
}

bool WriteJsonFileAtomic(const std::filesystem::path& path, const json& value) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        spdlog::error("announcements: cannot create parent dir for {}: {}", path.string(),
                      ec.message());
        return false;
    }

    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            spdlog::error("announcements: failed to open {} for writing", tmp.string());
            return false;
        }
        out << value.dump(2);
        out.flush();
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        spdlog::error("announcements: rename {} → {} failed: {}", tmp.string(), path.string(),
                      ec.message());
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Announcement <-> JSON (on disk mirrors the wire format)
// ---------------------------------------------------------------------------

std::optional<Announcement> AnnouncementFromJson(const json& j, std::string& error_out) {
    if (!j.is_object()) {
        error_out = "entry is not an object";
        return std::nullopt;
    }
    Announcement a;
    try {
        a.id                  = j.value("id", std::string{});
        a.type                = j.value("type", std::string{});
        a.severity            = j.value("severity", std::string{});
        a.title_zh            = OptionalString(j.value("title", json::object()), "zh");
        a.title_en            = OptionalString(j.value("title", json::object()), "en");
        a.body_zh             = OptionalString(j.value("body", json::object()), "zh");
        a.body_en             = OptionalString(j.value("body", json::object()), "en");
        a.starts_at           = OptionalString(j, "starts_at");
        a.ends_at             = j.value("ends_at", std::string{});
        a.scheduled_update_at = OptionalString(j, "scheduled_update_at");
        a.dismissible         = j.value("dismissible", true);
        a.created_at          = j.value("created_at", std::string{});
        a.updated_at          = j.value("updated_at", std::string{});
    } catch (const std::exception& e) {
        error_out = std::string("bad entry: ") + e.what();
        return std::nullopt;
    }
    return a;
}

} // namespace

// ---------------------------------------------------------------------------
// Public statics
// ---------------------------------------------------------------------------

std::string AnnouncementService::FormatIso8601Utc(clock_type::time_point tp) {
    const auto time = clock_type::to_time_t(tp);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    std::array<char, 32> buf{};
    std::strftime(buf.data(), buf.size(), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf.data());
}

bool AnnouncementService::IsValidIso8601Utc(const std::string& s) {
    static const std::regex kPattern(R"(^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$)");
    return std::regex_match(s, kPattern);
}

bool AnnouncementService::IsValidId(const std::string& id) {
    static const std::regex kPattern(R"(^[a-zA-Z0-9_-]{1,64}$)");
    return std::regex_match(id, kPattern);
}

json AnnouncementService::ToWireJson(const Announcement& a) {
    json out;
    out["id"]       = a.id;
    out["type"]     = a.type;
    out["severity"] = a.severity;

    json title = json::object();
    WriteOptional(title, "zh", a.title_zh);
    WriteOptional(title, "en", a.title_en);
    out["title"] = std::move(title);

    json body = json::object();
    WriteOptional(body, "zh", a.body_zh);
    WriteOptional(body, "en", a.body_en);
    out["body"] = std::move(body);

    WriteOptional(out, "starts_at", a.starts_at);
    out["ends_at"] = a.ends_at;
    WriteOptional(out, "scheduled_update_at", a.scheduled_update_at);

    out["dismissible"] = a.dismissible;
    out["created_at"]  = a.created_at;
    out["updated_at"]  = a.updated_at;
    return out;
}

// ---------------------------------------------------------------------------
// Construction / state
// ---------------------------------------------------------------------------

AnnouncementService::AnnouncementService(std::filesystem::path data_dir)
    : data_dir_(std::move(data_dir)), data_file_(data_dir_ / kAnnouncementsFile) {
    std::unique_lock lk(mtx_);
    LoadFromDiskLocked();
    RefreshVersionHashLocked();
    spdlog::info("announcements: loaded {} entries from {}", state_.items.size(),
                 data_file_.string());
}

void AnnouncementService::LoadFromDiskLocked() {
    state_.items.clear();
    if (!std::filesystem::exists(data_file_)) return;

    auto parsed = ReadJsonFile(data_file_, kMaxFileBytes);
    if (!parsed) return;
    if (!parsed->is_array()) {
        spdlog::warn("announcements: {} is not a JSON array, ignoring", data_file_.string());
        return;
    }

    for (const auto& entry : *parsed) {
        std::string error;
        auto parsed_entry = AnnouncementFromJson(entry, error);
        if (!parsed_entry) {
            spdlog::warn("announcements: skipping invalid entry: {}", error);
            continue;
        }
        std::string validate_err;
        if (!IsValidId(parsed_entry->id) || !IsAllowedType(parsed_entry->type) ||
            !IsAllowedSeverity(parsed_entry->severity) ||
            !IsValidIso8601Utc(parsed_entry->ends_at)) {
            spdlog::warn("announcements: skipping entry '{}' failing schema",
                         parsed_entry->id.empty() ? "<?>" : parsed_entry->id);
            continue;
        }
        state_.items.push_back(std::move(*parsed_entry));
    }
}

void AnnouncementService::PersistLocked() const {
    json arr = json::array();
    for (const auto& a : state_.items) arr.push_back(ToWireJson(a));
    if (!WriteJsonFileAtomic(data_file_, arr)) {
        spdlog::error("announcements: failed to persist {}", data_file_.string());
    }
}

void AnnouncementService::RefreshVersionHashLocked() {
    state_.version_hash = Fnv1a64Hex(CanonicalJsonForHash(state_.items));
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

std::vector<Announcement> AnnouncementService::ListActive(clock_type::time_point now_utc) const {
    const auto now = FormatIso8601Utc(now_utc);
    std::shared_lock lk(mtx_);
    std::vector<Announcement> out;
    out.reserve(state_.items.size());
    for (const auto& a : state_.items) {
        if (a.starts_at && *a.starts_at > now) continue;
        if (a.ends_at <= now) continue;
        out.push_back(a);
    }
    std::sort(out.begin(), out.end(), [](const Announcement& a, const Announcement& b) {
        const int ra = SeverityRank(a.severity);
        const int rb = SeverityRank(b.severity);
        if (ra != rb) return ra > rb;
        return a.updated_at < b.updated_at;
    });
    return out;
}

std::vector<Announcement> AnnouncementService::ListActive() const {
    return ListActive(clock_type::now());
}

std::size_t AnnouncementService::ActiveCount(clock_type::time_point now_utc) const {
    return ListActive(now_utc).size();
}

std::size_t AnnouncementService::ActiveCount() const { return ActiveCount(clock_type::now()); }

std::string AnnouncementService::VersionHash() const {
    std::shared_lock lk(mtx_);
    return state_.version_hash;
}

// ---------------------------------------------------------------------------
// Upsert / Remove
// ---------------------------------------------------------------------------

std::optional<Announcement> AnnouncementService::ValidateAndBuild(const json& body,
                                                                  std::string& error_out) const {
    if (!body.is_object()) {
        error_out = "body must be a JSON object";
        return std::nullopt;
    }

    auto id = body.value("id", std::string{});
    if (!IsValidId(id)) {
        error_out = "id must match ^[a-zA-Z0-9_-]{1,64}$";
        return std::nullopt;
    }

    auto type = body.value("type", std::string{});
    if (!IsAllowedType(type)) {
        error_out = "type must be one of: info, warning, maintenance";
        return std::nullopt;
    }

    auto severity = body.value("severity", std::string{});
    if (!IsAllowedSeverity(severity)) {
        error_out = "severity must be one of: info, warning, error";
        return std::nullopt;
    }

    auto title_zh = OptionalString(body.value("title", json::object()), "zh");
    auto title_en = OptionalString(body.value("title", json::object()), "en");
    if (!title_zh && !title_en) {
        error_out = "title must be non-empty in at least one language";
        return std::nullopt;
    }
    if (!StringLengthWithinLimit(title_zh, kMaxTextFieldBytes) ||
        !StringLengthWithinLimit(title_en, kMaxTextFieldBytes)) {
        error_out = "title exceeds 2000 characters";
        return std::nullopt;
    }

    auto body_zh = OptionalString(body.value("body", json::object()), "zh");
    auto body_en = OptionalString(body.value("body", json::object()), "en");
    if (!body_zh && !body_en) {
        error_out = "body must be non-empty in at least one language";
        return std::nullopt;
    }
    if (!StringLengthWithinLimit(body_zh, kMaxTextFieldBytes) ||
        !StringLengthWithinLimit(body_en, kMaxTextFieldBytes)) {
        error_out = "body exceeds 2000 characters";
        return std::nullopt;
    }

    auto starts_at = OptionalString(body, "starts_at");
    if (starts_at && !IsValidIso8601Utc(*starts_at)) {
        error_out = "starts_at must be an ISO8601 UTC string";
        return std::nullopt;
    }

    auto ends_at = body.value("ends_at", std::string{});
    if (ends_at.empty() || !IsValidIso8601Utc(ends_at)) {
        error_out = "ends_at is required and must be an ISO8601 UTC string";
        return std::nullopt;
    }
    const auto now_iso = FormatIso8601Utc(clock_type::now());
    if (ends_at <= now_iso) {
        error_out = "ends_at must be in the future";
        return std::nullopt;
    }
    if (starts_at && *starts_at >= ends_at) {
        error_out = "ends_at must be strictly later than starts_at";
        return std::nullopt;
    }

    auto scheduled_update_at = OptionalString(body, "scheduled_update_at");
    if (scheduled_update_at) {
        if (!IsValidIso8601Utc(*scheduled_update_at)) {
            error_out = "scheduled_update_at must be an ISO8601 UTC string";
            return std::nullopt;
        }
        if (starts_at && *scheduled_update_at < *starts_at) {
            error_out = "scheduled_update_at cannot be earlier than starts_at";
            return std::nullopt;
        }
        if (*scheduled_update_at > ends_at) {
            error_out = "scheduled_update_at cannot be later than ends_at";
            return std::nullopt;
        }
    }

    bool dismissible = true;
    if (body.contains("dismissible")) {
        if (!body["dismissible"].is_boolean()) {
            error_out = "dismissible must be a boolean";
            return std::nullopt;
        }
        dismissible = body["dismissible"].get<bool>();
    }

    Announcement a;
    a.id                  = std::move(id);
    a.type                = std::move(type);
    a.severity            = std::move(severity);
    a.title_zh            = std::move(title_zh);
    a.title_en            = std::move(title_en);
    a.body_zh             = std::move(body_zh);
    a.body_en             = std::move(body_en);
    a.starts_at           = std::move(starts_at);
    a.ends_at             = std::move(ends_at);
    a.scheduled_update_at = std::move(scheduled_update_at);
    a.dismissible         = dismissible;
    return a;
}

Announcement AnnouncementService::MergeForUpsert(const std::optional<Announcement>& existing,
                                                 Announcement incoming,
                                                 const std::string& now_iso) {
    incoming.updated_at = now_iso;
    if (existing && !existing->created_at.empty()) {
        incoming.created_at = existing->created_at;
    } else {
        incoming.created_at = now_iso;
    }
    return incoming;
}

ServiceResult AnnouncementService::Upsert(const json& body) {
    std::string error;
    auto built = ValidateAndBuild(body, error);
    if (!built) { return ServiceResult::Error(400, "invalid_announcement", error); }

    const std::string target_id = built->id;
    const auto now_iso          = FormatIso8601Utc(clock_type::now());

    std::unique_lock lk(mtx_);
    std::optional<Announcement> existing;
    auto it = std::find_if(state_.items.begin(), state_.items.end(),
                           [&](const Announcement& a) { return a.id == target_id; });
    if (it != state_.items.end()) existing = *it;

    auto merged = MergeForUpsert(existing, std::move(*built), now_iso);
    if (it != state_.items.end()) {
        *it = std::move(merged);
    } else {
        state_.items.push_back(std::move(merged));
    }

    RefreshVersionHashLocked();
    PersistLocked();

    const auto stored_it = std::find_if(state_.items.begin(), state_.items.end(),
                                        [&](const Announcement& a) { return a.id == target_id; });
    return ServiceResult::Success(200, {{"announcement", ToWireJson(*stored_it)},
                                        {"announcements_version", state_.version_hash}});
}

ServiceResult AnnouncementService::Remove(const std::string& id) {
    if (!IsValidId(id)) {
        return ServiceResult::Error(400, "invalid_id", "id must match ^[a-zA-Z0-9_-]{1,64}$");
    }

    std::unique_lock lk(mtx_);
    auto it = std::find_if(state_.items.begin(), state_.items.end(),
                           [&](const Announcement& a) { return a.id == id; });
    if (it == state_.items.end()) {
        return ServiceResult::Error(404, "not_found", "announcement not found");
    }
    state_.items.erase(it);
    RefreshVersionHashLocked();
    PersistLocked();
    return ServiceResult::Success(200, {{"announcements_version", state_.version_hash}});
}

} // namespace chromaprint3d::backend
