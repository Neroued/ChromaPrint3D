#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace chromaprint3d::backend {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port         = 8080;

    std::string data_dir;
    std::string web_dir;
    std::string model_pack_path;

    std::string log_level = "info";
    std::string cors_origin;

    std::int64_t max_upload_mb       = 50;
    std::int64_t http_threads        = 4;
    std::int64_t max_tasks           = 4;
    std::int64_t task_ttl_seconds    = 3600;
    std::int64_t session_ttl_seconds = 3600;
    std::int64_t max_task_queue      = 256;

    std::int64_t max_tasks_per_owner      = 32;
    std::int64_t max_task_result_mb       = 512;
    std::int64_t max_pixels_per_image     = 4096LL * 4096LL;
    std::int64_t max_session_colordbs     = 10;
    std::int64_t board_cache_ttl          = 600;
    std::int64_t board_geometry_cache_ttl = 600;
    std::int64_t board_geometry_cache_max = 16;
    bool require_cors_origin              = false;
    bool allow_open_cors                  = true;

    // Announcement system: a shared secret that guards POST/DELETE on
    // /api/v1/announcements. When empty, write routes return 404 (not 401) so
    // that servers without the feature enabled do not leak route existence.
    // Precedence: CLI --announcement-token > CHROMAPRINT3D_ANNOUNCEMENT_TOKEN env.
    std::string announcement_token;
    // When false (default) the auth advice rejects non-HTTPS announcement
    // writes. Set via --announcement-allow-http for local debugging only.
    bool announcement_allow_http = false;
    // Directory that stores `announcements.json`. When empty, falls back to
    // `data_dir`. Kept separate so deployments can bind-mount a writable
    // volume at this path while the rest of `data_dir` (and the container
    // root filesystem) stays read-only.
    std::string announcements_dir;
};

struct ConfigParseResult {
    bool ok = false;
    std::optional<ServerConfig> config;
    std::string error_message;
    bool show_help = false;
};

std::string BuildUsage(const char* exe_name);
ConfigParseResult ParseConfig(int argc, char** argv);

} // namespace chromaprint3d::backend
