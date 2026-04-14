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
