#include "config/server_config.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_map>

namespace chromaprint3d::backend {
namespace {

bool ParseI64(const std::string& s, std::int64_t& out) {
    try {
        size_t idx = 0;
        auto v     = std::stoll(s, &idx, 10);
        if (idx != s.size()) return false;
        out = v;
        return true;
    } catch (...) { return false; }
}

bool ParseBool(const std::string& s, bool& out) {
    std::string lowered = s;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
        out = true;
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
        out = false;
        return true;
    }
    return false;
}

bool ValidateI64Range(const char* flag, std::int64_t value, std::int64_t min_value,
                      std::int64_t max_value, ConfigParseResult& result) {
    if (value >= min_value && value <= max_value) return true;
    result.error_message = std::string("Invalid value for ") + flag + ": " + std::to_string(value) +
                           " (expected " + std::to_string(min_value) + ".." +
                           std::to_string(max_value) + ")";
    return false;
}

} // namespace

std::string BuildUsage(const char* exe_name) {
    std::ostringstream oss;
    oss << "Usage: " << exe_name << " --data <dir> [options]\n"
        << "Options:\n"
        << "  --host HOST                 Bind address (default: 0.0.0.0)\n"
        << "  --port PORT                 HTTP port (default: 8080)\n"
        << "  --data DIR                  Data root directory (required)\n"
        << "  --web DIR                   Static web root directory\n"
        << "  --model-pack PATH           Model package json path\n"
        << "  --log-level LEVEL           trace/debug/info/warn/error/off\n"
        << "  --cors-origin URL           Allowed CORS origin for cross-origin mode\n"
        << "  --require-cors-origin BOOL  Fail startup when --cors-origin is empty (0/1)\n"
        << "  --max-upload-mb N           Max request body MB (default: 50, range: 1-512)\n"
        << "  --http-threads N            Drogon HTTP worker threads (default: 4)\n"
        << "  --max-tasks N               Async task worker threads (default: 4)\n"
        << "  --max-queue N               Max queued async tasks (default: 256, range: 1-10000)\n"
        << "  --task-ttl N                Completed task TTL seconds (default: 3600, range: "
           "60-604800)\n"
        << "  --session-ttl N             Session TTL seconds (default: 3600, range: 60-604800)\n"
        << "  --max-owner-tasks N         Max pending/running tasks per owner (default: 32, range: "
           "1-1024)\n"
        << "  --max-result-mb N           Max total in-memory result payload MB (default: 512, "
           "range: 16-8192)\n"
        << "  --max-pixels N              Max decoded pixels per image (default: 16777216, range: "
           "1048576-268435456)\n"
        << "  --max-session-colordbs N    Max uploaded ColorDB count per session (default: 10)\n"
        << "  --board-cache-ttl N         Calibration board cache TTL seconds (default: 600)\n"
        << "  --board-geometry-cache-ttl N Board geometry cache TTL seconds (default: 600)\n"
        << "  --board-geometry-cache-max N Max cached board geometries (default: 16)\n"
        << "  --announcement-token TOKEN  Secret for writing announcements (overrides env\n"
        << "                              CHROMAPRINT3D_ANNOUNCEMENT_TOKEN); when empty, the\n"
        << "                              write routes return 404 and the feature is disabled\n"
        << "  --announcement-allow-http BOOL  Allow non-HTTPS announcement writes (default: 0,\n"
        << "                                  local debugging only)\n"
        << "  -h, --help                  Show this help\n";
    return oss.str();
}

ConfigParseResult ParseConfig(int argc, char** argv) {
    ConfigParseResult result;
    ServerConfig cfg;

    // Seed announcement token from environment first; CLI overrides it below.
    if (const char* env_token = std::getenv("CHROMAPRINT3D_ANNOUNCEMENT_TOKEN");
        env_token != nullptr) {
        cfg.announcement_token = env_token;
    }

    std::unordered_map<std::string, std::int64_t*> i64_flags = {
        {"--max-upload-mb", &cfg.max_upload_mb},
        {"--http-threads", &cfg.http_threads},
        {"--max-tasks", &cfg.max_tasks},
        {"--max-queue", &cfg.max_task_queue},
        {"--task-ttl", &cfg.task_ttl_seconds},
        {"--session-ttl", &cfg.session_ttl_seconds},
        {"--max-owner-tasks", &cfg.max_tasks_per_owner},
        {"--max-result-mb", &cfg.max_task_result_mb},
        {"--max-pixels", &cfg.max_pixels_per_image},
        {"--max-session-colordbs", &cfg.max_session_colordbs},
        {"--board-cache-ttl", &cfg.board_cache_ttl},
        {"--board-geometry-cache-ttl", &cfg.board_geometry_cache_ttl},
        {"--board-geometry-cache-max", &cfg.board_geometry_cache_max},
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            result.show_help = true;
            result.ok        = true;
            result.config    = cfg;
            return result;
        }
        if (i + 1 >= argc) {
            result.error_message = "Missing value for argument: " + arg;
            return result;
        }
        std::string value = argv[++i];

        if (arg == "--host") {
            cfg.host = value;
            continue;
        }
        if (arg == "--port") {
            std::int64_t parsed = 0;
            if (!ParseI64(value, parsed) || parsed <= 0 || parsed > 65535) {
                result.error_message = "Invalid port: " + value;
                return result;
            }
            cfg.port = static_cast<int>(parsed);
            continue;
        }
        if (arg == "--data") {
            cfg.data_dir = value;
            continue;
        }
        if (arg == "--web") {
            cfg.web_dir = value;
            continue;
        }
        if (arg == "--model-pack") {
            cfg.model_pack_path = value;
            continue;
        }
        if (arg == "--log-level") {
            cfg.log_level = value;
            continue;
        }
        if (arg == "--cors-origin") {
            cfg.cors_origin = value;
            continue;
        }
        if (arg == "--require-cors-origin") {
            bool parsed = false;
            if (!ParseBool(value, parsed)) {
                result.error_message = "Invalid boolean value for --require-cors-origin: " + value;
                return result;
            }
            cfg.require_cors_origin = parsed;
            continue;
        }
        if (arg == "--announcement-token") {
            cfg.announcement_token = value;
            continue;
        }
        if (arg == "--announcement-allow-http") {
            bool parsed = false;
            if (!ParseBool(value, parsed)) {
                result.error_message =
                    "Invalid boolean value for --announcement-allow-http: " + value;
                return result;
            }
            cfg.announcement_allow_http = parsed;
            continue;
        }

        auto it = i64_flags.find(arg);
        if (it == i64_flags.end()) {
            result.error_message = "Unknown argument: " + arg;
            return result;
        }
        std::int64_t parsed = 0;
        if (!ParseI64(value, parsed)) {
            result.error_message = "Invalid integer value for " + arg + ": " + value;
            return result;
        }
        *it->second = parsed;
    }

    if (cfg.data_dir.empty()) {
        result.error_message = "Missing required argument: --data";
        return result;
    }

    if (!ValidateI64Range("--max-upload-mb", cfg.max_upload_mb, 1, 512, result)) return result;
    if (!ValidateI64Range("--http-threads", cfg.http_threads, 1, 256, result)) return result;
    if (!ValidateI64Range("--max-tasks", cfg.max_tasks, 1, 256, result)) return result;
    if (!ValidateI64Range("--max-queue", cfg.max_task_queue, 1, 10000, result)) return result;
    if (!ValidateI64Range("--task-ttl", cfg.task_ttl_seconds, 60, 604800, result)) return result;
    if (!ValidateI64Range("--session-ttl", cfg.session_ttl_seconds, 60, 604800, result))
        return result;
    if (!ValidateI64Range("--max-owner-tasks", cfg.max_tasks_per_owner, 1, 1024, result))
        return result;
    if (!ValidateI64Range("--max-result-mb", cfg.max_task_result_mb, 16, 8192, result))
        return result;
    if (!ValidateI64Range("--max-pixels", cfg.max_pixels_per_image, 1024 * 1024, 268435456, result))
        return result;
    if (!ValidateI64Range("--max-session-colordbs", cfg.max_session_colordbs, 1, 256, result))
        return result;
    if (!ValidateI64Range("--board-cache-ttl", cfg.board_cache_ttl, 60, 86400, result))
        return result;
    if (!ValidateI64Range("--board-geometry-cache-ttl", cfg.board_geometry_cache_ttl, 60, 86400,
                          result))
        return result;
    if (!ValidateI64Range("--board-geometry-cache-max", cfg.board_geometry_cache_max, 1, 256,
                          result))
        return result;

    if (cfg.require_cors_origin && cfg.cors_origin.empty()) {
        result.error_message =
            "--require-cors-origin is enabled but --cors-origin is not configured";
        return result;
    }

    cfg.allow_open_cors = cfg.cors_origin.empty() && !cfg.require_cors_origin;
    result.ok           = true;
    result.config       = cfg;
    return result;
}

} // namespace chromaprint3d::backend
