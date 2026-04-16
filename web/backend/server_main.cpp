#include "application/server_facade.h"
#include "config/server_config.h"
#include "presentation/announcement_auth_advice.h"
#include "presentation/backend_runtime.h"
#include "presentation/cors_advice.h"

#include "chromaprint3d/logging.h"
#include "chromaprint3d/version.h"

#include <drogon/drogon.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    using chromaprint3d::backend::BuildUsage;
    using chromaprint3d::backend::InitBackendRuntime;
    using chromaprint3d::backend::ParseConfig;
    using chromaprint3d::backend::RegisterAnnouncementAuthAdvice;
    using chromaprint3d::backend::RegisterCorsAdvice;
    using chromaprint3d::backend::ServerFacade;

    auto parsed = ParseConfig(argc, argv);
    if (!parsed.ok || !parsed.config.has_value()) {
        std::cerr << "Error: " << parsed.error_message << "\n\n";
        std::cerr << BuildUsage(argv[0]);
        return parsed.show_help ? 0 : 1;
    }
    if (parsed.show_help) {
        std::cout << BuildUsage(argv[0]);
        return 0;
    }

    auto cfg = *parsed.config;
    ChromaPrint3D::InitLogging(ChromaPrint3D::ParseLogLevel(cfg.log_level));
    spdlog::info("ChromaPrint3D Server v{}", CHROMAPRINT3D_VERSION_STRING);
    spdlog::info("Configuration: host={}, port={}, data={}, max_upload={}MB, http_threads={}, "
                 "task_workers={}, max_queue={}, task_ttl={}s, session_ttl={}s, cors_origin={}",
                 cfg.host, cfg.port, cfg.data_dir, cfg.max_upload_mb, cfg.http_threads,
                 cfg.max_tasks, cfg.max_task_queue, cfg.task_ttl_seconds, cfg.session_ttl_seconds,
                 cfg.cors_origin.empty() ? "(allow all)" : cfg.cors_origin);

    try {
        auto facade = std::make_shared<ServerFacade>(cfg);
        InitBackendRuntime(facade);
        RegisterCorsAdvice(cfg);
        RegisterAnnouncementAuthAdvice(cfg);
        if (cfg.announcement_token.empty()) {
            spdlog::info("Announcement write routes disabled (no --announcement-token configured)");
        } else {
            spdlog::info("Announcement write routes enabled; allow_http={}",
                         cfg.announcement_allow_http ? "true" : "false");
        }

        auto upload_dir = std::filesystem::temp_directory_path() / "chromaprint3d_uploads";
        std::filesystem::create_directories(upload_dir);

        auto& app = drogon::app();
        app.setClientMaxBodySize(static_cast<size_t>(cfg.max_upload_mb) * 1024U * 1024U)
            .setClientMaxMemoryBodySize(1024 * 1024)
            .setUploadPath(upload_dir.string())
            .setThreadNum(static_cast<size_t>(cfg.http_threads))
            .addListener(cfg.host, static_cast<uint16_t>(cfg.port));

        if (!cfg.web_dir.empty()) {
            if (std::filesystem::is_directory(cfg.web_dir)) {
                app.setDocumentRoot(cfg.web_dir);
                spdlog::info("Serving static files from: {}", cfg.web_dir);
            } else {
                spdlog::warn("--web directory does not exist: {}", cfg.web_dir);
            }
        }

        app.run();
    } catch (const std::exception& e) {
        spdlog::error("Fatal server error: {}", e.what());
        return 1;
    }

    return 0;
}
