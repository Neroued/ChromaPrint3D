#include "server/server_options.h"
#include "server/server_context.h"
#include "server/http_utils.h"
#include "server/routes_health.h"
#include "server/routes_convert.h"
#include "server/routes_colordb.h"
#include "server/routes_calibration.h"
#include "server/routes_session.h"

#include "chromaprint3d/logging.h"
#include "chromaprint3d/model_package.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <filesystem>
#include <optional>

using namespace ChromaPrint3D;

int main(int argc, char** argv) {
    ServerOptions opts;
    if (!ParseArgs(argc, argv, opts)) { return 1; }

    InitLogging(ParseLogLevel(opts.log_level));

    spdlog::info("ChromaPrint3D Server v1.0.0");
    spdlog::info("Configuration: port={}, host={}, data={}, max_upload={}MB, max_tasks={}, "
                 "task_ttl={}s, log_level={}",
                 opts.port, opts.host, opts.data_dir, opts.max_upload_mb, opts.max_tasks,
                 opts.task_ttl_seconds, opts.log_level);

    // Load ColorDBs
    spdlog::info("Loading ColorDBs from: {}", opts.data_dir);
    ColorDBCache db_cache;
    try {
        db_cache.LoadFromDirectory(opts.data_dir);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load ColorDBs: {}", e.what());
        return 1;
    }
    spdlog::info("Loaded {} ColorDB(s)", db_cache.databases.size());

    // Load model package if specified
    std::optional<ModelPackage> model_pack;
    if (!opts.model_pack_path.empty()) {
        try {
            model_pack.emplace(ModelPackage::LoadFromJson(opts.model_pack_path));
            spdlog::info("Loaded model package: {}", model_pack->name);
        } catch (const std::exception& e) {
            spdlog::error("Failed to load model package: {}", e.what());
            return 1;
        }
    }

    // Create services
    TaskManager task_mgr(opts.max_tasks, opts.task_ttl_seconds);

    SessionManager session_mgr;
    session_mgr.ttl_seconds = opts.task_ttl_seconds;

    BoardCache board_cache;
    board_cache.ttl_seconds = 600;

    BoardGeometryCache geometry_cache;

    // Load 8-color recipe store
    EightColorRecipeStore recipe_store;
    {
        std::string recipes_dir = opts.recipes_dir;
        if (recipes_dir.empty()) {
            auto parent = std::filesystem::path(opts.data_dir).parent_path();
            auto candidate = parent / "recipes";
            if (std::filesystem::is_directory(candidate)) { recipes_dir = candidate.string(); }
        }
        if (!recipes_dir.empty()) {
            auto recipe_path = std::filesystem::path(recipes_dir) / "8color_boards.json";
            if (std::filesystem::is_regular_file(recipe_path)) {
                try {
                    recipe_store = EightColorRecipeStore::LoadFromFile(recipe_path.string());
                } catch (const std::exception& e) {
                    spdlog::warn("Failed to load 8-color recipes: {}", e.what());
                }
            } else {
                spdlog::info("No 8-color recipe file found at {}", recipe_path.string());
            }
        }
    }

    // Create HTTP server and context
    httplib::Server svr;
    svr.set_payload_max_length(static_cast<size_t>(opts.max_upload_mb) * 1024 * 1024);

    ServerContext ctx{svr, task_mgr, db_cache, model_pack ? &model_pack.value() : nullptr,
                      session_mgr, board_cache, geometry_cache, recipe_store};

    // Register routes
    RegisterHealthRoutes(ctx);
    RegisterConvertRoutes(ctx);
    RegisterColorDBRoutes(ctx);
    RegisterCalibrationRoutes(ctx);
    RegisterSessionRoutes(ctx);

    // Mount static files if specified
    if (!opts.web_dir.empty()) {
        if (std::filesystem::is_directory(opts.web_dir)) {
            svr.set_mount_point("/", opts.web_dir);
            spdlog::info("Serving static files from: {}", opts.web_dir);
        } else {
            spdlog::warn("--web directory does not exist: {}", opts.web_dir);
        }
    }

    // Error handler
    svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        AddCorsHeaders(req, res);
        json j = ErrorJson("Not found");
        if (res.status == 413) { j = ErrorJson("Payload too large"); }
        res.set_content(j.dump(), "application/json");
    });

    // Exception handler
    svr.set_exception_handler(
        [](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            AddCorsHeaders(req, res);
            std::string msg = "Internal server error";
            try {
                if (ep) { std::rethrow_exception(ep); }
            } catch (const std::exception& e) { msg = e.what(); } catch (...) {
            }
            spdlog::error("Unhandled exception: {}", msg);
            res.set_content(ErrorJson(msg).dump(), "application/json");
            res.status = 500;
        });

    // Request logger (skip noisy health-check endpoint)
    svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        if (req.path == "/api/health") { return; }
        auto elapsed = std::chrono::steady_clock::now() - req.start_time_;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        spdlog::info("{} {} {} {} {}ms", req.remote_addr, req.method, req.path, res.status, ms);
    });

    spdlog::info("Starting server on {}:{}", opts.host, opts.port);
    if (!svr.listen(opts.host, opts.port)) {
        spdlog::error("Failed to start server on {}:{}", opts.host, opts.port);
        return 1;
    }

    return 0;
}
