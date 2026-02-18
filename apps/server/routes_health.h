#pragma once

#include "server_context.h"
#include "http_utils.h"
#include "chromaprint3d/version.h"

inline void RegisterHealthRoutes(ServerContext& ctx) {
    ctx.server.Get("/api/health",
                   [&ctx](const httplib::Request& req, httplib::Response& res) {
                       AddCorsHeaders(req, res);
                       json j = {
                           {"status", "ok"},
                           {"version", CHROMAPRINT3D_VERSION_STRING},
                           {"active_tasks", ctx.task_mgr.ActiveTaskCount()},
                           {"total_tasks", ctx.task_mgr.TotalTaskCount()},
                       };
                       SetJsonResponse(res, j);
                   });
}
