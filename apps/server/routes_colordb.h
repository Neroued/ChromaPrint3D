#pragma once

#include "server_context.h"
#include "http_utils.h"

#include "chromaprint3d/pipeline.h"

inline void RegisterColorDBRoutes(ServerContext& ctx) {
    // List ColorDBs
    ctx.server.Get("/api/colordbs",
                   [&ctx](const httplib::Request& req, httplib::Response& res) {
                       AddCorsHeaders(req, res);
                       json databases = json::array();
                       for (const auto& db : ctx.db_cache.databases) {
                           json j      = ColorDBInfoToJson(db);
                           j["source"] = "global";
                           databases.push_back(j);
                       }
                       std::string token = GetSessionToken(req);
                       if (!token.empty()) {
                           UserSession* session = ctx.session_mgr.Find(token);
                           if (session) {
                               for (const auto& [name, db] : session->color_dbs) {
                                   json j      = ColorDBInfoToJson(db);
                                   j["source"] = "session";
                                   databases.push_back(j);
                               }
                           }
                       }
                       SetJsonResponse(res, json{{"databases", databases}});
                   });

    // Default config
    ctx.server.Get("/api/config/defaults",
                   [](const httplib::Request& req, httplib::Response& res) {
                       AddCorsHeaders(req, res);
                       ConvertRequest defaults;
                       json j = {
                           {"scale", defaults.scale},
                           {"max_width", defaults.max_width},
                           {"max_height", defaults.max_height},
                           {"print_mode", "0.08x5"},
                           {"color_space", "lab"},
                           {"k_candidates", defaults.k_candidates},
                           {"cluster_count", defaults.cluster_count},
                           {"model_enable", defaults.model_enable},
                           {"model_only", defaults.model_only},
                           {"model_threshold", defaults.model_threshold},
                           {"model_margin", defaults.model_margin},
                           {"flip_y", defaults.flip_y},
                           {"pixel_mm", defaults.pixel_mm},
                           {"layer_height_mm", defaults.layer_height_mm},
                           {"generate_preview", defaults.generate_preview},
                           {"generate_source_mask", defaults.generate_source_mask},
                       };
                       SetJsonResponse(res, j);
                   });
}
