#pragma once

#include "server_context.h"
#include "http_utils.h"

#include <spdlog/spdlog.h>

#include <string>

inline void RegisterSessionRoutes(ServerContext& ctx) {
    // List session ColorDBs
    ctx.server.Get("/api/session/colordbs",
                   [&ctx](const httplib::Request& req, httplib::Response& res) {
                       AddCorsHeaders(req, res);
                       std::string token = GetSessionToken(req);
                       json databases    = json::array();
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

    // Delete session ColorDB
    ctx.server.Delete(
        "/api/session/colordbs/:name",
        [&ctx](const httplib::Request& req, httplib::Response& res) {
            AddCorsHeaders(req, res);
            std::string token = GetSessionToken(req);
            if (token.empty()) {
                SetJsonResponse(res, ErrorJson("No session"), 401);
                return;
            }
            UserSession* session = ctx.session_mgr.Find(token);
            if (!session) {
                SetJsonResponse(res, ErrorJson("Session not found"), 404);
                return;
            }
            std::string name = req.path_params.at("name");
            auto it          = session->color_dbs.find(name);
            if (it == session->color_dbs.end()) {
                SetJsonResponse(res, ErrorJson("ColorDB not found in session"), 404);
                return;
            }
            session->color_dbs.erase(it);
            spdlog::info("Session ColorDB deleted: name={}, session={}", name,
                         token.substr(0, 8));
            SetJsonResponse(res, json{{"deleted", true}});
        });

    // Download session ColorDB as JSON
    ctx.server.Get(
        "/api/session/colordbs/:name/download",
        [&ctx](const httplib::Request& req, httplib::Response& res) {
            AddCorsHeaders(req, res);
            std::string token = GetSessionToken(req);
            if (token.empty()) {
                SetJsonResponse(res, ErrorJson("No session"), 401);
                return;
            }
            UserSession* session = ctx.session_mgr.Find(token);
            if (!session) {
                SetJsonResponse(res, ErrorJson("Session not found"), 404);
                return;
            }
            std::string name = req.path_params.at("name");
            auto it          = session->color_dbs.find(name);
            if (it == session->color_dbs.end()) {
                SetJsonResponse(res, ErrorJson("ColorDB not found in session"), 404);
                return;
            }
            std::string json_str = it->second.ToJsonString();
            res.set_content(json_str, "application/json");
            res.set_header("Content-Disposition",
                           "attachment; filename=\"" + name + ".json\"");
            res.status = 200;
        });

    // Upload ColorDB JSON to session
    ctx.server.Post(
        "/api/session/colordbs/upload",
        [&ctx](const httplib::Request& req, httplib::Response& res) {
            AddCorsHeaders(req, res);
            std::string token = EnsureSession(req, res, ctx.session_mgr);

            if (!req.form.has_file("file")) {
                SetJsonResponse(res, ErrorJson("Missing required field: file"), 400);
                return;
            }

            std::string db_name;
            if (req.form.has_field("name")) { db_name = req.form.get_field("name"); }

            httplib::FormData file_data = req.form.get_file("file");
            if (file_data.content.empty()) {
                SetJsonResponse(res, ErrorJson("\u4e0a\u4f20\u7684\u6587\u4ef6\u4e3a\u7a7a"), 400);
                return;
            }

            ColorDB new_db;
            try {
                new_db = ColorDB::FromJsonString(file_data.content);
            } catch (const std::exception& e) {
                SetJsonResponse(
                    res,
                    ErrorJson("\u65e0\u6548\u7684 ColorDB JSON \u6587\u4ef6: " + std::string(e.what())),
                    400);
                return;
            }

            if (!db_name.empty()) { new_db.name = db_name; }
            if (!IsValidDBName(new_db.name)) {
                SetJsonResponse(
                    res,
                    ErrorJson("ColorDB \u540d\u79f0\u65e0\u6548 (\u4ec5\u5141\u8bb8\u5b57\u6bcd\u3001\u6570\u5b57\u548c\u4e0b\u5212\u7ebf\uff0c1-64 \u5b57\u7b26)"),
                    400);
                return;
            }

            if (ctx.db_cache.FindByName(new_db.name)) {
                SetJsonResponse(
                    res,
                    ErrorJson("\u540d\u79f0 \"" + new_db.name +
                              "\" \u4e0e\u5168\u5c40\u6570\u636e\u5e93\u51b2\u7a81\uff0c\u8bf7\u4f7f\u7528\u5176\u4ed6\u540d\u79f0"),
                    409);
                return;
            }

            UserSession* session = ctx.session_mgr.Find(token);
            if (!session) {
                SetJsonResponse(res, ErrorJson("Session error"), 500);
                return;
            }
            if (static_cast<int>(session->color_dbs.size()) >= kMaxSessionColorDBs) {
                SetJsonResponse(
                    res,
                    ErrorJson("Session ColorDB \u6570\u91cf\u5df2\u8fbe\u4e0a\u9650 (\u6700\u591a " +
                              std::to_string(kMaxSessionColorDBs) + " \u4e2a)"),
                    429);
                return;
            }

            std::string final_name = new_db.name;
            json db_json           = ColorDBInfoToJson(new_db);
            db_json["source"]      = "session";
            session->color_dbs[final_name] = std::move(new_db);

            spdlog::info("ColorDB uploaded: name={}, session={}", final_name,
                         token.substr(0, 8));
            SetJsonResponse(res, db_json);
        });

    // CORS preflight handler
    ctx.server.Options("/(.*)", [](const httplib::Request& req, httplib::Response& res) {
        AddCorsHeaders(req, res);
        res.status = 204;
    });
}
