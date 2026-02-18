#pragma once

#include "server_context.h"
#include "http_utils.h"

#include "chromaprint3d/calib.h"

#include <spdlog/spdlog.h>

#include <string>
#include <vector>

inline void RegisterCalibrationRoutes(ServerContext& ctx) {
    // Generate calibration board
    ctx.server.Post(
        "/api/calibration/generate-board",
        [&ctx](const httplib::Request& req, httplib::Response& res) {
            AddCorsHeaders(req, res);
            ctx.session_mgr.CleanExpired();

            json body;
            try {
                body = json::parse(req.body);
            } catch (const json::exception& e) {
                SetJsonResponse(res, ErrorJson(std::string("Invalid JSON: ") + e.what()), 400);
                return;
            }

            if (!body.contains("palette") || !body["palette"].is_array()) {
                SetJsonResponse(res, ErrorJson("Missing or invalid 'palette' array"), 400);
                return;
            }

            CalibrationBoardConfig cfg;
            auto& palette_arr       = body["palette"];
            cfg.recipe.num_channels = static_cast<int>(palette_arr.size());
            if (cfg.recipe.num_channels < CalibrationRecipeSpec::kMinChannels ||
                cfg.recipe.num_channels > CalibrationRecipeSpec::kMaxChannels) {
                SetJsonResponse(res, ErrorJson("palette size must be 2-8"), 400);
                return;
            }
            cfg.palette.resize(palette_arr.size());
            for (size_t i = 0; i < palette_arr.size(); ++i) {
                const auto& ch          = palette_arr[i];
                cfg.palette[i].color    = ch.value("color", "");
                cfg.palette[i].material = ch.value("material", "PLA Basic");
            }

            if (body.contains("color_layers")) {
                cfg.recipe.color_layers = body["color_layers"].get<int>();
            }
            if (body.contains("layer_height_mm")) {
                cfg.layer_height_mm = body["layer_height_mm"].get<float>();
            }

            try {
                const int num_ch = cfg.recipe.num_channels;
                const int c_layers = cfg.recipe.color_layers;

                CalibrationBoardResult result;
                const auto* cached = ctx.geometry_cache.Find(num_ch, c_layers);
                if (cached) {
                    spdlog::info("Geometry cache hit for {}ch/{}L", num_ch, c_layers);
                    result = BuildResultFromMeshes(*cached, cfg.palette);
                } else {
                    spdlog::info("Geometry cache miss for {}ch/{}L, generating...",
                                 num_ch, c_layers);
                    CalibrationBoardMeshes meshes = GenCalibrationBoardMeshes(cfg);
                    result = BuildResultFromMeshes(meshes, cfg.palette);
                    ctx.geometry_cache.Store(num_ch, c_layers, std::move(meshes));
                }

                std::string meta_str = result.meta.ToJsonString();
                std::string board_id = ctx.board_cache.Store(std::move(result));
                spdlog::info("Calibration board generated: id={}", board_id.substr(0, 8));
                json resp;
                resp["board_id"] = board_id;
                resp["meta"]     = json::parse(meta_str);
                SetJsonResponse(res, resp);
            } catch (const std::exception& e) {
                spdlog::error("Board generation failed: {}", e.what());
                SetJsonResponse(res, ErrorJson(std::string("Generation failed: ") + e.what()),
                                500);
            }
        });

    // Download calibration board 3MF
    ctx.server.Get("/api/calibration/boards/:id/3mf",
                   [&ctx](const httplib::Request& req, httplib::Response& res) {
                       AddCorsHeaders(req, res);
                       std::string id    = req.path_params.at("id");
                       const auto* entry = ctx.board_cache.Find(id);
                       if (!entry) {
                           SetJsonResponse(res, ErrorJson("Board not found or expired"), 404);
                           return;
                       }
                       std::string filename = entry->meta.name.empty()
                                                  ? "calibration_board"
                                                  : entry->meta.name;
                       filename += ".3mf";
                       SetBinaryResponse(
                           res, entry->model_3mf,
                           "application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
                           filename);
                   });

    // Download calibration board meta JSON
    ctx.server.Get("/api/calibration/boards/:id/meta",
                   [&ctx](const httplib::Request& req, httplib::Response& res) {
                       AddCorsHeaders(req, res);
                       std::string id    = req.path_params.at("id");
                       const auto* entry = ctx.board_cache.Find(id);
                       if (!entry) {
                           SetJsonResponse(res, ErrorJson("Board not found or expired"), 404);
                           return;
                       }
                       std::string filename = entry->meta.name.empty()
                                                  ? "calibration_board_meta"
                                                  : entry->meta.name + "_meta";
                       filename += ".json";
                       std::string meta_json = entry->meta.ToJsonString();
                       res.set_content(meta_json, "application/json");
                       res.set_header("Content-Disposition",
                                      "attachment; filename=\"" + filename + "\"");
                       res.status = 200;
                   });

    // Generate 8-color calibration board from pre-computed recipes
    ctx.server.Post(
        "/api/calibration/generate-8color-board",
        [&ctx](const httplib::Request& req, httplib::Response& res) {
            AddCorsHeaders(req, res);
            ctx.session_mgr.CleanExpired();

            if (!ctx.recipe_store.loaded) {
                SetJsonResponse(res, ErrorJson("8-color recipe data not available on server"), 503);
                return;
            }

            json body;
            try {
                body = json::parse(req.body);
            } catch (const json::exception& e) {
                SetJsonResponse(res, ErrorJson(std::string("Invalid JSON: ") + e.what()), 400);
                return;
            }

            if (!body.contains("palette") || !body["palette"].is_array()) {
                SetJsonResponse(res, ErrorJson("Missing or invalid 'palette' array"), 400);
                return;
            }
            if (!body.contains("board_index") || !body["board_index"].is_number_integer()) {
                SetJsonResponse(res, ErrorJson("Missing or invalid 'board_index'"), 400);
                return;
            }

            auto& palette_arr = body["palette"];
            const int num_ch  = static_cast<int>(palette_arr.size());
            if (num_ch != ctx.recipe_store.num_channels) {
                SetJsonResponse(
                    res,
                    ErrorJson("palette size must be " +
                              std::to_string(ctx.recipe_store.num_channels)),
                    400);
                return;
            }

            const int board_index = body["board_index"].get<int>();
            const auto* board_set = ctx.recipe_store.FindBoard(board_index);
            if (!board_set) {
                SetJsonResponse(res, ErrorJson("Invalid board_index: " +
                                               std::to_string(board_index)),
                                400);
                return;
            }

            CalibrationBoardConfig cfg;
            cfg.recipe.num_channels = num_ch;
            cfg.recipe.color_layers = ctx.recipe_store.color_layers;
            cfg.recipe.layer_order =
                (ctx.recipe_store.layer_order == "Bottom2Top") ? LayerOrder::Bottom2Top
                                                               : LayerOrder::Top2Bottom;
            cfg.base_layers      = ctx.recipe_store.base_layers;
            cfg.base_channel_idx = ctx.recipe_store.base_channel_idx;
            cfg.layer_height_mm  = ctx.recipe_store.layer_height_mm;
            cfg.layout.line_width_mm = ctx.recipe_store.line_width_mm;
            cfg.palette.resize(palette_arr.size());
            for (size_t i = 0; i < palette_arr.size(); ++i) {
                const auto& ch          = palette_arr[i];
                cfg.palette[i].color    = ch.value("color", "");
                cfg.palette[i].material = ch.value("material", "PLA Basic");
            }

            try {
                const int c_layers = cfg.recipe.color_layers;

                CalibrationBoardResult result;
                const auto* cached =
                    ctx.geometry_cache.Find(num_ch, c_layers, board_index);
                if (cached) {
                    spdlog::info("Geometry cache hit for 8ch board {}", board_index);
                    result = BuildResultFromMeshes(*cached, cfg.palette);
                } else {
                    spdlog::info("Generating 8ch board {} ({}x{}, {} recipes)...",
                                 board_index, board_set->grid_rows, board_set->grid_cols,
                                 board_set->recipes.size());
                    CalibrationBoardMeta meta = BuildCalibrationBoardMetaCustom(
                        cfg, board_set->grid_rows, board_set->grid_cols, board_set->recipes);
                    meta.name += "_board" + std::to_string(board_index);
                    CalibrationBoardMeshes meshes =
                        GenCalibrationBoardMeshesFromMeta(std::move(meta));
                    result = BuildResultFromMeshes(meshes, cfg.palette);
                    ctx.geometry_cache.Store(num_ch, c_layers, std::move(meshes),
                                            board_index);
                }

                std::string meta_str = result.meta.ToJsonString();
                std::string board_id = ctx.board_cache.Store(std::move(result));
                spdlog::info("8-color board {} generated: id={}", board_index,
                             board_id.substr(0, 8));
                json resp;
                resp["board_id"] = board_id;
                resp["meta"]     = json::parse(meta_str);
                SetJsonResponse(res, resp);
            } catch (const std::exception& e) {
                spdlog::error("8-color board generation failed: {}", e.what());
                SetJsonResponse(res,
                                ErrorJson(std::string("Generation failed: ") + e.what()), 500);
            }
        });

    // Build ColorDB from calibration photo
    ctx.server.Post(
        "/api/calibration/build-colordb",
        [&ctx](const httplib::Request& req, httplib::Response& res) {
            AddCorsHeaders(req, res);
            std::string token = EnsureSession(req, res, ctx.session_mgr);

            if (!req.form.has_file("image")) {
                SetJsonResponse(res, ErrorJson("Missing required field: image"), 400);
                return;
            }
            if (!req.form.has_file("meta")) {
                SetJsonResponse(res, ErrorJson("Missing required field: meta"), 400);
                return;
            }

            std::string db_name;
            if (req.form.has_field("name")) { db_name = req.form.get_field("name"); }
            if (db_name.empty() || !IsValidDBName(db_name)) {
                SetJsonResponse(
                    res,
                    ErrorJson("Invalid ColorDB name (alphanumeric and underscore only, "
                              "1-64 chars)"),
                    400);
                return;
            }

            httplib::FormData image_file = req.form.get_file("image");
            httplib::FormData meta_file  = req.form.get_file("meta");
            if (image_file.content.empty()) {
                SetJsonResponse(res, ErrorJson("Image file is empty"), 400);
                return;
            }
            if (meta_file.content.empty()) {
                SetJsonResponse(res, ErrorJson("Meta file is empty"), 400);
                return;
            }

            std::vector<uint8_t> image_buffer(image_file.content.begin(),
                                              image_file.content.end());

            CalibrationBoardMeta meta;
            try {
                meta = CalibrationBoardMeta::FromJsonString(meta_file.content);
            } catch (const std::exception& e) {
                SetJsonResponse(res, ErrorJson(std::string("Invalid meta JSON: ") + e.what()),
                                400);
                return;
            }

            try {
                ColorDB new_db = GenColorDBFromBuffer(image_buffer, meta);
                new_db.name    = db_name;

                UserSession* session = ctx.session_mgr.Find(token);
                if (!session) {
                    SetJsonResponse(res, ErrorJson("Session error"), 500);
                    return;
                }
                if (static_cast<int>(session->color_dbs.size()) >= kMaxSessionColorDBs) {
                    SetJsonResponse(
                        res,
                        ErrorJson("Session ColorDB limit reached (max " +
                                  std::to_string(kMaxSessionColorDBs) + ")"),
                        429);
                    return;
                }

                json db_json                    = ColorDBInfoToJson(new_db);
                db_json["source"]               = "session";
                session->color_dbs[db_name]     = std::move(new_db);

                spdlog::info("ColorDB built: name={}, session={}", db_name,
                             token.substr(0, 8));
                SetJsonResponse(res, db_json);
            } catch (const std::exception& e) {
                spdlog::error("ColorDB build failed: {}", e.what());
                SetJsonResponse(
                    res, ErrorJson(std::string("ColorDB build failed: ") + e.what()), 500);
            }
        });
}
