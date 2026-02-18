#pragma once

#include "server_context.h"
#include "http_utils.h"
#include "request_builder.h"

#include <spdlog/spdlog.h>

#include <string>
#include <vector>

inline void RegisterConvertRoutes(ServerContext& ctx) {
    // Submit conversion task
    ctx.server.Post(
        "/api/convert",
        [&ctx](const httplib::Request& req, httplib::Response& res) {
            AddCorsHeaders(req, res);

            if (!req.form.has_file("image")) {
                SetJsonResponse(res, ErrorJson("Missing required field: image"), 400);
                return;
            }

            httplib::FormData image_file = req.form.get_file("image");
            if (image_file.content.empty()) {
                SetJsonResponse(res, ErrorJson("Uploaded image is empty"), 400);
                return;
            }

            std::vector<uint8_t> image_buffer(image_file.content.begin(),
                                              image_file.content.end());
            std::string image_name = image_file.filename;
            if (image_name.empty()) { image_name = "upload"; }
            auto dot_pos = image_name.rfind('.');
            if (dot_pos != std::string::npos) { image_name = image_name.substr(0, dot_pos); }

            json params = json::object();
            if (req.form.has_field("params")) {
                std::string params_str = req.form.get_field("params");
                try {
                    params = json::parse(params_str);
                } catch (const json::exception& e) {
                    SetJsonResponse(
                        res, ErrorJson(std::string("Invalid params JSON: ") + e.what()), 400);
                    return;
                }
            }

            UserSession* session = nullptr;
            std::string token    = GetSessionToken(req);
            if (!token.empty()) { session = ctx.session_mgr.Find(token); }

            ConvertRequest convert_req;
            try {
                convert_req = BuildConvertRequest(params, image_buffer, image_name, ctx.db_cache,
                                                  ctx.model_pack, session);
            } catch (const std::exception& e) {
                SetJsonResponse(res, ErrorJson(std::string("Invalid parameters: ") + e.what()),
                                400);
                return;
            }

            std::string task_id = ctx.task_mgr.Submit(std::move(convert_req));
            spdlog::info("Task submitted: id={}, image={}, size={} bytes", task_id, image_name,
                         image_buffer.size());
            SetJsonResponse(res, json{{"task_id", task_id}}, 202);
        });

    // Get task status
    ctx.server.Get("/api/tasks/:id",
                   [&ctx](const httplib::Request& req, httplib::Response& res) {
                       AddCorsHeaders(req, res);
                       std::string id = req.path_params.at("id");
                       auto task      = ctx.task_mgr.GetTask(id);
                       if (!task.has_value()) {
                           SetJsonResponse(res, ErrorJson("Task not found"), 404);
                           return;
                       }
                       SetJsonResponse(res, TaskInfoToJson(task.value()));
                   });

    // Download 3MF result
    ctx.server.Get("/api/tasks/:id/result",
                   [&ctx](const httplib::Request& req, httplib::Response& res) {
                       AddCorsHeaders(req, res);
                       std::string id = req.path_params.at("id");

                       auto task = ctx.task_mgr.GetTask(id);
                       if (!task.has_value()) {
                           SetJsonResponse(res, ErrorJson("Task not found"), 404);
                           return;
                       }
                       if (task->status != TaskInfo::Status::Completed) {
                           SetJsonResponse(res, ErrorJson("Task not completed"), 409);
                           return;
                       }

                       const auto* buf = ctx.task_mgr.GetTaskResultBuffer(id, "model_3mf");
                       if (!buf || buf->empty()) {
                           SetJsonResponse(res, ErrorJson("3MF result not available"), 404);
                           return;
                       }

                       std::string filename =
                           (task->image_name.empty() ? task->id.substr(0, 8) : task->image_name) +
                           ".3mf";
                       SetBinaryResponse(
                           res, *buf,
                           "application/vnd.ms-package.3dmanufacturing-3dmodel+xml", filename);
                   });

    // Download preview image
    ctx.server.Get("/api/tasks/:id/preview",
                   [&ctx](const httplib::Request& req, httplib::Response& res) {
                       AddCorsHeaders(req, res);
                       std::string id = req.path_params.at("id");

                       auto task = ctx.task_mgr.GetTask(id);
                       if (!task.has_value()) {
                           SetJsonResponse(res, ErrorJson("Task not found"), 404);
                           return;
                       }
                       if (task->status != TaskInfo::Status::Completed) {
                           SetJsonResponse(res, ErrorJson("Task not completed"), 409);
                           return;
                       }

                       const auto* buf = ctx.task_mgr.GetTaskResultBuffer(id, "preview_png");
                       if (!buf || buf->empty()) {
                           SetJsonResponse(res, ErrorJson("Preview not available"), 404);
                           return;
                       }
                       SetBinaryResponse(res, *buf, "image/png");
                   });

    // Download source mask image
    ctx.server.Get("/api/tasks/:id/source-mask",
                   [&ctx](const httplib::Request& req, httplib::Response& res) {
                       AddCorsHeaders(req, res);
                       std::string id = req.path_params.at("id");

                       auto task = ctx.task_mgr.GetTask(id);
                       if (!task.has_value()) {
                           SetJsonResponse(res, ErrorJson("Task not found"), 404);
                           return;
                       }
                       if (task->status != TaskInfo::Status::Completed) {
                           SetJsonResponse(res, ErrorJson("Task not completed"), 409);
                           return;
                       }

                       const auto* buf =
                           ctx.task_mgr.GetTaskResultBuffer(id, "source_mask_png");
                       if (!buf || buf->empty()) {
                           SetJsonResponse(res, ErrorJson("Source mask not available"), 404);
                           return;
                       }
                       SetBinaryResponse(res, *buf, "image/png");
                   });

    // Delete task
    ctx.server.Delete("/api/tasks/:id",
                      [&ctx](const httplib::Request& req, httplib::Response& res) {
                          AddCorsHeaders(req, res);
                          std::string id = req.path_params.at("id");
                          if (ctx.task_mgr.DeleteTask(id)) {
                              SetJsonResponse(res, json{{"deleted", true}});
                          } else {
                              auto task = ctx.task_mgr.GetTask(id);
                              if (!task.has_value()) {
                                  SetJsonResponse(res, ErrorJson("Task not found"), 404);
                              } else {
                                  SetJsonResponse(res, ErrorJson("Cannot delete running task"),
                                                  409);
                              }
                          }
                      });

    // List all tasks
    ctx.server.Get("/api/tasks",
                   [&ctx](const httplib::Request& req, httplib::Response& res) {
                       AddCorsHeaders(req, res);
                       auto tasks = ctx.task_mgr.ListTasks();
                       json arr   = json::array();
                       for (const auto& t : tasks) { arr.push_back(TaskInfoToJson(t)); }
                       SetJsonResponse(res, json{{"tasks", arr}});
                   });
}
