#pragma once

#include "task_manager.h"

#include "chromaprint3d/color_db.h"

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

using namespace ChromaPrint3D;
using json = nlohmann::json;

inline json ErrorJson(const std::string& message) { return json{{"error", message}}; }

inline void SetJsonResponse(httplib::Response& res, const json& j, int status = 200) {
    res.set_content(j.dump(), "application/json");
    res.status = status;
}

inline void SetBinaryResponse(httplib::Response& res, const std::vector<uint8_t>& data,
                               const std::string& content_type, const std::string& filename = "") {
    res.set_content(std::string(reinterpret_cast<const char*>(data.data()), data.size()),
                    content_type);
    if (!filename.empty()) {
        res.set_header("Content-Disposition", "attachment; filename=\"" + filename + "\"");
    }
    res.status = 200;
}

inline void AddCorsHeaders(const httplib::Request& req, httplib::Response& res) {
    std::string origin = req.has_header("Origin") ? req.get_header_value("Origin") : "";
    if (!origin.empty()) {
        res.set_header("Access-Control-Allow-Origin", origin);
    } else {
        res.set_header("Access-Control-Allow-Origin", "*");
    }
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.set_header("Access-Control-Allow-Credentials", "true");
    res.set_header("Access-Control-Max-Age", "86400");
}

inline json TaskInfoToJson(const TaskInfo& info) {
    json j;
    j["id"]       = info.id;
    j["status"]   = TaskStatusToString(info.status);
    j["stage"]    = ConvertStageToString(info.stage);
    j["progress"] = info.progress;

    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(info.created_at.time_since_epoch())
            .count();
    j["created_at_ms"] = elapsed_ms;

    if (!info.error_message.empty()) {
        j["error"] = info.error_message;
    } else {
        j["error"] = nullptr;
    }

    if (info.status == TaskInfo::Status::Completed) {
        j["result"] = {
            {"image_width", info.result.image_width},
            {"image_height", info.result.image_height},
            {"stats",
             {{"clusters_total", info.result.stats.clusters_total},
              {"db_only", info.result.stats.db_only},
              {"model_fallback", info.result.stats.model_fallback},
              {"model_queries", info.result.stats.model_queries},
              {"avg_db_de", info.result.stats.avg_db_de},
              {"avg_model_de", info.result.stats.avg_model_de}}},
            {"has_3mf", !info.result.model_3mf.empty()},
            {"has_preview", !info.result.preview_png.empty()},
            {"has_source_mask", !info.result.source_mask_png.empty()},
        };
    } else {
        j["result"] = nullptr;
    }

    return j;
}

inline json ColorDBInfoToJson(const ColorDB& db) {
    json j;
    j["name"]             = db.name;
    j["num_channels"]     = db.NumChannels();
    j["num_entries"]      = db.entries.size();
    j["max_color_layers"] = db.max_color_layers;
    j["base_layers"]      = db.base_layers;
    j["layer_height_mm"]  = db.layer_height_mm;
    j["line_width_mm"]    = db.line_width_mm;

    json palette = json::array();
    for (const auto& ch : db.palette) {
        palette.push_back({{"color", ch.color}, {"material", ch.material}});
    }
    j["palette"] = palette;
    return j;
}
