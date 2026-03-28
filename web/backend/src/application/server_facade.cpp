#include "application/server_facade.h"

#include "chromaprint3d/calib.h"
#include "chromaprint3d/image_io.h"
#include "chromaprint3d/pipeline.h"
#include "chromaprint3d/print_profile.h"
#include "chromaprint3d/shape_width_analyzer.h"
#include "chromaprint3d/slicer_preset.h"
#include "chromaprint3d/version.h"

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>

namespace chromaprint3d::backend {

using json = nlohmann::json;
using namespace ChromaPrint3D;
using neroued::vectorizer::VectorizerConfig;

namespace {

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

PrintMode ParsePrintMode(const std::string& value) {
    if (value == "0.08x5" || value == "0p08x5") return PrintMode::Mode0p08x5;
    if (value == "0.04x10" || value == "0p04x10") return PrintMode::Mode0p04x10;
    throw std::runtime_error("Invalid print_mode: " + value);
}

ColorSpace ParseColorSpace(const std::string& value) {
    const std::string lowered = ToLowerAscii(value);
    if (lowered == "lab") return ColorSpace::Lab;
    if (lowered == "rgb") return ColorSpace::Rgb;
    throw std::runtime_error("Invalid color_space: " + value);
}

ClusterMethod ParseClusterMethod(const std::string& value) {
    const std::string lowered = ToLowerAscii(value);
    if (lowered == "slic") return ClusterMethod::Slic;
    if (lowered == "kmeans" || lowered == "k_means" || lowered == "k-means") {
        return ClusterMethod::KMeans;
    }
    throw std::runtime_error("Invalid cluster_method: " + value);
}

const char* ClusterMethodToString(ClusterMethod method) {
    switch (method) {
    case ClusterMethod::Slic:
        return "slic";
    case ClusterMethod::KMeans:
        return "kmeans";
    }
    return "slic";
}

std::string StripExtension(const std::string& filename) {
    auto dot = filename.rfind('.');
    if (dot == std::string::npos) return filename;
    return filename.substr(0, dot);
}

const char* LayerOrderToString(LayerOrder order) {
    return order == LayerOrder::Bottom2Top ? "Bottom2Top" : "Top2Bottom";
}

json LayerPreviewsToJson(const LayerPreviewResult& layer_previews) {
    json palette = json::array();
    for (const auto& channel : layer_previews.palette) {
        palette.push_back({
            {"channel_idx", channel.channel_idx},
            {"color", channel.color},
            {"material", channel.material},
        });
    }

    json artifacts = json::array();
    for (std::size_t i = 0; i < layer_previews.layer_pngs.size(); ++i) {
        artifacts.push_back("layer-preview-" + std::to_string(i));
    }

    return {
        {"enabled", !layer_previews.layer_pngs.empty()},
        {"layers", layer_previews.layers},
        {"width", layer_previews.width},
        {"height", layer_previews.height},
        {"layer_order", LayerOrderToString(layer_previews.layer_order)},
        {"palette", std::move(palette)},
        {"artifacts", std::move(artifacts)},
    };
}

} // namespace

ServiceResult ServiceResult::Success(int status, json d) {
    ServiceResult r;
    r.ok          = true;
    r.status_code = status;
    r.data        = std::move(d);
    return r;
}

ServiceResult ServiceResult::Error(int status, std::string c, std::string m) {
    ServiceResult r;
    r.ok          = false;
    r.status_code = status;
    r.code        = std::move(c);
    r.message     = std::move(m);
    return r;
}

ServerFacade::ServerFacade(ServerConfig cfg)
    : cfg_(std::move(cfg)), data_(cfg_),
      sessions_(cfg_.session_ttl_seconds, cfg_.max_session_colordbs),
      tasks_(cfg_.max_tasks, cfg_.max_task_queue, cfg_.task_ttl_seconds, cfg_.max_tasks_per_owner,
             cfg_.max_task_result_mb * 1024 * 1024, cfg_.data_dir),
      boards_(cfg_.board_cache_ttl, cfg_.data_dir) {}

std::string ServerFacade::EnsureSession(const std::optional<std::string>& existing_token,
                                        bool* created) {
    return sessions_.EnsureSession(existing_token, created);
}

std::optional<SessionSnapshot> ServerFacade::SessionSnapshotByToken(const std::string& token) {
    return sessions_.Snapshot(token);
}

ServiceResult ServerFacade::Health() const {
    return ServiceResult::Success(200, {
                                           {"status", "ok"},
                                           {"version", CHROMAPRINT3D_VERSION_STRING},
                                           {"active_tasks", tasks_.ActiveTaskCount()},
                                           {"total_tasks", tasks_.TotalTaskCount()},
                                       });
}

ServiceResult ServerFacade::ConvertDefaults() const {
    ConvertRasterRequest defaults;
    return ServiceResult::Success(
        200, {
                 {"scale", defaults.scale},
                 {"max_width", defaults.max_width},
                 {"max_height", defaults.max_height},
                 {"target_width_mm", defaults.target_width_mm},
                 {"target_height_mm", defaults.target_height_mm},
                 {"print_mode", "0.08x5"},
                 {"color_space", "lab"},
                 {"k_candidates", defaults.k_candidates},
                 {"cluster_method", ClusterMethodToString(defaults.cluster_method)},
                 {"cluster_count", defaults.cluster_count},
                 {"slic_target_superpixels", defaults.slic_target_superpixels},
                 {"slic_compactness", defaults.slic_compactness},
                 {"slic_iterations", defaults.slic_iterations},
                 {"slic_min_region_ratio", defaults.slic_min_region_ratio},
                 {"dither", "none"},
                 {"dither_strength", defaults.dither_strength},
                 {"model_enable", defaults.model_enable},
                 {"model_only", defaults.model_only},
                 {"model_threshold", defaults.model_threshold},
                 {"model_margin", defaults.model_margin},
                 {"flip_y", defaults.flip_y},
                 {"pixel_mm", defaults.pixel_mm},
                 {"layer_height_mm", defaults.layer_height_mm},
                 {"base_layers", nullptr},
                 {"double_sided", defaults.double_sided},
                 {"transparent_layer_mm", defaults.transparent_layer_mm},
                 {"generate_preview", defaults.generate_preview},
                 {"generate_source_mask", defaults.generate_source_mask},
             });
}

ServiceResult ServerFacade::VectorizeDefaults() const {
    VectorizerConfig cfg;
    return ServiceResult::Success(200, {
                                           {"num_colors", cfg.num_colors},
                                           {"min_region_area", cfg.min_region_area},
                                           {"curve_fit_error", cfg.curve_fit_error},
                                           {"corner_angle_threshold", cfg.corner_angle_threshold},
                                           {"smoothing_spatial", cfg.smoothing_spatial},
                                           {"smoothing_color", cfg.smoothing_color},
                                           {"upscale_short_edge", cfg.upscale_short_edge},
                                           {"max_working_pixels", cfg.max_working_pixels},
                                           {"slic_region_size", cfg.slic_region_size},
                                           {"edge_sensitivity", cfg.edge_sensitivity},
                                           {"refine_passes", cfg.refine_passes},
                                           {"max_merge_color_dist", cfg.max_merge_color_dist},
                                           {"thin_line_max_radius", cfg.thin_line_max_radius},
                                           {"min_contour_area", cfg.min_contour_area},
                                           {"min_hole_area", cfg.min_hole_area},
                                           {"contour_simplify", cfg.contour_simplify},
                                           {"enable_coverage_fix", cfg.enable_coverage_fix},
                                           {"min_coverage_ratio", cfg.min_coverage_ratio},
                                           {"smoothness", cfg.smoothness},
                                           {"detail_level", cfg.detail_level},
                                           {"merge_segment_tolerance", cfg.merge_segment_tolerance},
                                           {"enable_antialias_detect", cfg.enable_antialias_detect},
                                           {"aa_tolerance", cfg.aa_tolerance},
                                           {"svg_enable_stroke", cfg.svg_enable_stroke},
                                           {"svg_stroke_width", cfg.svg_stroke_width},
                                       });
}

ServiceResult ServerFacade::ListDatabases(const std::optional<std::string>& session_token) {
    json arr = json::array();
    for (const auto& entry : data_.ColorDbCache().databases) {
        auto j      = ColorDbInfoToJson(entry.db, entry.material_type, entry.vendor);
        j["source"] = "global";
        arr.push_back(std::move(j));
    }
    if (session_token && !session_token->empty()) {
        auto dbs = sessions_.ListSessionDbs(*session_token);
        for (const auto& db : dbs) {
            auto j      = ColorDbInfoToJson(db);
            j["source"] = "session";
            arr.push_back(std::move(j));
        }
    }
    return ServiceResult::Success(200, {{"databases", std::move(arr)}});
}

ServiceResult ServerFacade::ListSessionDatabases(const std::string& token) {
    if (token.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    json arr = json::array();
    for (const auto& db : sessions_.ListSessionDbs(token)) {
        auto j      = ColorDbInfoToJson(db);
        j["source"] = "session";
        arr.push_back(std::move(j));
    }
    return ServiceResult::Success(200, {{"databases", std::move(arr)}});
}

ServiceResult ServerFacade::UploadSessionDatabase(const std::string& token,
                                                  const std::string& json_content,
                                                  const std::optional<std::string>& override_name) {
    if (token.empty()) return ServiceResult::Error(401, "unauthorized", "No session");

    ColorDB db;
    try {
        db = ColorDB::FromJsonString(json_content);
    } catch (const std::exception& e) {
        return ServiceResult::Error(400, "invalid_colordb",
                                    "Invalid ColorDB JSON: " + std::string(e.what()));
    }
    if (override_name && !override_name->empty()) db.name = *override_name;
    if (!SessionStore::IsValidDbName(db.name)) {
        return ServiceResult::Error(400, "invalid_name",
                                    "ColorDB name must be [a-zA-Z0-9_], length 1-64");
    }
    if (data_.ColorDbCache().FindByName(db.name)) {
        return ServiceResult::Error(409, "name_conflict", "Name conflicts with global database");
    }
    std::string err;
    if (!sessions_.PutSessionDb(token, db, &err)) {
        return ServiceResult::Error(429, "session_limit", err);
    }
    auto out      = ColorDbInfoToJson(db);
    out["source"] = "session";
    return ServiceResult::Success(200, std::move(out));
}

ServiceResult ServerFacade::DeleteSessionDatabase(const std::string& token,
                                                  const std::string& name) {
    if (token.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto existing = sessions_.GetSessionDb(token, name);
    if (!existing) return ServiceResult::Error(404, "not_found", "ColorDB not found in session");
    sessions_.RemoveSessionDb(token, name);
    return ServiceResult::Success(200, {{"deleted", true}});
}

ServiceResult ServerFacade::DownloadSessionDatabase(const std::string& token,
                                                    const std::string& name, std::string& json_out,
                                                    std::string& filename_out) {
    if (token.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto existing = sessions_.GetSessionDb(token, name);
    if (!existing) return ServiceResult::Error(404, "not_found", "ColorDB not found in session");
    json_out     = existing->ToJsonString();
    filename_out = name + ".json";
    return ServiceResult::Success(200, json::object());
}

ServiceResult ServerFacade::SubmitConvertRaster(const std::string& owner,
                                                const std::vector<uint8_t>& image,
                                                const std::string& image_name,
                                                const std::optional<std::string>& params_json) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto accept = tasks_.CanAccept(owner);
    if (!accept.ok) {
        return ServiceResult::Error(accept.status_code, "queue_rejected", accept.message);
    }
    auto valid = ValidateDecodedImage(image);
    if (!valid.ok) return valid;

    json params;
    auto parsed = ParseJsonObject(params_json, params);
    if (!parsed.ok) return parsed;

    auto session = sessions_.Snapshot(owner);

    ConvertRasterRequest req;
    auto built = BuildRasterRequest(params, image, image_name, session, req);
    if (!built.ok) return built;

    auto submit = tasks_.SubmitConvertRaster(owner, std::move(req), StripExtension(image_name));
    if (!submit.ok) {
        return ServiceResult::Error(submit.status_code, "submit_failed", submit.message);
    }
    return ServiceResult::Success(202, {{"task_id", submit.task_id}, {"kind", "convert"}});
}

ServiceResult ServerFacade::SubmitConvertVector(const std::string& owner,
                                                const std::vector<uint8_t>& svg,
                                                const std::string& svg_name,
                                                const std::optional<std::string>& params_json) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto accept = tasks_.CanAccept(owner);
    if (!accept.ok) {
        return ServiceResult::Error(accept.status_code, "queue_rejected", accept.message);
    }

    json params;
    auto parsed = ParseJsonObject(params_json, params);
    if (!parsed.ok) return parsed;
    auto session = sessions_.Snapshot(owner);

    ConvertVectorRequest req;
    auto built = BuildVectorRequest(params, svg, svg_name, session, req);
    if (!built.ok) return built;

    auto submit = tasks_.SubmitConvertVector(owner, std::move(req), StripExtension(svg_name));
    if (!submit.ok) {
        return ServiceResult::Error(submit.status_code, "submit_failed", submit.message);
    }
    return ServiceResult::Success(202, {{"task_id", submit.task_id}, {"kind", "convert"}});
}

ServiceResult ServerFacade::AnalyzeVectorWidth(const std::vector<uint8_t>& svg,
                                               const std::optional<std::string>& params_json) {
    if (svg.empty()) { return ServiceResult::Error(400, "invalid_request", "SVG buffer is empty"); }

    json params;
    auto parsed = ParseJsonObject(params_json, params);
    if (!parsed.ok) return parsed;

    float min_area       = params.value("min_area_mm2", 1.0f);
    float min_area_ratio = params.value("min_area_ratio", 0.001f);
    float raster_res     = params.value("raster_px_per_mm", 10.0f);

    VectorProcConfig vpc;
    vpc.flip_y = params.value("flip_y", false);

    try {
        VectorProc vproc(vpc);
        auto vpr = vproc.RunFromBuffer(svg, "analyze");

        float total_area         = vpr.width_mm * vpr.height_mm;
        float effective_min_area = std::max(min_area, min_area_ratio * total_area);
        auto analysis            = AnalyzeShapeWidths(vpr, effective_min_area, raster_res);

        json shapes_json = json::array();
        for (const auto& s : analysis.shapes) {
            uint8_t r8, g8, b8;
            s.color.ToRgb255(r8, g8, b8);
            char hex[8];
            std::snprintf(hex, sizeof(hex), "#%02X%02X%02X", r8, g8, b8);

            shapes_json.push_back({
                {"index", s.index},
                {"color", std::string(hex)},
                {"area_mm2", std::round(s.area_mm2 * 100.0f) / 100.0f},
                {"min_width_mm", std::round(s.min_width_mm * 1000.0f) / 1000.0f},
                {"median_width_mm", std::round(s.median_width_mm * 1000.0f) / 1000.0f},
            });
        }

        return ServiceResult::Success(
            200, {
                     {"image_width_mm", analysis.image_width_mm},
                     {"image_height_mm", analysis.image_height_mm},
                     {"total_shapes", analysis.total_shapes},
                     {"filtered_count", analysis.filtered_count},
                     {"min_area_threshold_mm2", analysis.min_area_threshold_mm2},
                     {"shapes", shapes_json},
                 });
    } catch (const std::exception& e) {
        spdlog::error("AnalyzeVectorWidth failed: {}", e.what());
        return ServiceResult::Error(500, "analysis_failed", e.what());
    }
}

ServiceResult ServerFacade::SubmitMatting(const std::string& owner,
                                          const std::vector<uint8_t>& image,
                                          const std::string& image_name,
                                          const std::string& method) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto accept = tasks_.CanAccept(owner);
    if (!accept.ok) {
        return ServiceResult::Error(accept.status_code, "queue_rejected", accept.message);
    }
    auto valid = ValidateDecodedImage(image);
    if (!valid.ok) return valid;

    if (!data_.MattingRegistry().Has(method)) {
        return ServiceResult::Error(400, "invalid_method", "Unknown matting method: " + method);
    }
    auto submit = tasks_.SubmitMatting(owner, image, method, StripExtension(image_name),
                                       data_.MattingRegistry());
    if (!submit.ok) {
        return ServiceResult::Error(submit.status_code, "submit_failed", submit.message);
    }
    return ServiceResult::Success(202, {{"task_id", submit.task_id}, {"kind", "matting"}});
}

ServiceResult ServerFacade::PostprocessMatting(const std::string& owner, const std::string& task_id,
                                               const std::string& body_json) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");

    json body;
    try {
        body = json::parse(body_json);
    } catch (const json::parse_error& e) {
        return ServiceResult::Error(400, "invalid_json",
                                    std::string("JSON parse error: ") + e.what());
    }
    if (!body.is_object()) {
        return ServiceResult::Error(400, "invalid_json", "Expected JSON object");
    }

    ChromaPrint3D::MattingPostprocessParams params;
    if (body.contains("threshold")) params.threshold = body["threshold"].get<float>();
    if (body.contains("morph_close_size"))
        params.morph_close_size = body["morph_close_size"].get<int>();
    if (body.contains("morph_close_iterations"))
        params.morph_close_iterations = body["morph_close_iterations"].get<int>();
    if (body.contains("min_region_area"))
        params.min_region_area = body["min_region_area"].get<int>();
    if (body.contains("reframe")) {
        const auto& rf = body["reframe"];
        if (!rf.is_object()) {
            return ServiceResult::Error(400, "invalid_params", "reframe must be an object");
        }
        if (rf.contains("enabled")) params.reframe_enabled = rf["enabled"].get<bool>();
        if (rf.contains("padding_px")) params.reframe_padding_px = rf["padding_px"].get<int>();
    }
    if (body.contains("outline") && body["outline"].is_object()) {
        const auto& ol = body["outline"];
        if (ol.contains("enabled")) params.outline_enabled = ol["enabled"].get<bool>();
        if (ol.contains("width")) params.outline_width = ol["width"].get<int>();
        if (ol.contains("color") && ol["color"].is_array() && ol["color"].size() == 3) {
            params.outline_color = {ol["color"][0].get<uint8_t>(), ol["color"][1].get<uint8_t>(),
                                    ol["color"][2].get<uint8_t>()};
        }
        if (ol.contains("mode")) {
            auto mode_str = ol["mode"].get<std::string>();
            if (mode_str == "inner")
                params.outline_mode = ChromaPrint3D::OutlineMode::Inner;
            else if (mode_str == "outer")
                params.outline_mode = ChromaPrint3D::OutlineMode::Outer;
            else
                params.outline_mode = ChromaPrint3D::OutlineMode::Center;
        }
    }
    if (params.reframe_padding_px < 0 || params.reframe_padding_px > 4096) {
        return ServiceResult::Error(400, "invalid_params",
                                    "reframe.padding_px must be in [0,4096]");
    }

    int status          = 500;
    std::string message = "Unknown error";
    bool ok             = tasks_.PostprocessMatting(owner, task_id, params, status, message);
    if (!ok) return ServiceResult::Error(status, "postprocess_failed", message);

    json artifacts = json::array({"processed-mask", "processed-foreground"});
    if (params.outline_enabled) artifacts.push_back("outline");

    return ServiceResult::Success(200, {{"artifacts", std::move(artifacts)}});
}

ServiceResult ServerFacade::SubmitVectorize(const std::string& owner,
                                            const std::vector<uint8_t>& image,
                                            const std::string& image_name,
                                            const std::optional<std::string>& params_json) {
    const auto params_bytes = params_json.has_value() ? params_json->size() : 0U;
    spdlog::info("SubmitVectorize requested: image='{}', bytes={}, params_bytes={}, owner_len={}",
                 image_name, image.size(), params_bytes, owner.size());
    if (owner.empty()) {
        spdlog::warn("SubmitVectorize rejected: no session");
        return ServiceResult::Error(401, "unauthorized", "No session");
    }
    auto accept = tasks_.CanAccept(owner);
    if (!accept.ok) {
        spdlog::warn("SubmitVectorize rejected by queue gate: status={}, reason={}",
                     accept.status_code, accept.message);
        return ServiceResult::Error(accept.status_code, "queue_rejected", accept.message);
    }
    auto valid = ValidateDecodedImage(image);
    if (!valid.ok) {
        spdlog::warn("SubmitVectorize rejected: invalid image, status={}, reason={}",
                     valid.status_code, valid.message);
        return valid;
    }

    json params;
    auto parsed = ParseJsonObject(params_json, params);
    if (!parsed.ok) {
        spdlog::warn("SubmitVectorize rejected: invalid params json, status={}, reason={}",
                     parsed.status_code, parsed.message);
        return parsed;
    }

    VectorizerConfig cfg;
    auto built = BuildVectorizeConfig(params, cfg);
    if (!built.ok) {
        spdlog::warn("SubmitVectorize rejected: invalid vectorize config, status={}, reason={}",
                     built.status_code, built.message);
        return built;
    }
    spdlog::debug(
        "SubmitVectorize config: num_colors={}, min_region_area={}, curve_fit_error={:.3f}, "
        "corner_angle_threshold={:.1f}, smoothing_spatial={:.1f}, smoothing_color={:.1f}, "
        "upscale_short_edge={}, max_working_pixels={}, slic_region_size={}, "
        "edge_sensitivity={:.2f}, refine_passes={}, max_merge_color_dist={:.1f}, "
        "svg_enable_stroke={}, svg_stroke_width={:.2f}",
        cfg.num_colors, cfg.min_region_area, cfg.curve_fit_error, cfg.corner_angle_threshold,
        cfg.smoothing_spatial, cfg.smoothing_color, cfg.upscale_short_edge, cfg.max_working_pixels,
        cfg.slic_region_size, cfg.edge_sensitivity, cfg.refine_passes, cfg.max_merge_color_dist,
        cfg.svg_enable_stroke, cfg.svg_stroke_width);

    auto submit = tasks_.SubmitVectorize(owner, image, cfg, StripExtension(image_name));
    if (!submit.ok) {
        spdlog::warn("SubmitVectorize failed to enqueue: status={}, reason={}", submit.status_code,
                     submit.message);
        return ServiceResult::Error(submit.status_code, "submit_failed", submit.message);
    }
    spdlog::info("SubmitVectorize accepted: task_id={}, image='{}'", submit.task_id, image_name);
    return ServiceResult::Success(202, {{"task_id", submit.task_id}, {"kind", "vectorize"}});
}

ServiceResult ServerFacade::SubmitConvertRasterMatchOnly(
    const std::string& owner, const std::vector<uint8_t>& image, const std::string& image_name,
    const std::optional<std::string>& params_json) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto accept = tasks_.CanAccept(owner);
    if (!accept.ok) {
        return ServiceResult::Error(accept.status_code, "queue_rejected", accept.message);
    }
    auto valid = ValidateDecodedImage(image);
    if (!valid.ok) return valid;

    json params;
    auto parsed = ParseJsonObject(params_json, params);
    if (!parsed.ok) return parsed;

    auto session = sessions_.Snapshot(owner);

    ConvertRasterRequest req;
    auto built = BuildRasterRequest(params, image, image_name, session, req);
    if (!built.ok) return built;

    if (req.dither != ChromaPrint3D::DitherMethod::None) {
        return ServiceResult::Error(400, "invalid_request", "Recipe editing requires dither=None");
    }

    auto submit =
        tasks_.SubmitConvertRasterMatchOnly(owner, std::move(req), StripExtension(image_name));
    if (!submit.ok) {
        return ServiceResult::Error(submit.status_code, "submit_failed", submit.message);
    }
    return ServiceResult::Success(202, {{"task_id", submit.task_id}, {"kind", "convert"}});
}

ServiceResult ServerFacade::SubmitConvertVectorMatchOnly(
    const std::string& owner, const std::vector<uint8_t>& svg, const std::string& svg_name,
    const std::optional<std::string>& params_json) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto accept = tasks_.CanAccept(owner);
    if (!accept.ok) {
        return ServiceResult::Error(accept.status_code, "queue_rejected", accept.message);
    }

    json params;
    auto parsed = ParseJsonObject(params_json, params);
    if (!parsed.ok) return parsed;
    auto session = sessions_.Snapshot(owner);

    ConvertVectorRequest req;
    auto built = BuildVectorRequest(params, svg, svg_name, session, req);
    if (!built.ok) return built;

    auto submit =
        tasks_.SubmitConvertVectorMatchOnly(owner, std::move(req), StripExtension(svg_name));
    if (!submit.ok) {
        return ServiceResult::Error(submit.status_code, "submit_failed", submit.message);
    }
    return ServiceResult::Success(202, {{"task_id", submit.task_id}, {"kind", "convert"}});
}

ServiceResult ServerFacade::RecipeEditorSummary(const std::string& owner,
                                                const std::string& task_id) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto summary = tasks_.GetRecipeEditorSummary(owner, task_id);
    if (!summary) return ServiceResult::Error(404, "not_found", "Summary not available");
    return ServiceResult::Success(200, std::move(*summary));
}

ServiceResult ServerFacade::RecipeEditorAlternatives(const std::string& owner,
                                                     const std::string& task_id,
                                                     const std::string& body_json) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");

    json body;
    try {
        body = json::parse(body_json);
    } catch (...) { return ServiceResult::Error(400, "invalid_request", "Invalid JSON body"); }

    if (!body.contains("target_lab") || !body["target_lab"].is_object()) {
        return ServiceResult::Error(400, "invalid_request", "Missing target_lab");
    }
    const auto& tlab = body["target_lab"];
    ChromaPrint3D::Lab target(tlab.value("L", 0.0f), tlab.value("a", 0.0f), tlab.value("b", 0.0f));

    int max_candidates = body.value("max_candidates", 10);
    int offset         = body.value("offset", 0);

    const ChromaPrint3D::ModelPackage* model_pack = nullptr;
    if (data_.ModelPack().has_value()) { model_pack = &(*data_.ModelPack()); }

    auto result =
        tasks_.QueryRecipeAlternatives(owner, task_id, target, max_candidates, offset, model_pack);
    if (!result) return ServiceResult::Error(404, "not_found", "Alternatives not available");
    return ServiceResult::Success(200, {{"candidates", std::move(*result)}});
}

ServiceResult ServerFacade::RecipeEditorReplace(const std::string& owner,
                                                const std::string& task_id,
                                                const std::string& body_json) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");

    json body;
    try {
        body = json::parse(body_json);
    } catch (...) { return ServiceResult::Error(400, "invalid_request", "Invalid JSON body"); }

    if (!body.contains("region_ids") || !body["region_ids"].is_array()) {
        return ServiceResult::Error(400, "invalid_request", "Missing region_ids array");
    }
    if (!body.contains("new_recipe") || !body["new_recipe"].is_array()) {
        return ServiceResult::Error(400, "invalid_request", "Missing new_recipe array");
    }
    if (!body.contains("new_mapped_lab") || !body["new_mapped_lab"].is_object()) {
        return ServiceResult::Error(400, "invalid_request", "Missing new_mapped_lab");
    }

    std::vector<int> region_ids;
    for (const auto& v : body["region_ids"]) { region_ids.push_back(v.get<int>()); }

    std::vector<uint8_t> new_recipe;
    for (const auto& v : body["new_recipe"]) { new_recipe.push_back(v.get<uint8_t>()); }

    const auto& lab_obj = body["new_mapped_lab"];
    ChromaPrint3D::Lab new_mapped(lab_obj.value("L", 0.0f), lab_obj.value("a", 0.0f),
                                  lab_obj.value("b", 0.0f));

    bool new_from_model = body.value("from_model", false);

    int status          = 500;
    std::string message = "Unknown error";
    json out_summary;
    bool ok = tasks_.ReplaceRecipe(owner, task_id, region_ids, new_recipe, new_mapped,
                                   new_from_model, status, message, out_summary);
    if (!ok) return ServiceResult::Error(status, "replace_failed", message);
    return ServiceResult::Success(200, std::move(out_summary));
}

ServiceResult ServerFacade::RecipeEditorGenerate(const std::string& owner,
                                                 const std::string& task_id) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");

    const ChromaPrint3D::ModelPackage* model_pack = nullptr;
    if (data_.ModelPack().has_value()) { model_pack = &(*data_.ModelPack()); }

    auto submit = tasks_.SubmitGenerateModel(owner, task_id, model_pack);
    if (!submit.ok) {
        return ServiceResult::Error(submit.status_code, "generate_failed", submit.message);
    }
    return ServiceResult::Success(202, {{"task_id", submit.task_id}});
}

ServiceResult ServerFacade::ListTasks(const std::string& owner) const {
    if (owner.empty()) return ServiceResult::Success(200, {{"tasks", json::array()}});
    json tasks = json::array();
    for (const auto& t : tasks_.ListTasks(owner)) tasks.push_back(TaskToJson(t));
    return ServiceResult::Success(200, {{"tasks", std::move(tasks)}});
}

ServiceResult ServerFacade::GetTask(const std::string& owner, const std::string& id) const {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto task = tasks_.FindTask(owner, id);
    if (!task) return ServiceResult::Error(404, "not_found", "Task not found");
    return ServiceResult::Success(200, TaskToJson(*task));
}

ServiceResult ServerFacade::DeleteTask(const std::string& owner, const std::string& id) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    int status          = 500;
    std::string message = "Unknown error";
    bool ok             = tasks_.DeleteTask(owner, id, status, message);
    if (!ok) return ServiceResult::Error(status, "delete_failed", message);
    return ServiceResult::Success(200, {{"deleted", true}});
}

ServiceResult ServerFacade::DownloadTaskArtifact(const std::string& owner, const std::string& id,
                                                 const std::string& artifact,
                                                 TaskArtifact& out) const {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    int status          = 500;
    std::string message = "Unknown error";
    auto data           = tasks_.LoadArtifact(owner, id, artifact, status, message);
    if (!data) return ServiceResult::Error(status, "artifact_error", message);
    out = std::move(*data);
    return ServiceResult::Success(200, json::object());
}

ServiceResult ServerFacade::MattingMethods() const {
    json arr = json::array();
    for (const auto& key : data_.MattingRegistry().Available()) {
        auto* provider = data_.MattingRegistry().Get(key);
        arr.push_back({
            {"key", key},
            {"name", provider ? provider->Name() : key},
            {"description", provider ? provider->Description() : ""},
        });
    }
    return ServiceResult::Success(200, {{"methods", std::move(arr)}});
}

ServiceResult ServerFacade::GenerateBoard(const json& body) {
    if (!body.contains("palette") || !body["palette"].is_array()) {
        return ServiceResult::Error(400, "invalid_request", "Missing or invalid 'palette' array");
    }
    CalibrationBoardConfig cfg;
    auto& palette_arr       = body["palette"];
    cfg.recipe.num_channels = static_cast<int>(palette_arr.size());
    if (cfg.recipe.num_channels < CalibrationRecipeSpec::kMinChannels ||
        cfg.recipe.num_channels > CalibrationRecipeSpec::kMaxChannels) {
        return ServiceResult::Error(400, "invalid_request", "palette size must be 2-8");
    }
    cfg.palette.resize(palette_arr.size());
    for (size_t i = 0; i < palette_arr.size(); ++i) {
        const auto& ch          = palette_arr[i];
        cfg.palette[i].color    = ch.value("color", "");
        cfg.palette[i].material = ch.value("material", "PLA Basic");
    }
    if (body.contains("color_layers")) cfg.recipe.color_layers = body["color_layers"].get<int>();
    if (body.contains("layer_height_mm")) {
        cfg.layer_height_mm = body["layer_height_mm"].get<float>();
    }

    std::string raw_nozzle = body.value("nozzle_size", "n04");
    std::string raw_face   = body.value("face_orientation", "faceup");
    NozzleSize nozzle      = ParseNozzleSize(raw_nozzle);
    FaceOrientation face   = ParseFaceOrientation(raw_face);
    spdlog::info("GenerateBoard: nozzle_size='{}' -> {}, face_orientation='{}' -> {}", raw_nozzle,
                 NozzleSizeTag(nozzle), raw_face, FaceOrientationTag(face));

    auto presets_path = std::filesystem::path(cfg_.data_dir) / "presets";
    bool has_preset   = std::filesystem::is_directory(presets_path);

    try {
        const int num_ch   = cfg.recipe.num_channels;
        const int c_layers = cfg.recipe.color_layers;

        CalibrationBoardResult result;
        auto cached = boards_.FindGeometry(num_ch, c_layers);

        if (has_preset) {
            FilamentConfig fil_config = FilamentConfig::LoadFromDir(presets_path.string());
            PrintProfile profile;
            profile.layer_height_mm  = cfg.layer_height_mm;
            profile.palette          = cfg.palette;
            profile.nozzle_size      = nozzle;
            profile.face_orientation = face;
            for (size_t i = 0; i < profile.palette.size(); ++i) {
                auto& ch = profile.palette[i];
                if (ch.hex_color.empty()) {
                    ch.hex_color = fil_config.ResolveHexColor(ch.color, static_cast<int>(i));
                }
            }
            cfg.palette = profile.palette;
            auto preset = SlicerPreset::FromProfile(presets_path.string(), profile, &fil_config);

            if (cached) {
                result = BuildResultFromMeshes(*cached, cfg.palette, preset, face);
            } else {
                auto meshes = GenCalibrationBoardMeshes(cfg);
                result      = BuildResultFromMeshes(meshes, cfg.palette, preset, face);
                boards_.StoreGeometry(num_ch, c_layers, std::move(meshes));
            }
        } else {
            if (cached) {
                result = BuildResultFromMeshes(*cached, cfg.palette, face);
            } else {
                auto meshes = GenCalibrationBoardMeshes(cfg);
                result      = BuildResultFromMeshes(meshes, cfg.palette, face);
                boards_.StoreGeometry(num_ch, c_layers, std::move(meshes));
            }
        }

        std::string meta_str = result.meta.ToJsonString();
        std::string board_id = boards_.StoreBoard(std::move(result));
        return ServiceResult::Success(200,
                                      {{"board_id", board_id}, {"meta", json::parse(meta_str)}});
    } catch (const std::exception& e) {
        return ServiceResult::Error(500, "generate_failed", e.what());
    }
}

ServiceResult ServerFacade::Generate8ColorBoard(const json& body) {
    if (!data_.RecipeStore().loaded) {
        return ServiceResult::Error(503, "service_unavailable",
                                    "8-color recipe data not available on server");
    }
    if (!body.contains("palette") || !body["palette"].is_array()) {
        return ServiceResult::Error(400, "invalid_request", "Missing or invalid 'palette' array");
    }
    if (!body.contains("board_index") || !body["board_index"].is_number_integer()) {
        return ServiceResult::Error(400, "invalid_request", "Missing or invalid 'board_index'");
    }

    auto& palette_arr = body["palette"];
    const int num_ch  = static_cast<int>(palette_arr.size());
    if (num_ch != data_.RecipeStore().num_channels) {
        return ServiceResult::Error(400, "invalid_request",
                                    "palette size must be " +
                                        std::to_string(data_.RecipeStore().num_channels));
    }

    const int board_index = body["board_index"].get<int>();
    const auto* board_set = data_.RecipeStore().FindBoard(board_index);
    if (!board_set) {
        return ServiceResult::Error(400, "invalid_request",
                                    "Invalid board_index: " + std::to_string(board_index));
    }

    CalibrationBoardConfig cfg;
    cfg.recipe.num_channels     = num_ch;
    cfg.recipe.color_layers     = data_.RecipeStore().color_layers;
    cfg.recipe.layer_order      = (data_.RecipeStore().layer_order == "Bottom2Top")
                                      ? LayerOrder::Bottom2Top
                                      : LayerOrder::Top2Bottom;
    cfg.base_layers             = data_.RecipeStore().base_layers;
    cfg.base_channel_idx        = data_.RecipeStore().base_channel_idx;
    cfg.layer_height_mm         = data_.RecipeStore().layer_height_mm;
    cfg.layout.line_width_mm    = data_.RecipeStore().layout.line_width_mm;
    cfg.layout.resolution_scale = data_.RecipeStore().layout.resolution_scale;
    cfg.layout.tile_factor      = data_.RecipeStore().layout.tile_factor;
    cfg.layout.gap_factor       = data_.RecipeStore().layout.gap_factor;
    cfg.layout.margin_factor    = data_.RecipeStore().layout.margin_factor;
    cfg.palette.resize(palette_arr.size());
    for (size_t i = 0; i < palette_arr.size(); ++i) {
        const auto& ch          = palette_arr[i];
        cfg.palette[i].color    = ch.value("color", "");
        cfg.palette[i].material = ch.value("material", "PLA Basic");
    }

    std::string raw_nozzle = body.value("nozzle_size", "n04");
    std::string raw_face   = body.value("face_orientation", "faceup");
    NozzleSize nozzle      = ParseNozzleSize(raw_nozzle);
    FaceOrientation face   = ParseFaceOrientation(raw_face);
    spdlog::info("Generate8ColorBoard: nozzle_size='{}' -> {}, face_orientation='{}' -> {}",
                 raw_nozzle, NozzleSizeTag(nozzle), raw_face, FaceOrientationTag(face));

    auto presets_path = std::filesystem::path(cfg_.data_dir) / "presets";
    bool has_preset   = std::filesystem::is_directory(presets_path);

    try {
        CalibrationBoardResult result;
        auto cached = boards_.FindGeometry(num_ch, cfg.recipe.color_layers, board_index);

        auto build_with_preset = [&](const CalibrationBoardMeshes& meshes_ref) {
            if (has_preset) {
                FilamentConfig fil_config = FilamentConfig::LoadFromDir(presets_path.string());
                PrintProfile profile;
                profile.layer_height_mm  = cfg.layer_height_mm;
                profile.palette          = cfg.palette;
                profile.nozzle_size      = nozzle;
                profile.face_orientation = face;
                for (size_t i = 0; i < profile.palette.size(); ++i) {
                    auto& ch = profile.palette[i];
                    if (ch.hex_color.empty()) {
                        ch.hex_color = fil_config.ResolveHexColor(ch.color, static_cast<int>(i));
                    }
                }
                cfg.palette = profile.palette;
                auto preset =
                    SlicerPreset::FromProfile(presets_path.string(), profile, &fil_config);
                return BuildResultFromMeshes(meshes_ref, cfg.palette, preset, face);
            }
            return BuildResultFromMeshes(meshes_ref, cfg.palette, face);
        };

        if (cached) {
            result = build_with_preset(*cached);
        } else {
            auto meta = BuildCalibrationBoardMetaCustom(cfg, board_set->grid_rows,
                                                        board_set->grid_cols, board_set->recipes);
            meta.name += "_board" + std::to_string(board_index);
            auto meshes = GenCalibrationBoardMeshesFromMeta(std::move(meta));
            result      = build_with_preset(meshes);
            boards_.StoreGeometry(num_ch, cfg.recipe.color_layers, std::move(meshes), board_index);
        }

        std::string meta_str = result.meta.ToJsonString();
        std::string board_id = boards_.StoreBoard(std::move(result));
        return ServiceResult::Success(200,
                                      {{"board_id", board_id}, {"meta", json::parse(meta_str)}});
    } catch (const std::exception& e) {
        return ServiceResult::Error(500, "generate_failed", e.what());
    }
}

ServiceResult ServerFacade::DownloadBoardModel(const std::string& board_id, TaskArtifact& out) {
    auto board = boards_.FindBoard(board_id);
    if (!board) return ServiceResult::Error(404, "not_found", "Board not found or expired");
    std::string filename = board->meta.name.empty() ? "calibration_board" : board->meta.name;
    filename += ".3mf";

    constexpr const char* kModelContentType =
        "application/vnd.ms-package.3dmanufacturing-3dmodel+xml";
    if (board->has_file_backed_model()) {
        const auto& fpath = board->model_3mf_path;
        std::error_code ec;
        bool exists      = std::filesystem::exists(fpath, ec);
        auto actual_size = exists ? std::filesystem::file_size(fpath, ec) : 0ULL;
        spdlog::debug("DownloadBoardModel: board={}, path={}, exists={}, actual_size={}, "
                      "expected_size={}",
                      board_id, fpath.string(), exists, actual_size, board->expected_file_size);
        if (!exists) {
            spdlog::error("DownloadBoardModel: file missing for board {}: {}", board_id,
                          fpath.string());
            return ServiceResult::Error(404, "not_found", "Board model file missing from disk");
        }
        if (board->expected_file_size > 0 && actual_size != board->expected_file_size) {
            spdlog::error("DownloadBoardModel: file size mismatch for board {}: expected={}, "
                          "actual={}",
                          board_id, board->expected_file_size, actual_size);
            return ServiceResult::Error(500, "corrupt_file",
                                        "Board model file is incomplete or corrupt");
        }
        out = TaskArtifact{{}, fpath, kModelContentType, std::move(filename)};
        return ServiceResult::Success(200, json::object());
    }
    if (board->model_3mf.empty()) {
        return ServiceResult::Error(404, "not_found", "Board model not available");
    }
    out = TaskArtifact{std::move(board->model_3mf), {}, kModelContentType, std::move(filename)};
    return ServiceResult::Success(200, json::object());
}

ServiceResult ServerFacade::DownloadBoardMeta(const std::string& board_id, TaskArtifact& out) {
    auto board = boards_.FindBoard(board_id);
    if (!board) return ServiceResult::Error(404, "not_found", "Board not found or expired");
    std::string filename =
        board->meta.name.empty() ? "calibration_board_meta" : board->meta.name + "_meta";
    filename += ".json";
    auto meta_json = board->meta.ToJsonString();
    std::vector<uint8_t> bytes(meta_json.begin(), meta_json.end());
    TaskArtifact artifact;
    artifact.bytes        = std::move(bytes);
    artifact.content_type = "application/json";
    artifact.filename     = std::move(filename);
    out                   = std::move(artifact);
    return ServiceResult::Success(200, json::object());
}

ServiceResult ServerFacade::LocateBoard(const std::vector<uint8_t>& image,
                                        const std::string& meta_json) {
    auto valid = ValidateDecodedImage(image);
    if (!valid.ok) return valid;

    CalibrationBoardMeta meta;
    try {
        meta = CalibrationBoardMeta::FromJsonString(meta_json);
    } catch (const std::exception& e) {
        return ServiceResult::Error(400, "invalid_meta",
                                    "Invalid meta JSON: " + std::string(e.what()));
    }

    try {
        auto result      = LocateCalibrationBoard(image, meta);
        json corners_arr = json::array();
        for (const auto& c : result.corners) { corners_arr.push_back(json::array({c[0], c[1]})); }
        json out            = json::object();
        out["corners"]      = std::move(corners_arr);
        out["image_width"]  = result.image_width;
        out["image_height"] = result.image_height;
        return ServiceResult::Success(200, std::move(out));
    } catch (const std::exception& e) {
        return ServiceResult::Error(422, "locate_failed", e.what());
    }
}

ServiceResult ServerFacade::BuildColorDb(const std::string& owner,
                                         const std::vector<uint8_t>& image,
                                         const std::string& meta_json, const std::string& db_name,
                                         const std::string& corners_json) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto valid = ValidateDecodedImage(image);
    if (!valid.ok) return valid;
    if (!SessionStore::IsValidDbName(db_name)) {
        return ServiceResult::Error(400, "invalid_name",
                                    "Invalid ColorDB name ([a-zA-Z0-9_], length 1-64)");
    }
    if (data_.ColorDbCache().FindByName(db_name)) {
        return ServiceResult::Error(409, "name_conflict", "Name conflicts with global database");
    }

    CalibrationBoardMeta meta;
    try {
        meta = CalibrationBoardMeta::FromJsonString(meta_json);
    } catch (const std::exception& e) {
        return ServiceResult::Error(400, "invalid_meta",
                                    "Invalid meta JSON: " + std::string(e.what()));
    }

    std::optional<std::array<std::array<float, 2>, 4>> corners_opt;
    if (!corners_json.empty()) {
        try {
            auto j = json::parse(corners_json);
            if (!j.is_array() || j.size() != 4) {
                return ServiceResult::Error(400, "invalid_corners",
                                            "corners must be an array of 4 [x,y] pairs");
            }
            std::array<std::array<float, 2>, 4> pts;
            for (size_t i = 0; i < 4; ++i) {
                if (!j[i].is_array() || j[i].size() != 2) {
                    return ServiceResult::Error(400, "invalid_corners",
                                                "Each corner must be [x, y]");
                }
                pts[i] = {j[i][0].get<float>(), j[i][1].get<float>()};
            }
            corners_opt = pts;
        } catch (const json::exception& e) {
            return ServiceResult::Error(400, "invalid_corners",
                                        "Failed to parse corners: " + std::string(e.what()));
        }
    }

    try {
        ColorDB db = corners_opt ? GenColorDBFromBuffer(image, meta, *corners_opt)
                                 : GenColorDBFromBuffer(image, meta);
        db.name    = db_name;
        std::string err;
        if (!sessions_.PutSessionDb(owner, db, &err)) {
            return ServiceResult::Error(429, "session_limit", err);
        }
        auto out      = ColorDbBuildResultToJson(db);
        out["source"] = "session";
        return ServiceResult::Success(200, std::move(out));
    } catch (const std::exception& e) {
        return ServiceResult::Error(500, "build_failed", e.what());
    }
}

json ServerFacade::ColorDbInfoToJson(const ColorDB& db, const std::string& material_type,
                                     const std::string& vendor) {
    static const auto builtin_fil = FilamentConfig::BuiltinDefaults();
    json palette                  = json::array();
    for (size_t i = 0; i < db.palette.size(); ++i) {
        const auto& ch = db.palette[i];
        json entry     = {{"color", ch.color}, {"material", ch.material}};
        if (!ch.hex_color.empty()) {
            entry["hex_color"] = ch.hex_color;
        } else {
            auto resolved = builtin_fil.ResolveHexColor(ch.color, static_cast<int>(i));
            if (!resolved.empty()) entry["hex_color"] = resolved;
        }
        palette.push_back(std::move(entry));
    }
    return {
        {"name", db.name},
        {"num_channels", db.NumChannels()},
        {"num_entries", db.entries.size()},
        {"max_color_layers", db.max_color_layers},
        {"base_layers", db.base_layers},
        {"layer_height_mm", db.layer_height_mm},
        {"line_width_mm", db.line_width_mm},
        {"material_type", material_type},
        {"vendor", vendor},
        {"palette", std::move(palette)},
    };
}

json ServerFacade::ColorDbBuildResultToJson(const ColorDB& db) {
    auto out = ColorDbInfoToJson(db);

    json entries = json::array();
    for (const auto& entry : db.entries) {
        ChromaPrint3D::Rgb rgb = entry.lab.ToRgb();
        uint8_t r8 = 0, g8 = 0, b8 = 0;
        rgb.ToRgb255(r8, g8, b8);
        char hex_buf[8];
        std::snprintf(hex_buf, sizeof(hex_buf), "#%02X%02X%02X", r8, g8, b8);

        json recipe_arr = json::array();
        for (uint8_t v : entry.recipe) { recipe_arr.push_back(static_cast<int>(v)); }

        entries.push_back({
            {"lab", {entry.lab.x, entry.lab.y, entry.lab.z}},
            {"hex", std::string(hex_buf)},
            {"recipe", std::move(recipe_arr)},
        });
    }
    out["entries"] = std::move(entries);
    return out;
}

json ServerFacade::TaskToJson(const TaskSnapshot& task) {
    json j;
    j["id"]     = task.id;
    j["kind"]   = TaskKindToString(task.kind);
    j["status"] = TaskStatusToString(task.status);
    j["created_at_ms"] =
        std::chrono::duration_cast<std::chrono::milliseconds>(task.created_at.time_since_epoch())
            .count();
    j["error"] = task.error.empty() ? json(nullptr) : json(task.error);

    if (auto cp = std::get_if<ConvertTaskPayload>(&task.payload)) {
        j["stage"]      = ConvertStageToString(cp->stage);
        j["progress"]   = cp->progress;
        j["match_only"] = cp->match_only;
        if (!cp->generate_error.empty()) { j["generate_error"] = cp->generate_error; }
        if (task.status == RuntimeTaskStatus::Completed) {
            j["result"] = {
                {"input_width", cp->result.input_width},
                {"input_height", cp->result.input_height},
                {"resolved_pixel_mm", cp->result.resolved_pixel_mm},
                {"physical_width_mm", cp->result.physical_width_mm},
                {"physical_height_mm", cp->result.physical_height_mm},
                {"stats",
                 {{"clusters_total", cp->result.stats.clusters_total},
                  {"db_only", cp->result.stats.db_only},
                  {"model_fallback", cp->result.stats.model_fallback},
                  {"model_queries", cp->result.stats.model_queries},
                  {"avg_db_de", cp->result.stats.avg_db_de},
                  {"avg_model_de", cp->result.stats.avg_model_de}}},
                {"has_3mf", !cp->result.model_3mf.empty() || cp->has_3mf_on_disk},
                {"has_preview", !cp->result.preview_png.empty()},
                {"has_source_mask", !cp->result.source_mask_png.empty()},
                {"has_region_map", !cp->region_map_binary.empty()},
                {"layer_previews", LayerPreviewsToJson(cp->result.layer_previews)},
            };
        } else {
            j["result"] = nullptr;
        }
        return j;
    }

    if (auto mp = std::get_if<MattingTaskPayload>(&task.payload)) {
        j["method"] = mp->method;
        if (task.status == RuntimeTaskStatus::Completed) {
            j["width"]     = mp->width;
            j["height"]    = mp->height;
            j["has_alpha"] = mp->has_alpha;
            j["timing"]    = {
                {"preprocess_ms", mp->timing.preprocess_ms},
                {"inference_ms", mp->timing.inference_ms},
                {"postprocess_ms", mp->timing.postprocess_ms},
                {"decode_ms", mp->decode_ms},
                {"encode_ms", mp->encode_ms},
                {"provider_ms", mp->timing.total_ms},
                {"pipeline_ms", mp->pipeline_ms},
            };
        } else {
            j["width"]     = 0;
            j["height"]    = 0;
            j["has_alpha"] = false;
            j["timing"]    = nullptr;
        }
        return j;
    }

    auto* vp        = std::get_if<VectorizeTaskPayload>(&task.payload);
    j["timing"]     = nullptr;
    j["width"]      = 0;
    j["height"]     = 0;
    j["num_shapes"] = 0;
    j["svg_size"]   = 0;
    if (vp && task.status == RuntimeTaskStatus::Completed) {
        const auto svg_size      = vp->svg_bytes > 0 ? vp->svg_bytes : vp->svg_content.size();
        j["width"]               = vp->width;
        j["height"]              = vp->height;
        j["num_shapes"]          = vp->num_shapes;
        j["svg_size"]            = svg_size;
        j["resolved_num_colors"] = vp->resolved_num_colors;
        j["timing"]              = {
            {"decode_ms", vp->decode_ms},
            {"vectorize_ms", vp->vectorize_ms},
            {"pipeline_ms", vp->pipeline_ms},
        };
    }
    return j;
}

const char* ServerFacade::ConvertStageToString(ConvertStage stage) {
    switch (stage) {
    case ConvertStage::LoadingResources:
        return "loading_resources";
    case ConvertStage::Preprocessing:
        return "preprocessing";
    case ConvertStage::Matching:
        return "matching";
    case ConvertStage::BuildingModel:
        return "building_model";
    case ConvertStage::Exporting:
        return "exporting";
    }
    return "unknown";
}

ServiceResult ServerFacade::ParseJsonObject(const std::optional<std::string>& raw,
                                            json& out) const {
    out = json::object();
    if (!raw || raw->empty()) return ServiceResult::Success(200, json::object());
    try {
        out = json::parse(*raw);
        if (!out.is_object()) {
            return ServiceResult::Error(400, "invalid_json", "Expected JSON object");
        }
        return ServiceResult::Success(200, json::object());
    } catch (const std::exception& e) {
        return ServiceResult::Error(400, "invalid_json", std::string("Invalid JSON: ") + e.what());
    }
}

ServiceResult ServerFacade::ValidateDecodedImage(const std::vector<uint8_t>& image) const {
    cv::Mat decoded;
    try {
        decoded = ChromaPrint3D::DecodeImageWithIcc(image);
    } catch (...) { return ServiceResult::Error(400, "invalid_image", "Failed to decode image"); }
    if (decoded.empty())
        return ServiceResult::Error(400, "invalid_image", "Failed to decode image");
    std::int64_t pixels = static_cast<std::int64_t>(decoded.cols) * decoded.rows;
    if (pixels > cfg_.max_pixels_per_image) {
        return ServiceResult::Error(413, "image_too_large",
                                    "Decoded image exceeds max pixels limit");
    }
    return ServiceResult::Success(200, json::object());
}

ServiceResult ServerFacade::ResolveSelectedColorDbs(
    const json& params, const std::optional<SessionSnapshot>& session,
    std::vector<const ColorDB*>& out_dbs,
    std::vector<std::shared_ptr<const ColorDB>>& session_owned, bool& has_bambu_pla) const {
    out_dbs.clear();
    has_bambu_pla = false;

    if (params.contains("db_names") && params["db_names"].is_array()) {
        std::vector<const ColorDB*> selected;
        selected.reserve(params["db_names"].size());
        for (const auto& name_val : params["db_names"]) {
            if (!name_val.is_string()) {
                return ServiceResult::Error(400, "invalid_params",
                                            "db_names must be an array of strings");
            }
            const std::string name = name_val.get<std::string>();
            const ColorDB* db      = nullptr;
            if (const auto* entry = data_.ColorDbCache().FindEntryByName(name)) {
                db = &entry->db;
                if (entry->material_type == "PLA" && entry->vendor == "BambooLab") {
                    has_bambu_pla = true;
                }
            }
            if (!db && session) {
                auto it = session->color_dbs.find(name);
                if (it != session->color_dbs.end()) {
                    auto copy = std::make_shared<const ColorDB>(it->second);
                    session_owned.push_back(copy);
                    db = copy.get();
                }
            }
            if (!db) {
                return ServiceResult::Error(400, "invalid_params", "ColorDB not found: " + name);
            }
            selected.push_back(db);
        }
        if (selected.empty()) {
            return ServiceResult::Error(400, "invalid_params", "No valid ColorDB names provided");
        }
        out_dbs = std::move(selected);
        return ServiceResult::Success(200, json::object());
    }

    out_dbs = data_.ColorDbCache().GetAll();
    for (const auto& entry : data_.ColorDbCache().databases) {
        if (entry.material_type == "PLA" && entry.vendor == "BambooLab") {
            has_bambu_pla = true;
            break;
        }
    }
    return ServiceResult::Success(200, json::object());
}

ServiceResult ServerFacade::BuildRasterRequest(const json& params,
                                               const std::vector<uint8_t>& image,
                                               const std::string& image_name,
                                               const std::optional<SessionSnapshot>& session,
                                               ConvertRasterRequest& out) const {
    out.image_buffer = image;
    out.image_name   = image_name;

    auto presets_path = std::filesystem::path(cfg_.data_dir) / "presets";
    if (std::filesystem::is_directory(presets_path)) { out.preset_dir = presets_path.string(); }

    bool has_bambu_pla = false;
    auto selected      = ResolveSelectedColorDbs(params, session, out.preloaded_dbs,
                                                 out.session_owned_dbs, has_bambu_pla);
    if (!selected.ok) return selected;

    if (data_.ModelPack().has_value() && has_bambu_pla) {
        out.preloaded_model_pack = &data_.ModelPack().value();
    }

    try {
        if (params.contains("scale")) out.scale = params["scale"].get<float>();
        if (params.contains("max_width")) out.max_width = params["max_width"].get<int>();
        if (params.contains("max_height")) out.max_height = params["max_height"].get<int>();
        if (params.contains("target_width_mm")) {
            out.target_width_mm = params["target_width_mm"].get<float>();
        }
        if (params.contains("target_height_mm")) {
            out.target_height_mm = params["target_height_mm"].get<float>();
        }
        if (params.contains("print_mode")) {
            out.print_mode = ParsePrintMode(params["print_mode"].get<std::string>());
        }
        if (params.contains("color_space")) {
            out.color_space = ParseColorSpace(params["color_space"].get<std::string>());
        }
        if (params.contains("k_candidates")) out.k_candidates = params["k_candidates"].get<int>();
        if (params.contains("cluster_method")) {
            out.cluster_method = ParseClusterMethod(params["cluster_method"].get<std::string>());
        }
        if (params.contains("cluster_count"))
            out.cluster_count = params["cluster_count"].get<int>();
        if (params.contains("slic_target_superpixels")) {
            out.slic_target_superpixels = params["slic_target_superpixels"].get<int>();
        }
        if (params.contains("slic_compactness")) {
            out.slic_compactness = params["slic_compactness"].get<float>();
        }
        if (params.contains("slic_iterations")) {
            out.slic_iterations = params["slic_iterations"].get<int>();
        }
        if (params.contains("slic_min_region_ratio")) {
            out.slic_min_region_ratio = params["slic_min_region_ratio"].get<float>();
        }
        if (params.contains("dither")) {
            std::string d = params["dither"].get<std::string>();
            if (d == "blue_noise") {
                out.dither = DitherMethod::BlueNoise;
            } else if (d == "floyd_steinberg") {
                out.dither = DitherMethod::FloydSteinberg;
            } else if (d == "none") {
                out.dither = DitherMethod::None;
            } else {
                throw std::runtime_error("Invalid dither method: " + d);
            }
        }
        if (params.contains("dither_strength")) {
            out.dither_strength = params["dither_strength"].get<float>();
        }
        if (params.contains("allowed_channels") && params["allowed_channels"].is_array()) {
            for (const auto& ch : params["allowed_channels"]) {
                std::string color    = ch.value("color", "");
                std::string material = ch.value("material", "");
                if (!color.empty()) out.allowed_channel_keys.push_back(color + "|" + material);
            }
        }
        if (params.contains("model_enable")) out.model_enable = params["model_enable"].get<bool>();
        if (params.contains("model_only")) out.model_only = params["model_only"].get<bool>();
        if (params.contains("model_threshold")) {
            out.model_threshold = params["model_threshold"].get<float>();
        }
        if (params.contains("model_margin")) out.model_margin = params["model_margin"].get<float>();
        if (params.contains("flip_y")) out.flip_y = params["flip_y"].get<bool>();
        if (params.contains("pixel_mm")) out.pixel_mm = params["pixel_mm"].get<float>();
        if (params.contains("layer_height_mm")) {
            out.layer_height_mm = params["layer_height_mm"].get<float>();
        }
        if (params.contains("base_layers")) out.base_layers = params["base_layers"].get<int>();
        if (params.contains("double_sided")) out.double_sided = params["double_sided"].get<bool>();
        if (params.contains("transparent_layer_mm")) {
            out.transparent_layer_mm = params["transparent_layer_mm"].get<float>();
        }
        if (params.contains("generate_preview")) {
            out.generate_preview = params["generate_preview"].get<bool>();
        }
        if (params.contains("generate_source_mask")) {
            out.generate_source_mask = params["generate_source_mask"].get<bool>();
        }
        if (params.contains("nozzle_size")) {
            out.nozzle_size = ParseNozzleSize(params["nozzle_size"].get<std::string>());
        }
        if (params.contains("face_orientation")) {
            out.face_orientation =
                ParseFaceOrientation(params["face_orientation"].get<std::string>());
        }
    } catch (const std::exception& e) {
        return ServiceResult::Error(400, "invalid_params", e.what());
    }

    if (out.scale <= 0.0f) return ServiceResult::Error(400, "invalid_params", "scale must be > 0");
    if (out.max_width < 0 || out.max_height < 0) {
        return ServiceResult::Error(400, "invalid_params", "max dimensions must be >= 0");
    }
    if (out.k_candidates < 1) {
        return ServiceResult::Error(400, "invalid_params", "k_candidates must be >= 1");
    }
    if (out.cluster_count < 0) {
        return ServiceResult::Error(400, "invalid_params", "cluster_count must be >= 0");
    }
    if (out.slic_target_superpixels < 0) {
        return ServiceResult::Error(400, "invalid_params", "slic_target_superpixels must be >= 0");
    }
    if (out.slic_compactness <= 0.0f) {
        return ServiceResult::Error(400, "invalid_params", "slic_compactness must be > 0");
    }
    if (out.slic_iterations < 1) {
        return ServiceResult::Error(400, "invalid_params", "slic_iterations must be >= 1");
    }
    if (out.slic_min_region_ratio < 0.0f || out.slic_min_region_ratio > 1.0f) {
        return ServiceResult::Error(400, "invalid_params",
                                    "slic_min_region_ratio must be in [0,1]");
    }
    if (out.dither_strength < 0.0f || out.dither_strength > 1.0f) {
        return ServiceResult::Error(400, "invalid_params", "dither_strength must be in [0,1]");
    }
    if (out.base_layers < -1) {
        return ServiceResult::Error(400, "invalid_params", "base_layers must be >= -1");
    }
    if (out.transparent_layer_mm < 0.0f) {
        return ServiceResult::Error(400, "invalid_params", "transparent_layer_mm must be >= 0");
    }
    if (out.cluster_method == ClusterMethod::Slic && out.dither != DitherMethod::None) {
        out.dither = DitherMethod::None;
    }

    return ServiceResult::Success(200, json::object());
}

ServiceResult ServerFacade::BuildVectorRequest(const json& params, const std::vector<uint8_t>& svg,
                                               const std::string& svg_name,
                                               const std::optional<SessionSnapshot>& session,
                                               ConvertVectorRequest& out) const {
    out.svg_buffer = svg;
    out.svg_name   = svg_name;

    auto vpresets = std::filesystem::path(cfg_.data_dir) / "presets";
    if (std::filesystem::is_directory(vpresets)) { out.preset_dir = vpresets.string(); }

    bool has_bambu_pla = false;
    auto selected      = ResolveSelectedColorDbs(params, session, out.preloaded_dbs,
                                                 out.session_owned_dbs, has_bambu_pla);
    if (!selected.ok) return selected;

    if (data_.ModelPack().has_value() && has_bambu_pla) {
        out.preloaded_model_pack = &data_.ModelPack().value();
    }

    try {
        if (params.contains("target_width_mm")) {
            out.target_width_mm = params["target_width_mm"].get<float>();
        }
        if (params.contains("target_height_mm")) {
            out.target_height_mm = params["target_height_mm"].get<float>();
        }
        if (params.contains("print_mode")) {
            out.print_mode = ParsePrintMode(params["print_mode"].get<std::string>());
        }
        if (params.contains("color_space")) {
            out.color_space = ParseColorSpace(params["color_space"].get<std::string>());
        }
        if (params.contains("k_candidates")) out.k_candidates = params["k_candidates"].get<int>();
        if (params.contains("model_enable")) out.model_enable = params["model_enable"].get<bool>();
        if (params.contains("model_only")) out.model_only = params["model_only"].get<bool>();
        if (params.contains("model_threshold")) {
            out.model_threshold = params["model_threshold"].get<float>();
        }
        if (params.contains("model_margin")) out.model_margin = params["model_margin"].get<float>();
        if (params.contains("flip_y")) out.flip_y = params["flip_y"].get<bool>();
        if (params.contains("layer_height_mm")) {
            out.layer_height_mm = params["layer_height_mm"].get<float>();
        }
        if (params.contains("base_layers")) out.base_layers = params["base_layers"].get<int>();
        if (params.contains("double_sided")) out.double_sided = params["double_sided"].get<bool>();
        if (params.contains("transparent_layer_mm")) {
            out.transparent_layer_mm = params["transparent_layer_mm"].get<float>();
        }
        if (params.contains("tessellation_tolerance_mm")) {
            out.tessellation_tolerance_mm = params["tessellation_tolerance_mm"].get<float>();
        }
        if (params.contains("gradient_pixel_mm")) {
            out.gradient_pixel_mm = params["gradient_pixel_mm"].get<float>();
        }
        if (params.contains("allowed_channels") && params["allowed_channels"].is_array()) {
            for (const auto& ch : params["allowed_channels"]) {
                std::string color    = ch.value("color", "");
                std::string material = ch.value("material", "");
                if (!color.empty()) out.allowed_channel_keys.push_back(color + "|" + material);
            }
        }
        if (params.contains("generate_preview")) {
            out.generate_preview = params["generate_preview"].get<bool>();
        }
        if (params.contains("generate_source_mask")) {
            out.generate_source_mask = params["generate_source_mask"].get<bool>();
        }
        if (params.contains("nozzle_size")) {
            out.nozzle_size = ParseNozzleSize(params["nozzle_size"].get<std::string>());
        }
        if (params.contains("face_orientation")) {
            out.face_orientation =
                ParseFaceOrientation(params["face_orientation"].get<std::string>());
        }
    } catch (const std::exception& e) {
        return ServiceResult::Error(400, "invalid_params", e.what());
    }

    if (out.k_candidates < 1) {
        return ServiceResult::Error(400, "invalid_params", "k_candidates must be >= 1");
    }
    if (out.base_layers < -1) {
        return ServiceResult::Error(400, "invalid_params", "base_layers must be >= -1");
    }
    if (out.transparent_layer_mm < 0.0f) {
        return ServiceResult::Error(400, "invalid_params", "transparent_layer_mm must be >= 0");
    }

    return ServiceResult::Success(200, json::object());
}

ServiceResult ServerFacade::BuildVectorizeConfig(const json& params, VectorizerConfig& out) const {
    out = VectorizerConfig{};
    spdlog::debug("BuildVectorizeConfig: params keys={}", params.size());
    try {
        if (params.contains("num_colors")) out.num_colors = params["num_colors"].get<int>();
        if (params.contains("min_region_area")) {
            out.min_region_area = params["min_region_area"].get<int>();
        }
        if (params.contains("curve_fit_error")) {
            out.curve_fit_error = params["curve_fit_error"].get<float>();
        }
        if (params.contains("corner_angle_threshold")) {
            out.corner_angle_threshold = params["corner_angle_threshold"].get<float>();
        }
        if (params.contains("smoothing_spatial")) {
            out.smoothing_spatial = params["smoothing_spatial"].get<float>();
        }
        if (params.contains("smoothing_color")) {
            out.smoothing_color = params["smoothing_color"].get<float>();
        }
        if (params.contains("upscale_short_edge")) {
            out.upscale_short_edge = params["upscale_short_edge"].get<int>();
        }
        if (params.contains("max_working_pixels")) {
            out.max_working_pixels = params["max_working_pixels"].get<int>();
        }
        if (params.contains("slic_region_size")) {
            out.slic_region_size = params["slic_region_size"].get<int>();
        }
        if (params.contains("edge_sensitivity")) {
            out.edge_sensitivity = params["edge_sensitivity"].get<float>();
        }
        if (params.contains("refine_passes")) {
            out.refine_passes = params["refine_passes"].get<int>();
        }
        if (params.contains("max_merge_color_dist")) {
            out.max_merge_color_dist = params["max_merge_color_dist"].get<float>();
        }
        if (params.contains("thin_line_max_radius")) {
            out.thin_line_max_radius = params["thin_line_max_radius"].get<float>();
        }
        if (params.contains("min_contour_area")) {
            out.min_contour_area = params["min_contour_area"].get<float>();
        }
        if (params.contains("min_hole_area"))
            out.min_hole_area = params["min_hole_area"].get<float>();
        if (params.contains("contour_simplify")) {
            out.contour_simplify = params["contour_simplify"].get<float>();
        }
        if (params.contains("enable_coverage_fix")) {
            out.enable_coverage_fix = params["enable_coverage_fix"].get<bool>();
        }
        if (params.contains("min_coverage_ratio")) {
            out.min_coverage_ratio = params["min_coverage_ratio"].get<float>();
        }
        if (params.contains("svg_enable_stroke")) {
            out.svg_enable_stroke = params["svg_enable_stroke"].get<bool>();
        }
        if (params.contains("svg_stroke_width")) {
            out.svg_stroke_width = params["svg_stroke_width"].get<float>();
        }
        if (params.contains("smoothness")) { out.smoothness = params["smoothness"].get<float>(); }
        if (params.contains("detail_level")) {
            out.detail_level = params["detail_level"].get<float>();
        }
        if (params.contains("merge_segment_tolerance")) {
            out.merge_segment_tolerance = params["merge_segment_tolerance"].get<float>();
        }
        if (params.contains("enable_antialias_detect")) {
            out.enable_antialias_detect = params["enable_antialias_detect"].get<bool>();
        }
        if (params.contains("aa_tolerance")) {
            out.aa_tolerance = params["aa_tolerance"].get<float>();
        }
    } catch (const std::exception& e) {
        return ServiceResult::Error(400, "invalid_params", e.what());
    }

    if (out.num_colors != 0 && (out.num_colors < 2 || out.num_colors > 256)) {
        return ServiceResult::Error(400, "invalid_params",
                                    "num_colors must be 0 (auto) or [2,256]");
    }
    if (out.curve_fit_error < 0.1f || out.curve_fit_error > 5.0f) {
        return ServiceResult::Error(400, "invalid_params", "curve_fit_error must be in [0.1,5]");
    }
    if (out.corner_angle_threshold < 90.0f || out.corner_angle_threshold > 175.0f) {
        return ServiceResult::Error(400, "invalid_params",
                                    "corner_angle_threshold must be in [90,175]");
    }
    if (out.smoothing_spatial < 0.0f || out.smoothing_spatial > 50.0f) {
        return ServiceResult::Error(400, "invalid_params", "smoothing_spatial must be in [0,50]");
    }
    if (out.smoothing_color < 0.0f || out.smoothing_color > 80.0f) {
        return ServiceResult::Error(400, "invalid_params", "smoothing_color must be in [0,80]");
    }
    if (out.upscale_short_edge < 0 || out.upscale_short_edge > 2000) {
        return ServiceResult::Error(400, "invalid_params",
                                    "upscale_short_edge must be in [0,2000]");
    }
    if (out.max_working_pixels < 0 || out.max_working_pixels > 100000000) {
        return ServiceResult::Error(400, "invalid_params",
                                    "max_working_pixels must be in [0,100000000]");
    }
    if (out.slic_region_size < 0 || out.slic_region_size > 100) {
        return ServiceResult::Error(400, "invalid_params", "slic_region_size must be in [0,100]");
    }
    if (out.edge_sensitivity < 0.0f || out.edge_sensitivity > 1.0f) {
        return ServiceResult::Error(400, "invalid_params", "edge_sensitivity must be in [0,1]");
    }
    if (out.refine_passes < 0 || out.refine_passes > 20) {
        return ServiceResult::Error(400, "invalid_params", "refine_passes must be in [0,20]");
    }
    if (out.max_merge_color_dist < 0.0f || out.max_merge_color_dist > 2000.0f) {
        return ServiceResult::Error(400, "invalid_params",
                                    "max_merge_color_dist must be in [0,2000]");
    }
    if (out.thin_line_max_radius < 0.5f || out.thin_line_max_radius > 10.0f) {
        return ServiceResult::Error(400, "invalid_params",
                                    "thin_line_max_radius must be in [0.5,10]");
    }
    if (out.min_region_area < 0 || out.min_contour_area < 0.0f || out.min_hole_area < 0.0f) {
        return ServiceResult::Error(400, "invalid_params", "area options must be >= 0");
    }
    if (out.contour_simplify < 0.0f || out.contour_simplify > 10.0f) {
        return ServiceResult::Error(400, "invalid_params", "contour_simplify must be in [0,10]");
    }
    if (out.min_coverage_ratio < 0.0f || out.min_coverage_ratio > 1.0f) {
        return ServiceResult::Error(400, "invalid_params", "min_coverage_ratio must be in [0,1]");
    }
    if (out.svg_stroke_width < 0.0f || out.svg_stroke_width > 20.0f) {
        return ServiceResult::Error(400, "invalid_params", "svg_stroke_width must be in [0,20]");
    }
    if (out.smoothness < 0.0f || out.smoothness > 1.0f) {
        return ServiceResult::Error(400, "invalid_params", "smoothness must be in [0,1]");
    }
    if (out.detail_level > 1.0f) {
        return ServiceResult::Error(400, "invalid_params", "detail_level must be <= 1");
    }
    if (out.merge_segment_tolerance < 0.0f || out.merge_segment_tolerance > 0.5f) {
        return ServiceResult::Error(400, "invalid_params",
                                    "merge_segment_tolerance must be in [0,0.5]");
    }
    if (out.aa_tolerance < 1.0f || out.aa_tolerance > 50.0f) {
        return ServiceResult::Error(400, "invalid_params", "aa_tolerance must be in [1,50]");
    }
    spdlog::debug(
        "BuildVectorizeConfig resolved: num_colors={}, min_region_area={}, curve_fit_error={:.3f}, "
        "corner_angle_threshold={:.1f}, smoothing_spatial={:.1f}, smoothing_color={:.1f}, "
        "upscale_short_edge={}, max_working_pixels={}, slic_region_size={}, "
        "edge_sensitivity={:.2f}, refine_passes={}, max_merge_color_dist={:.1f}, "
        "thin_line_max_radius={:.2f}, "
        "min_contour_area={:.2f}, min_hole_area={:.2f}, contour_simplify={:.2f}, "
        "enable_coverage_fix={}, min_coverage_ratio={:.4f}, "
        "svg_enable_stroke={}, svg_stroke_width={:.2f}",
        out.num_colors, out.min_region_area, out.curve_fit_error, out.corner_angle_threshold,
        out.smoothing_spatial, out.smoothing_color, out.upscale_short_edge, out.max_working_pixels,
        out.slic_region_size, out.edge_sensitivity, out.refine_passes, out.max_merge_color_dist,
        out.thin_line_max_radius, out.min_contour_area, out.min_hole_area, out.contour_simplify,
        out.enable_coverage_fix, out.min_coverage_ratio, out.svg_enable_stroke,
        out.svg_stroke_width);
    return ServiceResult::Success(200, json::object());
}

} // namespace chromaprint3d::backend
