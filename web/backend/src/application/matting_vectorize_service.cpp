#include "application/matting_vectorize_service.h"

#include "chromaprint3d/matting_postprocess.h"

#include <neroued/vectorizer/vectorizer.h>
#include <spdlog/spdlog.h>

namespace chromaprint3d::backend {

using json             = nlohmann::json;
using VectorizerConfig = neroued::vectorizer::VectorizerConfig;

MattingVectorizeService::MattingVectorizeService(const ServerConfig& cfg, DataRepository& data,
                                                 TaskRuntime& tasks)
    : cfg_(cfg), data_(data), tasks_(tasks) {}

ServiceResult MattingVectorizeService::SubmitMatting(const std::string& owner,
                                                     const std::vector<uint8_t>& image,
                                                     const std::string& image_name,
                                                     const std::string& method) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto accept = tasks_.CanAccept(owner);
    if (!accept.ok) {
        return ServiceResult::Error(accept.status_code, "queue_rejected", accept.message);
    }
    auto valid = ValidateDecodedImage(image, cfg_.max_pixels_per_image);
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

ServiceResult MattingVectorizeService::PostprocessMatting(const std::string& owner,
                                                          const std::string& task_id,
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

ServiceResult MattingVectorizeService::SubmitVectorize(
    const std::string& owner, const std::vector<uint8_t>& image, const std::string& image_name,
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
    auto valid = ValidateDecodedImage(image, cfg_.max_pixels_per_image);
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

ServiceResult MattingVectorizeService::BuildVectorizeConfig(const json& params,
                                                            VectorizerConfig& out) const {
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
