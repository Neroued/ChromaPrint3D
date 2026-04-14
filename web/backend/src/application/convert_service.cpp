#include "application/convert_service.h"

#include "chromaprint3d/pipeline.h"
#include "chromaprint3d/shape_width_analyzer.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace chromaprint3d::backend {

using json = nlohmann::json;
using namespace ChromaPrint3D;

ConvertService::ConvertService(const ServerConfig& cfg, DataRepository& data,
                               SessionStore& sessions, TaskRuntime& tasks)
    : cfg_(cfg), data_(data), sessions_(sessions), tasks_(tasks) {}

ServiceResult ConvertService::SubmitConvertRaster(const std::string& owner,
                                                  const std::vector<uint8_t>& image,
                                                  const std::string& image_name,
                                                  const std::optional<std::string>& params_json) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto accept = tasks_.CanAccept(owner);
    if (!accept.ok) {
        return ServiceResult::Error(accept.status_code, "queue_rejected", accept.message);
    }
    auto valid = ValidateDecodedImage(image, cfg_.max_pixels_per_image);
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

ServiceResult ConvertService::SubmitConvertVector(const std::string& owner,
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

ServiceResult ConvertService::AnalyzeVectorWidth(const std::vector<uint8_t>& svg,
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

ServiceResult ConvertService::SubmitConvertRasterMatchOnly(
    const std::string& owner, const std::vector<uint8_t>& image, const std::string& image_name,
    const std::optional<std::string>& params_json) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto accept = tasks_.CanAccept(owner);
    if (!accept.ok) {
        return ServiceResult::Error(accept.status_code, "queue_rejected", accept.message);
    }
    auto valid = ValidateDecodedImage(image, cfg_.max_pixels_per_image);
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

ServiceResult ConvertService::SubmitConvertVectorMatchOnly(
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

ServiceResult ConvertService::BuildRasterRequest(const json& params,
                                                 const std::vector<uint8_t>& image,
                                                 const std::string& image_name,
                                                 const std::optional<SessionSnapshot>& session,
                                                 ConvertRasterRequest& out) const {
    out.image_buffer = image;
    out.image_name   = image_name;

    auto presets_path = std::filesystem::path(cfg_.data_dir) / "presets";
    if (std::filesystem::is_directory(presets_path)) { out.preset_dir = presets_path.string(); }

    std::string common_vendor, common_material;
    auto selected = ResolveSelectedColorDbs(params, session, data_, out.preloaded_dbs,
                                            common_vendor, common_material);
    if (!selected.ok) return selected;

    if (!common_vendor.empty() && !common_material.empty()) {
        auto profile_keys = BuildProfileChannelKeys(out.preloaded_dbs);
        out.preloaded_model_pack =
            data_.ModelPackRegistry().Select(common_vendor, common_material, profile_keys);
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
        if (params.contains("color_layers")) {
            out.color_layers = params["color_layers"].get<int>();
        }
        if (params.contains("print_mode")) {
            const std::string pm = params["print_mode"].get<std::string>();
            if (pm == "0.08x5" || pm == "0p08x5") {
                out.color_layers    = 5;
                out.layer_height_mm = 0.08f;
            } else if (pm == "0.04x10" || pm == "0p04x10") {
                out.color_layers    = 10;
                out.layer_height_mm = 0.04f;
            }
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

    if (out.color_layers < 1 || out.color_layers > 20) {
        return ServiceResult::Error(400, "invalid_params", "color_layers must be between 1 and 20");
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

ServiceResult ConvertService::BuildVectorRequest(const json& params,
                                                 const std::vector<uint8_t>& svg,
                                                 const std::string& svg_name,
                                                 const std::optional<SessionSnapshot>& session,
                                                 ConvertVectorRequest& out) const {
    out.svg_buffer = svg;
    out.svg_name   = svg_name;

    auto vpresets = std::filesystem::path(cfg_.data_dir) / "presets";
    if (std::filesystem::is_directory(vpresets)) { out.preset_dir = vpresets.string(); }

    std::string vec_common_vendor, vec_common_material;
    auto selected = ResolveSelectedColorDbs(params, session, data_, out.preloaded_dbs,
                                            vec_common_vendor, vec_common_material);
    if (!selected.ok) return selected;

    if (!vec_common_vendor.empty() && !vec_common_material.empty()) {
        auto profile_keys = BuildProfileChannelKeys(out.preloaded_dbs);
        out.preloaded_model_pack =
            data_.ModelPackRegistry().Select(vec_common_vendor, vec_common_material, profile_keys);
    }

    try {
        if (params.contains("target_width_mm")) {
            out.target_width_mm = params["target_width_mm"].get<float>();
        }
        if (params.contains("target_height_mm")) {
            out.target_height_mm = params["target_height_mm"].get<float>();
        }
        if (params.contains("color_layers")) {
            out.color_layers = params["color_layers"].get<int>();
        }
        if (params.contains("print_mode")) {
            const std::string pm = params["print_mode"].get<std::string>();
            if (pm == "0.08x5" || pm == "0p08x5") {
                out.color_layers    = 5;
                out.layer_height_mm = 0.08f;
            } else if (pm == "0.04x10" || pm == "0p04x10") {
                out.color_layers    = 10;
                out.layer_height_mm = 0.04f;
            }
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

    if (out.color_layers < 1 || out.color_layers > 20) {
        return ServiceResult::Error(400, "invalid_params", "color_layers must be between 1 and 20");
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

} // namespace chromaprint3d::backend
