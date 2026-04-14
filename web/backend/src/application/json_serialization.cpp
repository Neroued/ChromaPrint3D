#include "application/json_serialization.h"

#include "chromaprint3d/print_profile.h"
#include "chromaprint3d/slicer_preset.h"

#include <cstdio>

namespace chromaprint3d::backend {

using json = nlohmann::json;
using namespace ChromaPrint3D;

const char* ClusterMethodToString(ClusterMethod method) {
    switch (method) {
    case ClusterMethod::Slic:
        return "slic";
    case ClusterMethod::KMeans:
        return "kmeans";
    }
    return "slic";
}

const char* LayerOrderToString(LayerOrder order) {
    return order == LayerOrder::Bottom2Top ? "Bottom2Top" : "Top2Bottom";
}

const char* ConvertStageToString(ConvertStage stage) {
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

json ColorDbInfoToJson(const ColorDB& db, const std::string& material_type,
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

json ColorDbBuildResultToJson(const ColorDB& db) {
    auto out = ColorDbInfoToJson(db);

    json entries = json::array();
    for (const auto& entry : db.entries) {
        Rgb rgb    = entry.lab.ToRgb();
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

json TaskToJson(const TaskSnapshot& task) {
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
        j["generation"] = {
            {"status", GenerationStatusToString(cp->generation.status)},
            {"error", cp->generation.error.empty() ? json(nullptr) : json(cp->generation.error)},
        };
        if (!cp->result.warnings.empty()) { j["warnings"] = cp->result.warnings; }
        if (task.status == RuntimeTaskStatus::Completed) {
            j["elapsed_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  task.completed_at - task.created_at)
                                  .count();
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
                {"has_region_map",
                 !cp->region_map_binary.empty() || cp->raster_region_map.has_value()},
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

} // namespace chromaprint3d::backend
