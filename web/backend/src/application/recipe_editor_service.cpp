#include "application/recipe_editor_service.h"

#include "application/colordb_resolution.h"

#include "chromaprint3d/forward_color_model.h"
#include "chromaprint3d/print_profile.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <vector>

namespace chromaprint3d::backend {

using json = nlohmann::json;

RecipeEditorService::RecipeEditorService(DataRepository& data, TaskRuntime& tasks)
    : data_(data), tasks_(tasks) {}

ServiceResult RecipeEditorService::RecipeEditorSummary(const std::string& owner,
                                                       const std::string& task_id) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto summary = tasks_.GetRecipeEditorSummary(owner, task_id);
    if (!summary) return ServiceResult::Error(404, "not_found", "Summary not available");
    return ServiceResult::Success(200, std::move(*summary));
}

ServiceResult RecipeEditorService::RecipeEditorAlternatives(const std::string& owner,
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

    std::string recipe_pattern = body.value("recipe_pattern", std::string{});

    const ChromaPrint3D::ModelPackage* model_pack = nullptr;
    auto db_names_opt                             = tasks_.GetTaskColorDbNames(owner, task_id);
    if (db_names_opt) {
        std::string mp_vendor, mp_material;
        ResolveTaskVendorMaterial(*db_names_opt, data_, mp_vendor, mp_material);
        if (!mp_vendor.empty() && !mp_material.empty()) {
            auto profile_opt = tasks_.GetTaskPrintProfile(owner, task_id);
            if (profile_opt) {
                std::vector<std::string> keys;
                keys.reserve(profile_opt->palette.size());
                for (const auto& ch : profile_opt->palette) {
                    keys.push_back(NormalizeChannelKey(ch));
                }
                model_pack = data_.ModelPackRegistry().Select(mp_vendor, mp_material, keys);
            }
        }
    }

    auto result = tasks_.QueryRecipeAlternatives(owner, task_id, target, max_candidates, offset,
                                                 model_pack, recipe_pattern);
    if (!result) return ServiceResult::Error(404, "not_found", "Alternatives not available");
    return ServiceResult::Success(200, {{"candidates", std::move(*result)}});
}

ServiceResult RecipeEditorService::RecipeEditorReplace(const std::string& owner,
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

ServiceResult RecipeEditorService::RecipeEditorGenerate(const std::string& owner,
                                                        const std::string& task_id) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");

    auto submit = tasks_.SubmitGenerateModel(owner, task_id);
    if (!submit.ok) {
        return ServiceResult::Error(submit.status_code, "generate_failed", submit.message);
    }
    return ServiceResult::Success(202, {{"task_id", submit.task_id}});
}

ServiceResult RecipeEditorService::RecipeEditorPredict(const std::string& owner,
                                                       const std::string& task_id,
                                                       const std::string& body_json) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");

    json body;
    try {
        body = json::parse(body_json);
    } catch (...) { return ServiceResult::Error(400, "invalid_json", "Cannot parse request body"); }
    if (!body.contains("recipe") || !body["recipe"].is_array()) {
        return ServiceResult::Error(400, "invalid_params", "Missing recipe array");
    }

    std::vector<int> palette_recipe;
    for (const auto& v : body["recipe"]) {
        if (!v.is_number_integer()) {
            return ServiceResult::Error(400, "invalid_params", "Recipe must be array of integers");
        }
        palette_recipe.push_back(v.get<int>());
    }
    if (palette_recipe.empty()) {
        return ServiceResult::Error(400, "invalid_params", "Recipe cannot be empty");
    }

    auto profile_opt = tasks_.GetTaskPrintProfile(owner, task_id);
    if (!profile_opt) {
        return ServiceResult::Error(404, "not_found", "Task not found or not in recipe-edit mode");
    }
    const auto& profile = *profile_opt;

    std::string common_vendor, common_material;
    auto db_names_opt = tasks_.GetTaskColorDbNames(owner, task_id);
    if (db_names_opt) {
        ResolveTaskVendorMaterial(*db_names_opt, data_, common_vendor, common_material);
    }

    std::vector<std::string> profile_channel_keys;
    for (const auto& ch : profile.palette) {
        profile_channel_keys.push_back(NormalizeChannelKey(ch));
    }

    const auto* model =
        data_.ForwardModels().Select(common_vendor, common_material, profile_channel_keys);
    if (!model) {
        return ServiceResult::Error(501, "no_forward_model",
                                    "No forward model available for this task's material scope");
    }

    std::vector<int> stage_recipe;
    stage_recipe.reserve(palette_recipe.size());
    for (int pidx : palette_recipe) {
        if (pidx < 0 || pidx >= static_cast<int>(profile.palette.size())) {
            return ServiceResult::Error(400, "invalid_params",
                                        "Recipe channel index out of palette range");
        }
        int stage_idx = model->MapChannelByName(profile.palette[pidx].color);
        if (stage_idx < 0) {
            return ServiceResult::Error(400, "channel_mapping_failed",
                                        "Cannot map palette channel '" +
                                            profile.palette[pidx].color +
                                            "' to forward model stage channel");
        }
        stage_recipe.push_back(stage_idx);
    }

    int base_stage_idx = 0;
    if (profile.base_channel_idx >= 0 &&
        profile.base_channel_idx < static_cast<int>(profile.palette.size())) {
        int mapped = model->MapChannelByName(profile.palette[profile.base_channel_idx].color);
        if (mapped >= 0) base_stage_idx = mapped;
    }

    ChromaPrint3D::PredictionConfig cfg;
    cfg.layer_height_mm  = profile.layer_height_mm;
    cfg.base_channel_idx = base_stage_idx;
    cfg.layer_order      = profile.layer_order;
    cfg.substrate_idx    = model->ResolveSubstrateIdx(base_stage_idx);

    ChromaPrint3D::Lab predicted = model->PredictLab(stage_recipe, cfg);

    uint8_t r8, g8, b8;
    predicted.ToRgb().ToRgb255(r8, g8, b8);
    char hex[8];
    std::snprintf(hex, sizeof(hex), "#%02X%02X%02X", r8, g8, b8);

    return ServiceResult::Success(
        200, {{"predicted_lab", {{"L", predicted.l()}, {"a", predicted.a()}, {"b", predicted.b()}}},
              {"hex", std::string(hex)},
              {"from_model", true}});
}

} // namespace chromaprint3d::backend
