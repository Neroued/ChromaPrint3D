#include "detail/candidate_select.h"
#include "detail/match_utils.h"
#include "detail/recipe_convert.h"
#include "chromaprint3d/error.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ChromaPrint3D {
namespace detail {

using detail::BuildChannelKey;
using detail::Dist2;
using detail::NearlyEqual;
using detail::NormalizeChannelKeyString;

namespace {

Lab TargetToLab(const cv::Vec3f& target_color, bool use_lab) {
    if (use_lab) { return Lab(target_color[0], target_color[1], target_color[2]); }
    return Rgb(target_color[0], target_color[1], target_color[2]).ToLab();
}

float ScoreDist2(const Lab& candidate_lab, const cv::Vec3f& target_color, bool use_lab) {
    if (use_lab) {
        return Dist2(candidate_lab, Lab(target_color[0], target_color[1], target_color[2]));
    }
    return Dist2(candidate_lab.ToRgb(), Rgb(target_color[0], target_color[1], target_color[2]));
}

} // namespace

CandidateResult FindBestDbCandidate(const cv::Vec3f& target_color, bool use_lab,
                                    const std::vector<PreparedDB>& prepared_dbs,
                                    const PrintProfile& profile, const MatchConfig& cfg) {
    CandidateResult best;
    const std::size_t k  = static_cast<std::size_t>(std::max(1, cfg.k_candidates));
    const bool use_top_k = cfg.k_candidates > 1;
    const Lab target_lab = TargetToLab(target_color, use_lab);

    for (const PreparedDB& prepared_db : prepared_dbs) {
        const ColorDB* search_db =
            prepared_db.filtered_db ? prepared_db.filtered_db.get() : prepared_db.db;
        if (search_db->entries.empty()) { continue; }

        if (!use_top_k) {
            const Entry& entry = use_lab ? search_db->NearestEntry(target_lab)
                                         : search_db->NearestEntry(Rgb(
                                               target_color[0], target_color[1], target_color[2]));
            std::vector<uint8_t> mapped_recipe;
            if (!ConvertRecipeToProfile(entry, prepared_db, profile, mapped_recipe)) { continue; }

            const float score_d2 = ScoreDist2(entry.lab, target_color, use_lab);
            if (!best.valid || score_d2 < best.score_dist2) {
                best.valid       = true;
                best.mapped_lab  = entry.lab;
                best.recipe      = std::move(mapped_recipe);
                best.score_dist2 = score_d2;
                best.lab_dist2   = Dist2(entry.lab, target_lab);
                best.from_model  = false;
            }
            continue;
        }

        if (use_lab) {
            auto candidates = search_db->NearestEntries(target_lab, k);
            for (const Entry* entry : candidates) {
                if (!entry) { continue; }
                std::vector<uint8_t> mapped_recipe;
                if (!ConvertRecipeToProfile(*entry, prepared_db, profile, mapped_recipe)) {
                    continue;
                }
                const float score_d2 = ScoreDist2(entry->lab, target_color, true);
                if (!best.valid || score_d2 < best.score_dist2) {
                    best.valid       = true;
                    best.mapped_lab  = entry->lab;
                    best.recipe      = std::move(mapped_recipe);
                    best.score_dist2 = score_d2;
                    best.lab_dist2   = Dist2(entry->lab, target_lab);
                    best.from_model  = false;
                }
            }
        } else {
            const Rgb target_rgb(target_color[0], target_color[1], target_color[2]);
            auto candidates = search_db->NearestEntries(target_rgb, k);
            for (const Entry* entry : candidates) {
                if (!entry) { continue; }
                std::vector<uint8_t> mapped_recipe;
                if (!ConvertRecipeToProfile(*entry, prepared_db, profile, mapped_recipe)) {
                    continue;
                }
                const float score_d2 = ScoreDist2(entry->lab, target_color, false);
                if (!best.valid || score_d2 < best.score_dist2) {
                    best.valid       = true;
                    best.mapped_lab  = entry->lab;
                    best.recipe      = std::move(mapped_recipe);
                    best.score_dist2 = score_d2;
                    best.lab_dist2   = Dist2(entry->lab, target_lab);
                    best.from_model  = false;
                }
            }
        }
    }

    return best;
}

std::optional<PreparedModel> PrepareModel(const ModelPackage* model_package,
                                          const ModelGateConfig& model_gate,
                                          const PrintProfile& profile) {
    if (!model_package || (!model_gate.enable && !model_gate.model_only)) { return std::nullopt; }
    const ModelModePackage* mode = model_package->FindMode(profile.mode);
    if (!mode) { return std::nullopt; }
    if (mode->color_layers != profile.color_layers) { return std::nullopt; }
    if (!NearlyEqual(mode->layer_height_mm, profile.layer_height_mm)) { return std::nullopt; }
    if (mode->layer_order != profile.layer_order) { return std::nullopt; }
    if (mode->NumCandidates() == 0) { return std::nullopt; }
    if (model_package->channel_keys.empty()) { return std::nullopt; }

    std::unordered_map<std::string, int> key_to_profile_channel;
    key_to_profile_channel.reserve(profile.palette.size());
    for (std::size_t i = 0; i < profile.palette.size(); ++i) {
        key_to_profile_channel.emplace(BuildChannelKey(profile.palette[i]), static_cast<int>(i));
    }

    std::vector<int> model_to_profile(model_package->channel_keys.size(), -1);
    for (std::size_t i = 0; i < model_package->channel_keys.size(); ++i) {
        const std::string key = NormalizeChannelKeyString(model_package->channel_keys[i]);
        auto it               = key_to_profile_channel.find(key);
        if (it != key_to_profile_channel.end()) { model_to_profile[i] = it->second; }
    }

    PreparedModel prepared;
    prepared.threshold    = std::max(0.0f, model_gate.threshold);
    prepared.margin       = std::max(0.0f, model_gate.margin);
    prepared.color_layers = profile.color_layers;
    prepared.layer_order  = profile.layer_order;
    prepared.pred_lab.reserve(mode->NumCandidates());
    prepared.mapped_recipes.reserve(mode->candidate_recipes.size());

    for (size_t i = 0; i < mode->NumCandidates(); ++i) {
        const uint8_t* src_recipe = mode->RecipeAt(i);
        if (!src_recipe) { continue; }
        const std::size_t write_base = prepared.mapped_recipes.size();
        bool valid_recipe            = true;
        for (int l = 0; l < prepared.color_layers; ++l) {
            const std::size_t src_ch = static_cast<std::size_t>(src_recipe[l]);
            if (src_ch >= model_to_profile.size()) {
                valid_recipe = false;
                break;
            }
            const int mapped_ch = model_to_profile[src_ch];
            if (mapped_ch < 0 || static_cast<std::size_t>(mapped_ch) >= profile.NumChannels()) {
                valid_recipe = false;
                break;
            }
            prepared.mapped_recipes.push_back(static_cast<uint8_t>(mapped_ch));
        }
        if (!valid_recipe) {
            prepared.mapped_recipes.resize(write_base);
            continue;
        }
        prepared.pred_lab.push_back(mode->pred_lab[i]);
    }

    if (prepared.pred_lab.empty()) { return std::nullopt; }
    if (prepared.mapped_recipes.size() !=
        prepared.pred_lab.size() * static_cast<std::size_t>(prepared.color_layers)) {
        throw InputError("PreparedModel recipe/lab size mismatch");
    }

    prepared.kd_indices.resize(prepared.pred_lab.size());
    for (std::size_t i = 0; i < prepared.kd_indices.size(); ++i) { prepared.kd_indices[i] = i; }

    const auto points  = std::span<const Lab>(prepared.pred_lab);
    const auto indices = std::span<const std::size_t>(prepared.kd_indices);
    prepared.lab_tree.Reset(points, indices, ModelLabProj{});
    prepared.rgb_tree.Reset(points, indices, ModelRgbProj{});
    return prepared;
}

CandidateResult FindBestModelCandidate(const cv::Vec3f& target_color, bool use_lab,
                                       const PreparedModel& model) {
    CandidateResult best;
    if (model.NumCandidates() == 0) { return best; }

    const Lab target_lab = TargetToLab(target_color, use_lab);
    std::size_t idx      = 0;
    float score_d2       = 0.0f;
    if (use_lab) {
        const auto neighbor = model.lab_tree.Nearest(target_lab);
        idx                 = static_cast<std::size_t>(neighbor.index);
        score_d2            = neighbor.dist2;
    } else {
        const Rgb target_rgb(target_color[0], target_color[1], target_color[2]);
        const auto neighbor = model.rgb_tree.Nearest(target_rgb);
        idx                 = static_cast<std::size_t>(neighbor.index);
        score_d2            = neighbor.dist2;
    }

    const uint8_t* recipe = model.RecipeAt(idx);
    if (!recipe) { return best; }

    best.valid      = true;
    best.mapped_lab = model.pred_lab[idx];
    best.recipe.assign(recipe, recipe + model.color_layers);
    best.score_dist2 = score_d2;
    best.lab_dist2   = Dist2(best.mapped_lab, target_lab);
    best.from_model  = true;
    return best;
}

CandidateDecision SelectCandidate(const cv::Vec3f& target_color, bool use_lab,
                                  const std::vector<PreparedDB>& prepared_dbs,
                                  const PrintProfile& profile, const MatchConfig& cfg,
                                  const PreparedModel* prepared_model, bool model_only) {
    CandidateDecision decision;
    if (model_only) {
        if (!prepared_model) {
            throw ConfigError(
                "Model-only matching requested but model package is unavailable");
        }
        decision.model_queried     = true;
        CandidateResult model_best = FindBestModelCandidate(target_color, use_lab, *prepared_model);
        if (!model_best.valid) {
            throw MatchError("No valid model candidate in model-only mode");
        }
        decision.model_de = std::sqrt(std::max(0.0f, model_best.lab_dist2));
        decision.db_de    = 0.0f;
        decision.selected = std::move(model_best);
        return decision;
    }

    CandidateResult db_best =
        FindBestDbCandidate(target_color, use_lab, prepared_dbs, profile, cfg);

    if (!db_best.valid) {
        if (prepared_model) {
            decision.model_queried = true;
            CandidateResult model_best =
                FindBestModelCandidate(target_color, use_lab, *prepared_model);
            if (model_best.valid) {
                decision.model_de = std::sqrt(std::max(0.0f, model_best.lab_dist2));
                decision.db_de    = 0.0f;
                decision.selected = std::move(model_best);
                return decision;
            }
        }
        throw MatchError(
            "No valid candidate: ColorDB has no recipe for the selected channels, "
            "and model fallback is unavailable");
    }

    decision.db_de    = std::sqrt(std::max(0.0f, db_best.lab_dist2));
    decision.selected = db_best;

    if (!prepared_model) { return decision; }
    if (decision.db_de <= prepared_model->threshold) { return decision; }

    decision.model_queried     = true;
    CandidateResult model_best = FindBestModelCandidate(target_color, use_lab, *prepared_model);
    if (!model_best.valid) { return decision; }
    decision.model_de = std::sqrt(std::max(0.0f, model_best.lab_dist2));

    if (decision.model_de + prepared_model->margin < decision.db_de) {
        decision.selected = std::move(model_best);
    }
    return decision;
}

void WriteRecipe(RecipeMap& result, std::size_t pixel_idx, const std::vector<uint8_t>& recipe) {
    if (result.color_layers <= 0) { return; }
    const std::size_t color_layers = static_cast<std::size_t>(result.color_layers);
    if (recipe.size() != color_layers) {
        throw InputError("Matched recipe layers do not match PrintProfile");
    }
    const std::size_t offset = pixel_idx * color_layers;
    if (offset + color_layers > result.recipes.size()) {
        throw InputError("RecipeMap recipes size mismatch");
    }
    std::copy(recipe.begin(), recipe.end(),
              result.recipes.begin() + static_cast<std::ptrdiff_t>(offset));
}

void WriteSourceMask(RecipeMap& result, std::size_t pixel_idx, bool from_model) {
    if (pixel_idx >= result.source_mask.size()) {
        throw InputError("RecipeMap source_mask size mismatch");
    }
    result.source_mask[pixel_idx] = from_model ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
}

} // namespace detail
} // namespace ChromaPrint3D
