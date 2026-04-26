#include "detail/candidate_select.h"
#include "detail/match_target.h"
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
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace ChromaPrint3D {
namespace detail {

using detail::AsLab;
using detail::BuildChannelKey;
using detail::Dist2;
using detail::NearlyEqual;
using detail::NormalizeChannelKeyString;
using detail::ScoreDist2;

template <typename T>
CandidateResult FindBestDbCandidate(const T& target_color,
                                    const std::vector<PreparedDB>& prepared_dbs,
                                    const PrintProfile& profile, const MatchConfig& cfg) {
    static_assert(std::is_same_v<T, Lab> || std::is_same_v<T, Rgb>,
                  "FindBestDbCandidate supports only Lab or Rgb target types");

    CandidateResult best;
    const std::size_t k  = static_cast<std::size_t>(std::max(1, cfg.k_candidates));
    const bool use_top_k = cfg.k_candidates > 1;
    const Lab target_lab = AsLab(target_color);

    for (const PreparedDB& prepared_db : prepared_dbs) {
        const ColorDB* search_db =
            prepared_db.filtered_db ? prepared_db.filtered_db.get() : prepared_db.db;
        if (search_db->entries.empty()) { continue; }

        if (!use_top_k) {
            const Entry& entry = search_db->NearestEntry(target_color);
            std::vector<uint8_t> mapped_recipe;
            if (!ConvertRecipeToProfile(entry, prepared_db, profile, mapped_recipe)) { continue; }

            const float score_d2 = ScoreDist2(entry.lab, target_color);
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

        auto candidates = search_db->NearestEntries(target_color, k);
        for (const Entry* entry : candidates) {
            if (!entry) { continue; }
            std::vector<uint8_t> mapped_recipe;
            if (!ConvertRecipeToProfile(*entry, prepared_db, profile, mapped_recipe)) { continue; }
            const float score_d2 = ScoreDist2(entry->lab, target_color);
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

    return best;
}

std::optional<PreparedModel> PrepareModel(const ModelPackage* model_package,
                                          const ModelGateConfig& model_gate,
                                          const PrintProfile& profile) {
    if (!model_package || (!model_gate.enable && !model_gate.model_only)) { return std::nullopt; }
    const ModelLayerPackage* mode = model_package->FindByColorLayers(profile.color_layers);
    if (!mode) { return std::nullopt; }
    if (mode->NumCandidates() == 0) { return std::nullopt; }
    if (model_package->channel_keys.empty()) { return std::nullopt; }

    PreparedModel prepared;
    prepared.threshold    = std::max(0.0f, model_gate.threshold);
    prepared.margin       = std::max(0.0f, model_gate.margin);
    prepared.color_layers = profile.color_layers;
    prepared.layer_order  = profile.layer_order;

    if (!NearlyEqual(mode->layer_height_mm, profile.layer_height_mm)) {
        spdlog::warn("ModelPackage layer_height_mm mismatch: pack={}, profile={}; "
                     "prediction accuracy may be reduced",
                     mode->layer_height_mm, profile.layer_height_mm);
        prepared.warnings.emplace_back("model_layer_height_mismatch");
    }
    if (mode->layer_order != profile.layer_order) {
        spdlog::warn("ModelPackage layer_order mismatch: pack={}, profile={}; "
                     "prediction accuracy may be reduced",
                     ToLayerOrderString(mode->layer_order),
                     ToLayerOrderString(profile.layer_order));
        prepared.warnings.emplace_back("model_layer_order_mismatch");
    }

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

    // Identity shortcut: if mapping is 1:1 positional and all candidates are
    // valid, reference the original data via spans instead of copying.
    bool is_identity = true;
    for (std::size_t i = 0; i < model_to_profile.size(); ++i) {
        if (model_to_profile[i] != static_cast<int>(i)) {
            is_identity = false;
            break;
        }
    }
    if (is_identity && model_package->channel_keys.size() <= profile.palette.size()) {
        bool all_valid = true;
        for (size_t i = 0; i < mode->NumCandidates(); ++i) {
            const uint8_t* src = mode->RecipeAt(i);
            if (!src) {
                all_valid = false;
                break;
            }
            for (int l = 0; l < prepared.color_layers; ++l) {
                const auto ch = static_cast<std::size_t>(src[l]);
                if (ch >= model_to_profile.size() || model_to_profile[ch] < 0 ||
                    static_cast<std::size_t>(model_to_profile[ch]) >= profile.NumChannels()) {
                    all_valid = false;
                    break;
                }
            }
            if (!all_valid) { break; }
        }
        if (all_valid) {
            prepared.owns_data     = false;
            prepared.pred_lab_view = std::span<const Lab>(mode->pred_lab);
            prepared.recipes_view  = std::span<const uint8_t>(mode->candidate_recipes);
            goto build_trees;
        }
    }

    {
        prepared.owns_data = true;
        prepared.pred_lab_owned.reserve(mode->NumCandidates());
        prepared.recipes_owned.reserve(mode->candidate_recipes.size());

        for (size_t i = 0; i < mode->NumCandidates(); ++i) {
            const uint8_t* src_recipe = mode->RecipeAt(i);
            if (!src_recipe) { continue; }
            const std::size_t write_base = prepared.recipes_owned.size();
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
                prepared.recipes_owned.push_back(static_cast<uint8_t>(mapped_ch));
            }
            if (!valid_recipe) {
                prepared.recipes_owned.resize(write_base);
                continue;
            }
            prepared.pred_lab_owned.push_back(mode->pred_lab[i]);
        }

        if (prepared.pred_lab_owned.empty()) { return std::nullopt; }
        if (prepared.recipes_owned.size() !=
            prepared.pred_lab_owned.size() * static_cast<std::size_t>(prepared.color_layers)) {
            throw InputError("PreparedModel recipe/lab size mismatch");
        }

        prepared.pred_lab_view = std::span<const Lab>(prepared.pred_lab_owned);
        prepared.recipes_view  = std::span<const uint8_t>(prepared.recipes_owned);
    }

build_trees:
    prepared.kd_indices.resize(prepared.pred_lab_view.size());
    for (std::size_t i = 0; i < prepared.kd_indices.size(); ++i) { prepared.kd_indices[i] = i; }

    const auto points  = prepared.pred_lab_view;
    const auto indices = std::span<const std::size_t>(prepared.kd_indices);
    prepared.lab_tree.Reset(points, indices, ModelLabProj{});
    prepared.rgb_tree.Reset(points, indices, ModelRgbProj{});
    return prepared;
}

template <typename T>
CandidateResult FindBestModelCandidate(const T& target_color, const PreparedModel& model) {
    static_assert(std::is_same_v<T, Lab> || std::is_same_v<T, Rgb>,
                  "FindBestModelCandidate supports only Lab or Rgb target types");

    CandidateResult best;
    if (model.NumCandidates() == 0) { return best; }

    const Lab target_lab = AsLab(target_color);
    std::size_t idx      = 0;
    float score_d2       = 0.0f;
    if constexpr (std::is_same_v<T, Lab>) {
        const auto neighbor = model.lab_tree.Nearest(target_color);
        idx                 = static_cast<std::size_t>(neighbor.index);
        score_d2            = neighbor.dist2;
    } else {
        const auto neighbor = model.rgb_tree.Nearest(target_color);
        idx                 = static_cast<std::size_t>(neighbor.index);
        score_d2            = neighbor.dist2;
    }

    const uint8_t* recipe = model.RecipeAt(idx);
    if (!recipe) { return best; }

    best.valid      = true;
    best.mapped_lab = model.pred_lab_view[idx];
    best.recipe.assign(recipe, recipe + model.color_layers);
    best.score_dist2 = score_d2;
    best.lab_dist2   = Dist2(best.mapped_lab, target_lab);
    best.from_model  = true;
    return best;
}

template <typename T>
CandidateDecision SelectCandidate(const T& target_color,
                                  const std::vector<PreparedDB>& prepared_dbs,
                                  const PrintProfile& profile, const MatchConfig& cfg,
                                  const PreparedModel* prepared_model, bool model_only) {
    static_assert(std::is_same_v<T, Lab> || std::is_same_v<T, Rgb>,
                  "SelectCandidate supports only Lab or Rgb target types");

    CandidateDecision decision;
    if (model_only) {
        if (!prepared_model) {
            throw ConfigError("Model-only matching requested but model package is unavailable");
        }
        decision.model_queried     = true;
        CandidateResult model_best = FindBestModelCandidate<T>(target_color, *prepared_model);
        if (!model_best.valid) { throw MatchError("No valid model candidate in model-only mode"); }
        decision.model_de = std::sqrt(std::max(0.0f, model_best.lab_dist2));
        decision.db_de    = 0.0f;
        decision.selected = std::move(model_best);
        return decision;
    }

    CandidateResult db_best = FindBestDbCandidate<T>(target_color, prepared_dbs, profile, cfg);

    if (!db_best.valid) {
        if (prepared_model) {
            decision.model_queried     = true;
            CandidateResult model_best = FindBestModelCandidate<T>(target_color, *prepared_model);
            if (model_best.valid) {
                decision.model_de = std::sqrt(std::max(0.0f, model_best.lab_dist2));
                decision.db_de    = 0.0f;
                decision.selected = std::move(model_best);
                return decision;
            }
        }
        throw MatchError("No valid candidate: ColorDB has no recipe for the selected channels, "
                         "and model fallback is unavailable");
    }

    decision.db_de    = std::sqrt(std::max(0.0f, db_best.lab_dist2));
    decision.selected = db_best;

    if (!prepared_model) { return decision; }
    if (decision.db_de <= prepared_model->threshold) { return decision; }

    decision.model_queried     = true;
    CandidateResult model_best = FindBestModelCandidate<T>(target_color, *prepared_model);
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

// ── Explicit instantiations ─────────────────────────────────────────────────

template CandidateResult FindBestDbCandidate<Lab>(const Lab&, const std::vector<PreparedDB>&,
                                                  const PrintProfile&, const MatchConfig&);
template CandidateResult FindBestDbCandidate<Rgb>(const Rgb&, const std::vector<PreparedDB>&,
                                                  const PrintProfile&, const MatchConfig&);

template CandidateResult FindBestModelCandidate<Lab>(const Lab&, const PreparedModel&);
template CandidateResult FindBestModelCandidate<Rgb>(const Rgb&, const PreparedModel&);

template CandidateDecision SelectCandidate<Lab>(const Lab&, const std::vector<PreparedDB>&,
                                                const PrintProfile&, const MatchConfig&,
                                                const PreparedModel*, bool);
template CandidateDecision SelectCandidate<Rgb>(const Rgb&, const std::vector<PreparedDB>&,
                                                const PrintProfile&, const MatchConfig&,
                                                const PreparedModel*, bool);

} // namespace detail
} // namespace ChromaPrint3D
