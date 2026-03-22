#include "chromaprint3d/recipe_alternatives.h"
#include "chromaprint3d/kdtree.h"
#include "detail/candidate_select.h"
#include "detail/recipe_convert.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <unordered_set>

namespace ChromaPrint3D {
namespace {

std::string RecipeToKey(const std::vector<uint8_t>& recipe) {
    std::string key;
    key.reserve(recipe.size() * 4);
    for (std::size_t i = 0; i < recipe.size(); ++i) {
        if (i > 0) { key += '-'; }
        key += std::to_string(recipe[i]);
    }
    return key;
}

float ComputeHueDiff(const Lab& a, const Lab& b) {
    const float ha      = std::atan2(a.b(), a.a());
    const float hb      = std::atan2(b.b(), b.a());
    constexpr float kPi = std::numbers::pi_v<float>;
    float diff          = std::abs(ha - hb);
    if (diff > kPi) { diff = 2.0f * kPi - diff; }
    return diff * (180.0f / kPi);
}

struct RawCandidate {
    std::vector<uint8_t> recipe;
    Lab predicted_color;
    bool from_model = false;
};

std::vector<RecipeCandidate> CollectAndRank(const Lab& target_color, std::vector<RawCandidate>& raw,
                                            int max_candidates, int offset) {
    std::vector<RecipeCandidate> candidates;
    candidates.reserve(raw.size());

    for (auto& r : raw) {
        RecipeCandidate c;
        c.recipe          = std::move(r.recipe);
        c.predicted_color = r.predicted_color;
        c.delta_e76       = Lab::DeltaE76(target_color, r.predicted_color);
        c.lightness_diff  = std::abs(target_color.l() - r.predicted_color.l());
        c.hue_diff        = ComputeHueDiff(target_color, r.predicted_color);
        c.from_model      = r.from_model;
        candidates.push_back(std::move(c));
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const RecipeCandidate& a, const RecipeCandidate& b) {
                  return a.delta_e76 < b.delta_e76;
              });

    const std::size_t start = static_cast<std::size_t>(std::max(0, offset));
    if (start >= candidates.size()) { return {}; }

    const std::size_t end =
        std::min(candidates.size(), start + static_cast<std::size_t>(max_candidates));
    return {std::make_move_iterator(candidates.begin() + static_cast<std::ptrdiff_t>(start)),
            std::make_move_iterator(candidates.begin() + static_cast<std::ptrdiff_t>(end))};
}

void SearchDBs(const Lab& target_color, const std::vector<detail::PreparedDB>& prepared_dbs,
               const PrintProfile& profile, bool use_lab, int search_k,
               std::vector<RawCandidate>& raw, std::unordered_set<std::string>& seen) {
    for (const auto& pdb : prepared_dbs) {
        const ColorDB* search_db = pdb.filtered_db ? pdb.filtered_db.get() : pdb.db;
        if (!search_db || search_db->entries.empty()) { continue; }

        auto entries =
            use_lab ? search_db->NearestEntries(target_color, static_cast<std::size_t>(search_k))
                    : search_db->NearestEntries(target_color.ToRgb(),
                                                static_cast<std::size_t>(search_k));

        for (const Entry* entry : entries) {
            if (!entry) { continue; }
            std::vector<uint8_t> mapped_recipe;
            if (!detail::ConvertRecipeToProfile(*entry, pdb, profile, mapped_recipe)) { continue; }

            std::string key = RecipeToKey(mapped_recipe);
            if (seen.count(key)) { continue; }
            seen.insert(std::move(key));

            raw.push_back({std::move(mapped_recipe), entry->lab, false});
        }
    }
}

void SearchModel(const Lab& target_color, const detail::PreparedModel& model, bool use_lab,
                 int search_k, std::vector<RawCandidate>& raw,
                 std::unordered_set<std::string>& seen) {
    const std::size_t model_k = static_cast<std::size_t>(search_k);
    const std::size_t n       = std::min(model_k, model.NumCandidates());

    using NeighborT = kdt::Neighbor<std::size_t, float>;
    std::vector<NeighborT> neighbors;
    if (use_lab) {
        model.lab_tree.KNearest(target_color, n, neighbors);
    } else {
        model.rgb_tree.KNearest(target_color.ToRgb(), n, neighbors);
    }

    for (const auto& neighbor : neighbors) {
        const std::size_t idx     = static_cast<std::size_t>(neighbor.index);
        const uint8_t* recipe_ptr = model.RecipeAt(idx);
        if (!recipe_ptr) { continue; }

        std::vector<uint8_t> recipe(recipe_ptr, recipe_ptr + model.color_layers);
        std::string key = RecipeToKey(recipe);
        if (seen.count(key)) { continue; }
        seen.insert(std::move(key));

        raw.push_back({std::move(recipe), model.pred_lab[idx], true});
    }
}

} // namespace

// ── RecipeSearchCache ────────────────────────────────────────────────────────

struct RecipeSearchCache::Impl {
    std::vector<detail::PreparedDB> prepared_dbs;
    std::optional<detail::PreparedModel> prepared_model;
    PrintProfile profile;
    bool use_lab = true;
};

RecipeSearchCache RecipeSearchCache::Build(std::span<const ColorDB> dbs,
                                           const PrintProfile& profile,
                                           const MatchConfig& match_cfg,
                                           const ModelPackage* model_package,
                                           const ModelGateConfig& model_gate) {
    auto impl            = std::make_shared<Impl>();
    impl->prepared_dbs   = detail::PrepareDBs(dbs, profile);
    impl->prepared_model = detail::PrepareModel(model_package, model_gate, profile);
    impl->profile        = profile;
    impl->use_lab        = (match_cfg.color_space == ColorSpace::Lab);

    RecipeSearchCache cache;
    cache.impl_ = std::move(impl);
    return cache;
}

bool RecipeSearchCache::IsValid() const { return impl_ && !impl_->prepared_dbs.empty(); }

// ── FindAlternativeRecipes (uncached) ────────────────────────────────────────

std::vector<RecipeCandidate>
FindAlternativeRecipes(const Lab& target_color, std::span<const ColorDB> dbs,
                       const PrintProfile& profile, const MatchConfig& match_cfg,
                       int max_candidates, int offset, const ModelPackage* model_package,
                       const ModelGateConfig& model_gate) {

    const int search_k = std::max(50, (max_candidates + offset) * 3);
    const bool use_lab = (match_cfg.color_space == ColorSpace::Lab);

    auto prepared_dbs = detail::PrepareDBs(dbs, profile);

    std::vector<RawCandidate> raw;
    std::unordered_set<std::string> seen;
    raw.reserve(static_cast<std::size_t>(search_k) * 2);

    SearchDBs(target_color, prepared_dbs, profile, use_lab, search_k, raw, seen);

    auto prepared_model = detail::PrepareModel(model_package, model_gate, profile);
    if (prepared_model) {
        SearchModel(target_color, *prepared_model, use_lab, search_k, raw, seen);
    }

    return CollectAndRank(target_color, raw, max_candidates, offset);
}

// ── FindAlternativeRecipes (cached) ──────────────────────────────────────────

std::vector<RecipeCandidate> FindAlternativeRecipes(const Lab& target_color,
                                                    const RecipeSearchCache& cache,
                                                    int max_candidates, int offset) {
    if (!cache.impl_) { return {}; }
    const auto& impl   = *cache.impl_;
    const int search_k = std::max(50, (max_candidates + offset) * 3);

    std::vector<RawCandidate> raw;
    std::unordered_set<std::string> seen;
    raw.reserve(static_cast<std::size_t>(search_k) * 2);

    SearchDBs(target_color, impl.prepared_dbs, impl.profile, impl.use_lab, search_k, raw, seen);

    if (impl.prepared_model) {
        SearchModel(target_color, *impl.prepared_model, impl.use_lab, search_k, raw, seen);
    }

    return CollectAndRank(target_color, raw, max_candidates, offset);
}

} // namespace ChromaPrint3D
