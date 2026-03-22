#pragma once

/// \file recipe_alternatives.h
/// \brief Alternative recipe retrieval for the recipe editor.

#include "color.h"
#include "color_db.h"
#include "model_package.h"
#include "print_profile.h"
#include "recipe_map.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ChromaPrint3D {

struct RecipeCandidate {
    std::vector<uint8_t> recipe;
    Lab predicted_color;
    float delta_e76      = 0.0f;
    float lightness_diff = 0.0f;
    float hue_diff       = 0.0f;
    bool from_model      = false;
};

/// Pre-computed search state for repeated alternative-recipe queries.
/// Build once per (dbs, profile, model) combination and reuse across calls.
/// Copyable via shared_ptr (the cached state is read-only after construction).
///
/// **Lifetime requirement**: the `ColorDB` objects referenced by the `dbs` span
/// passed to Build() must outlive this cache.  Internally the cache holds raw
/// pointers into those objects for fully-compatible DBs (where no per-entry
/// filtering is needed).  Destroying, moving, or reallocating the source
/// container while the cache is alive causes undefined behaviour.
class RecipeSearchCache {
public:
    RecipeSearchCache() = default;

    [[nodiscard]] static RecipeSearchCache
    Build(std::span<const ColorDB> dbs, const PrintProfile& profile, const MatchConfig& match_cfg,
          const ModelPackage* model_package = nullptr, const ModelGateConfig& model_gate = {});

    bool IsValid() const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    friend std::vector<RecipeCandidate> FindAlternativeRecipes(const Lab&, const RecipeSearchCache&,
                                                               int, int);
};

std::vector<RecipeCandidate> FindAlternativeRecipes(
    const Lab& target_color, std::span<const ColorDB> dbs, const PrintProfile& profile,
    const MatchConfig& match_cfg, int max_candidates, int offset = 0,
    const ModelPackage* model_package = nullptr, const ModelGateConfig& model_gate = {});

/// Overload using pre-built cache for repeated queries (avoids re-preparing DBs each call).
std::vector<RecipeCandidate> FindAlternativeRecipes(const Lab& target_color,
                                                    const RecipeSearchCache& cache,
                                                    int max_candidates, int offset = 0);

} // namespace ChromaPrint3D
