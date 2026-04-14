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
#include <string>
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

// ── Recipe pattern matching ─────────────────────────────────────────────────

/// Constraint for a single layer position in a recipe pattern.
struct LayerConstraint {
    bool wildcard = true;                  ///< true → matches any channel.
    std::vector<uint8_t> allowed_channels; ///< Valid when wildcard == false.
};

/// Parsed recipe pattern token (used internally by RecipePattern).
enum class PatternTokenKind : uint8_t {
    Letter,     ///< Matches specific channel(s) — consumes exactly 1 layer.
    SingleWild, ///< '?' — matches any single layer.
    MultiWild,  ///< '*' — matches 0 or more layers.
};

struct PatternToken {
    PatternTokenKind kind = PatternTokenKind::SingleWild;
    std::vector<uint8_t> allowed_channels; ///< Only meaningful for Letter kind.
};

/// A compiled recipe search pattern supporting glob-style matching.
///
/// Syntax (case-insensitive, `-` separators are ignored):
///   - Letter   → matches palette channel(s) whose color name starts with that letter.
///   - `?`      → matches any single layer.
///   - `*`      → matches 0 or more layers.
///   - Input shorter than color_layers (without `*`) → implicit trailing `*`.
struct RecipePattern {
    std::vector<PatternToken> tokens;

    /// Returns true if the pattern has no tokens (empty / not set).
    bool Empty() const { return tokens.empty(); }

    /// Tests whether a recipe matches this pattern.
    /// \param recipe Pointer to the first layer channel index.
    /// \param color_layers Number of layers in the recipe.
    bool Matches(const uint8_t* recipe, int color_layers) const;
    bool Matches(const std::vector<uint8_t>& recipe) const;
};

/// Parse a user-supplied pattern string into a compiled RecipePattern.
/// Returns an empty pattern if the string is empty or contains only separators.
RecipePattern ParseRecipePattern(const std::string& pattern, const std::vector<Channel>& palette,
                                 int color_layers);

// ── RecipeSearchCache ───────────────────────────────────────────────────────

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

    [[nodiscard]] static RecipeSearchCache Build(std::span<const ColorDB* const> db_ptrs,
                                                 const PrintProfile& profile,
                                                 const MatchConfig& match_cfg,
                                                 const ModelPackage* model_package = nullptr,
                                                 const ModelGateConfig& model_gate = {});

    bool IsValid() const;

    /// Access the merged palette used by this cache.
    const std::vector<Channel>& Palette() const;

    /// Number of color layers in the associated profile.
    int ColorLayers() const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    friend std::vector<RecipeCandidate> FindAlternativeRecipes(const Lab&, const RecipeSearchCache&,
                                                               int, int);
    friend std::vector<RecipeCandidate> FindRecipesByPattern(const RecipePattern&, const Lab&,
                                                             const RecipeSearchCache&, int, int);
};

// ── Search functions ────────────────────────────────────────────────────────

std::vector<RecipeCandidate> FindAlternativeRecipes(
    const Lab& target_color, std::span<const ColorDB> dbs, const PrintProfile& profile,
    const MatchConfig& match_cfg, int max_candidates, int offset = 0,
    const ModelPackage* model_package = nullptr, const ModelGateConfig& model_gate = {});

/// Overload using pre-built cache for repeated queries (avoids re-preparing DBs each call).
std::vector<RecipeCandidate> FindAlternativeRecipes(const Lab& target_color,
                                                    const RecipeSearchCache& cache,
                                                    int max_candidates, int offset = 0);

/// Find recipes matching a compiled pattern, ranked by DeltaE76 to target_color.
std::vector<RecipeCandidate> FindRecipesByPattern(const RecipePattern& pattern,
                                                  const Lab& target_color,
                                                  const RecipeSearchCache& cache,
                                                  int max_candidates, int offset = 0);

} // namespace ChromaPrint3D
