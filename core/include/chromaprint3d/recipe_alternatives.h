#pragma once

/// \file recipe_alternatives.h
/// \brief Alternative recipe retrieval for the recipe editor.

#include "color.h"
#include "color_db.h"
#include "model_package.h"
#include "print_profile.h"
#include "recipe_map.h"

#include <cstdint>
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

std::vector<RecipeCandidate> FindAlternativeRecipes(
    const Lab& target_color, std::span<const ColorDB> dbs, const PrintProfile& profile,
    const MatchConfig& match_cfg, int max_candidates, int offset = 0,
    const ModelPackage* model_package = nullptr, const ModelGateConfig& model_gate = {});

} // namespace ChromaPrint3D
