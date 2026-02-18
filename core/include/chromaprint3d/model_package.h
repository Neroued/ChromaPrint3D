#pragma once

/// \file model_package.h
/// \brief Trained model package for ML-assisted color matching.

#include "common.h"
#include "color.h"
#include "print_profile.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ChromaPrint3D {

/// A mode-specific slice of a ModelPackage — contains candidate recipes and
/// their predicted Lab colors for one PrintMode.
struct ModelModePackage {
    PrintMode mode;

    float layer_height_mm  = 0.08f;
    int color_layers       = 5;
    LayerOrder layer_order = LayerOrder::Top2Bottom;

    std::string base_channel_key;

    /// Flattened candidate recipes: size() == NumCandidates() * color_layers.
    std::vector<uint8_t> candidate_recipes;
    /// Predicted Lab for each candidate recipe.
    std::vector<Lab> pred_lab;

    size_t NumCandidates() const;
    const uint8_t* RecipeAt(size_t idx) const;
};

/// A collection of pre-computed model predictions for multiple print modes.
struct ModelPackage {
    std::string name;
    std::vector<std::string> channel_keys;

    float default_threshold = 5.0f; ///< DeltaE76 threshold before model query.
    float default_margin    = 0.7f; ///< Model must beat DB by this margin.

    std::vector<ModelModePackage> modes;

    /// Load a model package from a JSON file.
    static ModelPackage LoadFromJson(const std::string& path);

    /// Find the mode-specific data for \p mode, or nullptr if unavailable.
    const ModelModePackage* FindMode(PrintMode mode) const;
};

/// Configuration for the model gate that decides when to prefer model
/// predictions over DB nearest-neighbour results.
struct ModelGateConfig {
    bool enable     = false;
    bool model_only = false; ///< true: bypass DB matching entirely.
    float threshold = 5.0f;
    float margin    = 0.7f;
};

} // namespace ChromaPrint3D
