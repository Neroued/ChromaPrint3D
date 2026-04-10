#pragma once

/// \file vector_recipe_map.h
/// \brief Per-shape recipe mapping for vector images.

#include "common.h"
#include "color.h"
#include "model_package.h"
#include "recipe_map.h"
#include "print_profile.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ChromaPrint3D {

struct VectorProcResult;

/// Per-shape recipe map produced by matching vector shapes to ColorDB.
struct VectorRecipeMap {
    /// A matched shape entry.
    struct ShapeEntry {
        int shape_idx;               ///< Index into VectorProcResult.shapes.
        std::vector<uint8_t> recipe; ///< color_layers entries, each = channel index.
        Lab matched_color;
        bool from_model = false; ///< True when the final selected recipe came from model fallback.
    };

    int color_layers       = 0;
    int num_channels       = 0;
    LayerOrder layer_order = LayerOrder::Top2Bottom;
    std::vector<ShapeEntry> entries;

    /// Match each unique solid color in the vector image to a recipe.
    static VectorRecipeMap Match(const VectorProcResult& result, std::span<const ColorDB> dbs,
                                 const PrintProfile& profile, const MatchConfig& cfg = {},
                                 const ModelPackage* model_package = nullptr,
                                 const ModelGateConfig& model_gate = {},
                                 MatchStats* out_stats             = nullptr);

    /// Replace the recipe for specified entries.
    void ReplaceRecipeForEntries(const std::vector<int>& entry_indices,
                                 const std::vector<uint8_t>& new_recipe,
                                 const Lab& new_mapped_color, bool new_from_model);

    /// Upgrade all entries to a higher color_layers count by padding each
    /// entry's recipe with \p pad_channel at the bottom (near base plate).
    /// No-op if \p new_color_layers <= current color_layers.
    void UpgradeColorLayers(int new_color_layers, uint8_t pad_channel);
};

} // namespace ChromaPrint3D
