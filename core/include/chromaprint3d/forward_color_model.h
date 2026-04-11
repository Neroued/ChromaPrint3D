#pragma once

/// \file forward_color_model.h
/// \brief Beer-Lambert forward color model for predicting recipe colors.

#include "color.h"
#include "common.h"

#include <array>
#include <span>
#include <string>
#include <vector>

namespace ChromaPrint3D {

/// Declares which vendor + material_type a forward model is calibrated for.
/// Mirrors ModelPackageScope; kept separate for independent evolution.
struct ForwardModelScope {
    std::string vendor;        ///< e.g. "BambuLab"
    std::string material_type; ///< e.g. "PLA"
};

/// Per-prediction context (not stored in the model itself).
struct PredictionConfig {
    float layer_height_mm  = 0.08f;
    int base_channel_idx   = 0; ///< Stage-space channel index for the base/substrate.
    LayerOrder layer_order = LayerOrder::Top2Bottom;
    int substrate_idx      = 0;
};

/// A loaded forward color model (currently Stage B Beer-Lambert parameters).
///
/// The model predicts the color produced by a given recipe of stacked
/// translucent layers using the Beer-Lambert light absorption law.
struct ForwardColorModel {
    std::string name;
    std::string param_type; ///< "stage_B" | future types
    ForwardModelScope scope;
    std::vector<std::string> channel_keys; ///< Normalized "color|material" keys.

    int num_channels = 0;
    std::vector<std::string> channel_names;                 ///< Stage channel index -> color name.
    std::vector<std::array<float, 3>> E;                    ///< Equilibrium color [Nc][3].
    std::vector<std::array<float, 3>> k;                    ///< Absorption coefficient [Nc][3].
    std::vector<float> gamma;                               ///< Height-scale exponent [Nc].
    std::vector<std::vector<std::array<float, 3>>> delta_k; ///< Neighbor correction [Nc][Nc][3].
    std::vector<std::array<float, 3>> C0;                   ///< Substrate boundary color [Ns][3].
    std::vector<int> substrate_base_channel_idx;            ///< Per substrate.

    float micro_layer_height_mm = 0.04f;
    float height_ref_mm         = 0.04f;
    bool enable_height_scale    = true;

    /// Load a forward model from a JSON file.
    /// Throws std::runtime_error on parse failure.
    static ForwardColorModel LoadFromJson(const std::string& path);

    /// Whether this model's scope matches the given vendor + material_type.
    bool MatchesScope(const std::string& vendor, const std::string& material_type) const;

    /// Find a stage channel index by color name (case-insensitive).
    /// Returns -1 if not found.
    int MapChannelByName(const std::string& color_name) const;

    /// Resolve substrate index from the base channel's stage index.
    /// Falls back to 0 if no match.
    int ResolveSubstrateIdx(int base_channel_stage_idx) const;

    /// Predict the Lab color for a single recipe.
    /// \p recipe contains stage-space channel indices, length == color layers.
    Lab PredictLab(std::span<const int> recipe, const PredictionConfig& cfg) const;
};

/// Manages multiple ForwardColorModel instances loaded from a directory.
class ForwardModelRegistry {
public:
    /// Discover and load all .json files in \p dir.
    /// Individual files that fail to load are logged and skipped.
    void LoadFromDirectory(const std::string& dir);

    /// Select the best-matching model for a request.
    /// Logic mirrors ModelPackageRegistry::Select exactly:
    ///   1. scope.vendor + scope.material_type must match exactly.
    ///   2. All model channel_keys must be present in profile_channel_keys.
    ///   3. Among matches, the model with the most channel_keys wins.
    /// Returns nullptr if no model matches.
    const ForwardColorModel* Select(const std::string& vendor, const std::string& material_type,
                                    const std::vector<std::string>& profile_channel_keys) const;

    bool Empty() const { return models_.empty(); }

private:
    std::vector<ForwardColorModel> models_;
};

} // namespace ChromaPrint3D
