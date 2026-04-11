#pragma once

/// \file model_package.h
/// \brief Trained model package for ML-assisted color matching (v2 schema).

#include "common.h"
#include "color.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ChromaPrint3D {

/// Declares which vendor + material_type a ModelPackage is calibrated for.
struct ModelPackageScope {
    std::string vendor;        ///< e.g. "BambuLab"
    std::string material_type; ///< e.g. "PLA"
};

/// A layer-config-specific slice of a ModelPackage — contains candidate recipes
/// and their predicted Lab colors for one (color_layers, layer_height_mm) combination.
struct ModelLayerPackage {
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

/// A collection of pre-computed model predictions for multiple layer configurations.
struct ModelPackage {
    static constexpr int kSchemaVersion = 2;

    std::string name;
    ModelPackageScope scope;
    std::vector<std::string> channel_keys;

    float default_threshold = 5.0f; ///< DeltaE76 threshold before model query.
    float default_margin    = 0.7f; ///< Model must beat DB by this margin.

    std::vector<ModelLayerPackage> layer_packages;

    /// Load a v2 model package from a MessagePack file.
    /// Throws FormatError for .json files or v1 schema.
    static ModelPackage Load(const std::string& path);

    /// Find the layer package for the given \p color_layers, or nullptr if unavailable.
    const ModelLayerPackage* FindByColorLayers(int color_layers) const;

    /// Whether this package's scope matches the given vendor + material_type.
    bool MatchesScope(const std::string& vendor, const std::string& material_type) const;

    /// Rebuild the internal color_layers → layer_packages index.
    /// Called automatically by Load(); call manually after programmatic construction.
    void BuildIndex();

private:
    std::unordered_map<int, size_t> mode_index_; ///< color_layers -> layer_packages index
};

/// Manages multiple ModelPackage instances loaded from a directory.
class ModelPackageRegistry {
public:
    /// Discover and load all .msgpack files in \p dir.
    /// Individual files that fail to load are logged and skipped.
    void LoadFromDirectory(const std::string& dir);

    /// Load a single .msgpack file. Throws FormatError for .json files.
    void LoadSingle(const std::string& path);

    /// Select the best-matching package for a request.
    /// Requires ALL selected ColorDBs to share the same vendor + material_type.
    /// Returns nullptr if no package matches.
    const ModelPackage* Select(const std::string& vendor, const std::string& material_type,
                               const std::vector<std::string>& profile_channel_keys) const;

    const std::vector<ModelPackage>& All() const { return packages_; }

    bool Empty() const { return packages_.empty(); }

private:
    std::vector<ModelPackage> packages_;
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
