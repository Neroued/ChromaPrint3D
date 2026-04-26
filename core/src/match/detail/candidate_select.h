#pragma once

// Internal candidate selection logic for matching.
// NOT part of the public API.

#include "chromaprint3d/common.h"
#include "chromaprint3d/color.h"
#include "chromaprint3d/print_profile.h"
#include "chromaprint3d/recipe_map.h"
#include "chromaprint3d/model_package.h"
#include "chromaprint3d/kdtree.h"
#include "recipe_convert.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ChromaPrint3D {
namespace detail {

struct CandidateResult {
    bool valid = false;
    Lab mapped_lab;
    std::vector<uint8_t> recipe;
    float score_dist2 = std::numeric_limits<float>::infinity();
    float lab_dist2   = std::numeric_limits<float>::infinity();
    bool from_model   = false;
};

struct ModelLabProj {
    const Lab& operator()(const Lab& lab) const { return lab; }
};

struct ModelRgbProj {
    Rgb operator()(const Lab& lab) const { return lab.ToRgb(); }
};

using ModelLabTree = kdt::KDTree<Lab, 3, ModelLabProj, std::size_t, float>;
using ModelRgbTree = kdt::KDTree<Lab, 3, ModelRgbProj, std::size_t, float>;

struct PreparedModel {
    float threshold        = 5.0f;
    float margin           = 0.7f;
    int color_layers       = 0;
    LayerOrder layer_order = LayerOrder::Top2Bottom;

    // When owns_data == false, the span views reference ModelLayerPackage data
    // held by the ModelPackageRegistry (process-lifetime). The PreparedModel is
    // request-scoped and must not outlive the registry.
    bool owns_data = true;
    std::span<const Lab> pred_lab_view;
    std::span<const uint8_t> recipes_view;
    std::vector<Lab> pred_lab_owned;
    std::vector<uint8_t> recipes_owned;

    std::vector<std::size_t> kd_indices;
    ModelLabTree lab_tree;
    ModelRgbTree rgb_tree;

    std::vector<std::string> warnings;

    size_t NumCandidates() const { return pred_lab_view.size(); }

    const uint8_t* RecipeAt(size_t idx) const {
        if (idx >= NumCandidates() || color_layers <= 0) { return nullptr; }
        const size_t offset = idx * static_cast<size_t>(color_layers);
        if (offset + static_cast<size_t>(color_layers) > recipes_view.size()) { return nullptr; }
        return &recipes_view[offset];
    }
};

struct CandidateDecision {
    CandidateResult selected;
    float db_de        = 0.0f;
    float model_de     = 0.0f;
    bool model_queried = false;
};

/// Find the best DB candidate for a typed target color (`Lab` or `Rgb`).
/// The matching color space is encoded in the type `T`; there is no
/// `bool use_lab` flag.
template <typename T>
CandidateResult FindBestDbCandidate(const T& target_color,
                                    const std::vector<PreparedDB>& prepared_dbs,
                                    const PrintProfile& profile, const MatchConfig& cfg);

std::optional<PreparedModel> PrepareModel(const ModelPackage* model_package,
                                          const ModelGateConfig& model_gate,
                                          const PrintProfile& profile);

template <typename T>
CandidateResult FindBestModelCandidate(const T& target_color, const PreparedModel& model);

template <typename T>
CandidateDecision SelectCandidate(const T& target_color,
                                  const std::vector<PreparedDB>& prepared_dbs,
                                  const PrintProfile& profile, const MatchConfig& cfg,
                                  const PreparedModel* prepared_model, bool model_only);

// Explicit instantiations are emitted from candidate_select.cpp for
// `Lab` and `Rgb`. Other translation units must not instantiate these
// templates with a different type.
extern template CandidateResult FindBestDbCandidate<Lab>(const Lab&, const std::vector<PreparedDB>&,
                                                         const PrintProfile&, const MatchConfig&);
extern template CandidateResult FindBestDbCandidate<Rgb>(const Rgb&, const std::vector<PreparedDB>&,
                                                         const PrintProfile&, const MatchConfig&);

extern template CandidateResult FindBestModelCandidate<Lab>(const Lab&, const PreparedModel&);
extern template CandidateResult FindBestModelCandidate<Rgb>(const Rgb&, const PreparedModel&);

extern template CandidateDecision SelectCandidate<Lab>(const Lab&, const std::vector<PreparedDB>&,
                                                       const PrintProfile&, const MatchConfig&,
                                                       const PreparedModel*, bool);
extern template CandidateDecision SelectCandidate<Rgb>(const Rgb&, const std::vector<PreparedDB>&,
                                                       const PrintProfile&, const MatchConfig&,
                                                       const PreparedModel*, bool);

void WriteRecipe(RecipeMap& result, std::size_t pixel_idx, const std::vector<uint8_t>& recipe);

void WriteSourceMask(RecipeMap& result, std::size_t pixel_idx, bool from_model);

} // namespace detail
} // namespace ChromaPrint3D
