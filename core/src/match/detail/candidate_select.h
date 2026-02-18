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
    std::vector<Lab> pred_lab;
    std::vector<uint8_t> mapped_recipes;
    std::vector<std::size_t> kd_indices;
    ModelLabTree lab_tree;
    ModelRgbTree rgb_tree;

    size_t NumCandidates() const { return pred_lab.size(); }

    const uint8_t* RecipeAt(size_t idx) const {
        if (idx >= NumCandidates() || color_layers <= 0) { return nullptr; }
        const size_t offset = idx * static_cast<size_t>(color_layers);
        if (offset + static_cast<size_t>(color_layers) > mapped_recipes.size()) { return nullptr; }
        return &mapped_recipes[offset];
    }
};

struct CandidateDecision {
    CandidateResult selected;
    float db_de        = 0.0f;
    float model_de     = 0.0f;
    bool model_queried = false;
};

CandidateResult FindBestDbCandidate(const cv::Vec3f& target_color, bool use_lab,
                                    const std::vector<PreparedDB>& prepared_dbs,
                                    const PrintProfile& profile, const MatchConfig& cfg);

std::optional<PreparedModel> PrepareModel(const ModelPackage* model_package,
                                          const ModelGateConfig& model_gate,
                                          const PrintProfile& profile);

CandidateResult FindBestModelCandidate(const cv::Vec3f& target_color, bool use_lab,
                                       const PreparedModel& model);

CandidateDecision SelectCandidate(const cv::Vec3f& target_color, bool use_lab,
                                  const std::vector<PreparedDB>& prepared_dbs,
                                  const PrintProfile& profile, const MatchConfig& cfg,
                                  const PreparedModel* prepared_model, bool model_only);

void WriteRecipe(RecipeMap& result, std::size_t pixel_idx, const std::vector<uint8_t>& recipe);

void WriteSourceMask(RecipeMap& result, std::size_t pixel_idx, bool from_model);

} // namespace detail
} // namespace ChromaPrint3D
