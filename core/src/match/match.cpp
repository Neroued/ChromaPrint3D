#include "chromaprint3d/recipe_map.h"
#include "chromaprint3d/print_profile.h"
#include "chromaprint3d/imgproc.h"
#include "chromaprint3d/color_db.h"
#include "chromaprint3d/error.h"
#include "detail/candidate_select.h"
#include "detail/recipe_convert.h"

#include <spdlog/spdlog.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ChromaPrint3D {
namespace {

constexpr int kKMeansMaxIter = 30;
constexpr double kKMeansEps  = 1e-4;

void ValidateImageForMatch(const ImgProcResult& img, bool use_lab) {
    if (img.width <= 0 || img.height <= 0) {
        throw InputError("ImgProcResult width/height must be positive");
    }
    if (img.lab.empty()) { throw InputError("Image Lab data is empty"); }
    if (img.lab.type() != CV_32FC3) { throw InputError("Image Lab data must be CV_32FC3"); }
    if (img.lab.rows != img.height || img.lab.cols != img.width) {
        throw InputError("Image Lab size does not match ImgProcResult size");
    }
    if (!img.mask.empty() && (img.mask.rows != img.height || img.mask.cols != img.width)) {
        throw InputError("Image mask size does not match ImgProcResult size");
    }
    if (!use_lab) {
        if (img.rgb.empty()) { throw InputError("Image RGB data is empty"); }
        if (img.rgb.type() != CV_32FC3) {
            throw InputError("Image RGB data must be CV_32FC3");
        }
        if (img.rgb.rows != img.height || img.rgb.cols != img.width) {
            throw InputError("Image RGB size does not match ImgProcResult size");
        }
    }
}

} // namespace

RecipeMap RecipeMap::MatchFromImage(const ImgProcResult& img, std::span<const ColorDB> dbs,
                                    const PrintProfile& profile, const MatchConfig& cfg,
                                    const ModelPackage* model_package,
                                    const ModelGateConfig& model_gate, MatchStats* out_stats) {
    profile.Validate();
    if (dbs.empty() && !model_gate.model_only) {
        throw InputError("MatchFromImage requires at least one ColorDB");
    }

    spdlog::info("MatchFromImage: image={}x{}, dbs={}, color_space={}, k={}, clusters={}",
                 img.width, img.height, dbs.size(),
                 cfg.color_space == ColorSpace::Lab ? "Lab" : "RGB", cfg.k_candidates,
                 cfg.cluster_count);

    const bool use_lab = (cfg.color_space == ColorSpace::Lab);
    ValidateImageForMatch(img, use_lab);
    const bool model_only = model_gate.model_only;

    std::vector<detail::PreparedDB> prepared_dbs;
    if (!model_only) {
        prepared_dbs = detail::PrepareDBs(dbs, profile);
        spdlog::debug("PrepareDBs: {} DB(s) prepared", prepared_dbs.size());
    }

    const std::optional<detail::PreparedModel> prepared_model =
        detail::PrepareModel(model_package, model_gate, profile);
    if (prepared_model.has_value()) {
        spdlog::debug("PrepareModel: model ready, model_only={}", model_only);
    } else {
        spdlog::debug("PrepareModel: no model available");
    }
    if (model_only && !prepared_model.has_value()) {
        throw ConfigError("Model-only matching requires a compatible model package");
    }

    int stat_total_queries   = 0;
    int stat_db_only         = 0;
    int stat_model_used      = 0;
    int stat_model_queries   = 0;
    double stat_sum_db_de    = 0.0;
    double stat_sum_model_de = 0.0;

    RecipeMap result;
    result.name         = img.name;
    result.width        = img.width;
    result.height       = img.height;
    result.color_layers = profile.color_layers;
    result.num_channels = static_cast<int>(profile.NumChannels());
    result.layer_order  = profile.layer_order;

    const std::size_t pixel_count =
        static_cast<std::size_t>(img.width) * static_cast<std::size_t>(img.height);
    result.recipes.assign(pixel_count * static_cast<std::size_t>(result.color_layers), 0);
    result.mask.assign(pixel_count, 0);
    result.mapped_color.assign(pixel_count, Lab());
    result.source_mask.assign(pixel_count, static_cast<uint8_t>(0));

    const cv::Mat& target = use_lab ? img.lab : img.rgb;
    const bool has_mask   = !img.mask.empty();
    std::vector<std::size_t> valid_indices;
    valid_indices.reserve(pixel_count);

    for (int r = 0; r < img.height; ++r) {
        const uint8_t* mask_row = has_mask ? img.mask.ptr<uint8_t>(r) : nullptr;
        for (int c = 0; c < img.width; ++c) {
            const std::size_t idx =
                static_cast<size_t>(r) * static_cast<size_t>(img.width) + static_cast<size_t>(c);
            const uint8_t mask_value = has_mask ? mask_row[c] : static_cast<uint8_t>(255);
            result.mask[idx]         = (mask_value == 0) ? 0 : mask_value;
            if (mask_value != 0) { valid_indices.push_back(idx); }
        }
    }

    if (valid_indices.empty()) {
        if (out_stats) { *out_stats = MatchStats{}; }
        return result;
    }

    const int requested_clusters = std::max(0, cfg.cluster_count);
    const int k_clusters   = std::min(requested_clusters, static_cast<int>(valid_indices.size()));
    const bool use_cluster = (requested_clusters > 1 && k_clusters > 1 &&
                              static_cast<std::size_t>(k_clusters) < valid_indices.size());
    spdlog::info("MatchFromImage: valid_pixels={}, clustering={} (k={})", valid_indices.size(),
                 use_cluster ? "yes" : "no", k_clusters);

    auto accumulate_stats = [&](const detail::CandidateDecision& decision) {
        ++stat_total_queries;
        stat_sum_db_de += static_cast<double>(decision.db_de);
        if (decision.model_queried) {
            ++stat_model_queries;
            stat_sum_model_de += static_cast<double>(decision.model_de);
        }
        if (decision.selected.from_model) {
            ++stat_model_used;
        } else {
            ++stat_db_only;
        }
    };

    auto write_stats = [&]() {
        if (!out_stats) { return; }
        out_stats->clusters_total = stat_total_queries;
        out_stats->db_only        = stat_db_only;
        out_stats->model_fallback = stat_model_used;
        out_stats->model_queries  = stat_model_queries;
        out_stats->avg_db_de =
            (stat_total_queries > 0)
                ? static_cast<float>(stat_sum_db_de / static_cast<double>(stat_total_queries))
                : 0.0f;
        out_stats->avg_model_de =
            (stat_model_queries > 0)
                ? static_cast<float>(stat_sum_model_de / static_cast<double>(stat_model_queries))
                : 0.0f;
    };

    if (!use_cluster) {
        for (std::size_t idx : valid_indices) {
            const int r = static_cast<int>(idx / static_cast<std::size_t>(img.width));
            const int c = static_cast<int>(idx % static_cast<std::size_t>(img.width));
            const cv::Vec3f target_color = target.at<cv::Vec3f>(r, c);
            const detail::CandidateDecision decision = detail::SelectCandidate(
                target_color, use_lab, prepared_dbs, profile, cfg,
                prepared_model ? &prepared_model.value() : nullptr, model_only);
            if (!decision.selected.valid) {
                throw MatchError("No valid match candidate after DB/model selection");
            }

            accumulate_stats(decision);
            result.mapped_color[idx] = decision.selected.mapped_lab;
            detail::WriteRecipe(result, idx, decision.selected.recipe);
            detail::WriteSourceMask(result, idx, decision.selected.from_model);
        }
        write_stats();
        return result;
    }

    const cv::Mat target_flat = target.reshape(1, static_cast<int>(pixel_count));
    cv::Mat samples(static_cast<int>(valid_indices.size()), 3, CV_32FC1);
    for (std::size_t i = 0; i < valid_indices.size(); ++i) {
        target_flat.row(static_cast<int>(valid_indices[i]))
            .copyTo(samples.row(static_cast<int>(i)));
    }

    cv::Mat labels;
    cv::Mat centers;
    const cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                                    kKMeansMaxIter, kKMeansEps);
    cv::kmeans(samples, k_clusters, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    std::vector<detail::CandidateResult> cluster_candidates(static_cast<std::size_t>(k_clusters));
    for (int i = 0; i < k_clusters; ++i) {
        const cv::Vec3f center_color(centers.at<float>(i, 0), centers.at<float>(i, 1),
                                     centers.at<float>(i, 2));
        const detail::CandidateDecision decision = detail::SelectCandidate(
            center_color, use_lab, prepared_dbs, profile, cfg,
            prepared_model ? &prepared_model.value() : nullptr, model_only);
        if (!decision.selected.valid) {
            throw MatchError("Cluster center has no valid match candidate");
        }
        cluster_candidates[static_cast<std::size_t>(i)] = decision.selected;
        accumulate_stats(decision);
    }

    for (std::size_t i = 0; i < valid_indices.size(); ++i) {
        const int label = labels.at<int>(static_cast<int>(i), 0);
        if (label < 0 || label >= k_clusters) { throw InputError("Invalid kmeans label"); }
        const detail::CandidateResult& best = cluster_candidates[static_cast<std::size_t>(label)];
        const std::size_t idx               = valid_indices[i];
        result.mapped_color[idx]            = best.mapped_lab;
        detail::WriteRecipe(result, idx, best.recipe);
        detail::WriteSourceMask(result, idx, best.from_model);
    }

    write_stats();
    return result;
}

} // namespace ChromaPrint3D
