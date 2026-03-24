#include "chromaprint3d/recipe_map.h"
#include "chromaprint3d/print_profile.h"
#include "chromaprint3d/raster_proc.h"
#include "chromaprint3d/color_db.h"
#include "chromaprint3d/error.h"
#include "detail/candidate_select.h"
#include "detail/recipe_convert.h"
#include "detail/dither.h"
#include "slic_segmenter.h"

#include <spdlog/spdlog.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ChromaPrint3D {
namespace {

constexpr int kKMeansMaxIter = 30;
constexpr double kKMeansEps  = 1e-4;

/// Automatically select optimal K for kmeans color clustering.
/// Uses WCSS elbow detection with a perceptual quality floor (Delta E).
int EstimateClusterCount(const cv::Mat& samples) {
    constexpr int kCandidates[]    = {4, 8, 16, 32, 64, 96, 128};
    constexpr int kNumCandidates   = 7;
    constexpr int kMaxSubsamples   = 8192;
    constexpr int kQuickIter       = 15;
    constexpr double kQuickEps     = 1e-3;
    constexpr float kMaxMeanDeltaE = 3.0f;

    cv::Mat sub = samples;
    if (samples.rows > kMaxSubsamples) {
        cv::Mat indices(samples.rows, 1, CV_32SC1);
        for (int i = 0; i < samples.rows; ++i) indices.at<int>(i) = i;
        cv::randShuffle(indices);
        sub = cv::Mat(kMaxSubsamples, samples.cols, samples.type());
        for (int i = 0; i < kMaxSubsamples; ++i) {
            samples.row(indices.at<int>(i)).copyTo(sub.row(i));
        }
    }
    const int n = sub.rows;

    std::vector<int> valid_candidates;
    std::vector<double> wcss;
    valid_candidates.reserve(kNumCandidates);
    wcss.reserve(kNumCandidates);

    const cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, kQuickIter,
                                    kQuickEps);

    for (int ci = 0; ci < kNumCandidates; ++ci) {
        const int k = kCandidates[ci];
        if (k >= n) break;

        cv::Mat labels, centers;
        double compactness =
            cv::kmeans(sub, k, labels, criteria, 2, cv::KMEANS_PP_CENTERS, centers);
        valid_candidates.push_back(k);
        wcss.push_back(compactness);

        spdlog::debug("AutoK: k={:3d}  WCSS={:.1f}  mean_dE={:.2f}", k, compactness,
                      std::sqrt(compactness / std::max(1, n)));
    }

    if (valid_candidates.size() <= 1) {
        return valid_candidates.empty() ? 16 : valid_candidates[0];
    }

    // Elbow detection: find maximum perpendicular distance from first-to-last chord
    const int m       = static_cast<int>(valid_candidates.size());
    const double x0   = valid_candidates[0];
    const double y0   = wcss[0];
    const double x1   = valid_candidates[m - 1];
    const double y1   = wcss[m - 1];
    const double dx   = x1 - x0;
    const double dy   = y1 - y0;
    const double norm = std::sqrt(dx * dx + dy * dy);

    int elbow_idx   = 0;
    double max_dist = 0.0;
    for (int i = 0; i < m; ++i) {
        const double d =
            std::abs(dy * valid_candidates[i] - dx * wcss[i] + x1 * y0 - y1 * x0) / norm;
        if (d > max_dist) {
            max_dist  = d;
            elbow_idx = i;
        }
    }

    int chosen_k  = valid_candidates[elbow_idx];
    float mean_dE = static_cast<float>(std::sqrt(wcss[elbow_idx] / std::max(1, n)));

    if (mean_dE > kMaxMeanDeltaE) {
        for (int i = elbow_idx + 1; i < m; ++i) {
            float dE_i = static_cast<float>(std::sqrt(wcss[i] / std::max(1, n)));
            if (dE_i <= kMaxMeanDeltaE) {
                chosen_k = valid_candidates[i];
                mean_dE  = dE_i;
                break;
            }
        }
        if (mean_dE > kMaxMeanDeltaE && !valid_candidates.empty()) {
            chosen_k = valid_candidates[m - 1];
            mean_dE  = static_cast<float>(std::sqrt(wcss[m - 1] / std::max(1, n)));
        }
    }

    chosen_k = std::clamp(chosen_k, 4, std::min(128, n / 4));

    spdlog::debug("AutoK: selected k={} (elbow_k={}, mean_dE={:.2f})", chosen_k,
                  valid_candidates[elbow_idx], mean_dE);
    return chosen_k;
}

void ValidateImageForMatch(const RasterProcResult& img, bool use_lab) {
    if (img.width <= 0 || img.height <= 0) {
        throw InputError("RasterProcResult width/height must be positive");
    }
    if (img.lab.empty()) { throw InputError("Image Lab data is empty"); }
    if (img.lab.type() != CV_32FC3) { throw InputError("Image Lab data must be CV_32FC3"); }
    if (img.lab.rows != img.height || img.lab.cols != img.width) {
        throw InputError("Image Lab size does not match RasterProcResult size");
    }
    if (!img.mask.empty() && (img.mask.rows != img.height || img.mask.cols != img.width)) {
        throw InputError("Image mask size does not match RasterProcResult size");
    }
    if (!use_lab) {
        if (img.rgb.empty()) { throw InputError("Image RGB data is empty"); }
        if (img.rgb.type() != CV_32FC3) { throw InputError("Image RGB data must be CV_32FC3"); }
        if (img.rgb.rows != img.height || img.rgb.cols != img.width) {
            throw InputError("Image RGB size does not match RasterProcResult size");
        }
    }
}

} // namespace

RecipeMap RecipeMap::MatchFromRaster(const RasterProcResult& img, std::span<const ColorDB> dbs,
                                     const PrintProfile& profile, const MatchConfig& cfg,
                                     const ModelPackage* model_package,
                                     const ModelGateConfig& model_gate, MatchStats* out_stats) {
    profile.Validate();
    if (dbs.empty() && !model_gate.model_only) {
        throw InputError("MatchFromRaster requires at least one ColorDB");
    }

    spdlog::info("MatchFromRaster: image={}x{}, dbs={}, color_space={}, k={}, clusters={}",
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

    DitherMethod effective_dither = cfg.dither;
    if (cfg.cluster_method == ClusterMethod::Slic && cfg.dither != DitherMethod::None) {
        spdlog::warn("MatchFromRaster: SLIC selected, forcing dither=none");
        effective_dither = DitherMethod::None;
    }

    // ── Dither path (takes precedence over cluster / per-pixel) ──────────
    if (effective_dither != DitherMethod::None) {
        const detail::PreparedModel* pm =
            prepared_model.has_value() ? &prepared_model.value() : nullptr;
        MatchConfig dither_cfg = cfg;
        dither_cfg.dither      = effective_dither;

        detail::DitherStats ds;
        if (effective_dither == DitherMethod::BlueNoise) {
            detail::MatchWithBlueNoiseDither(result, target, img.mask, use_lab, prepared_dbs,
                                             profile, dither_cfg, pm, model_only,
                                             cfg.dither_strength, ds);
        } else {
            detail::MatchWithFloydSteinberg(result, target, img.mask, use_lab, prepared_dbs,
                                            profile, dither_cfg, pm, model_only,
                                            cfg.dither_strength, ds);
        }

        stat_total_queries = ds.total_queries;
        stat_db_only       = ds.db_only;
        stat_model_used    = ds.model_used;
        stat_model_queries = ds.model_queries;
        stat_sum_db_de     = ds.sum_db_de;
        stat_sum_model_de  = ds.sum_model_de;
        write_stats();
        return result;
    }

    auto run_per_pixel_matching = [&]() {
        int parallel_total_queries   = 0;
        int parallel_db_only         = 0;
        int parallel_model_used      = 0;
        int parallel_model_queries   = 0;
        double parallel_sum_db_de    = 0.0;
        double parallel_sum_model_de = 0.0;
        std::atomic<bool> has_parallel_error{false};
        std::string parallel_error_message;

#ifdef _OPENMP
#    pragma omp parallel for schedule(guided)                                                      \
        reduction(+ : parallel_total_queries, parallel_db_only, parallel_model_used,               \
                      parallel_model_queries, parallel_sum_db_de, parallel_sum_model_de)
#endif
        for (int i = 0; i < static_cast<int>(valid_indices.size()); ++i) {
            if (has_parallel_error.load(std::memory_order_relaxed)) { continue; }

            try {
                const std::size_t idx = valid_indices[static_cast<std::size_t>(i)];
                const int r           = static_cast<int>(idx / static_cast<std::size_t>(img.width));
                const int c           = static_cast<int>(idx % static_cast<std::size_t>(img.width));
                const cv::Vec3f target_color             = target.at<cv::Vec3f>(r, c);
                const detail::CandidateDecision decision = detail::SelectCandidate(
                    target_color, use_lab, prepared_dbs, profile, cfg,
                    prepared_model ? &prepared_model.value() : nullptr, model_only);
                if (!decision.selected.valid) {
                    throw MatchError("No valid match candidate after DB/model selection");
                }

                ++parallel_total_queries;
                parallel_sum_db_de += static_cast<double>(decision.db_de);
                if (decision.model_queried) {
                    ++parallel_model_queries;
                    parallel_sum_model_de += static_cast<double>(decision.model_de);
                }
                if (decision.selected.from_model) {
                    ++parallel_model_used;
                } else {
                    ++parallel_db_only;
                }

                result.mapped_color[idx] = decision.selected.mapped_lab;
                detail::WriteRecipe(result, idx, decision.selected.recipe);
                detail::WriteSourceMask(result, idx, decision.selected.from_model);
            } catch (const std::exception& e) {
#ifdef _OPENMP
#    pragma omp critical(match_from_raster_error)
#endif
                {
                    if (!has_parallel_error.load(std::memory_order_relaxed)) {
                        parallel_error_message = e.what();
                        has_parallel_error.store(true, std::memory_order_relaxed);
                    }
                }
            } catch (...) {
#ifdef _OPENMP
#    pragma omp critical(match_from_raster_error)
#endif
                {
                    if (!has_parallel_error.load(std::memory_order_relaxed)) {
                        parallel_error_message = "Unknown error in parallel raster matching";
                        has_parallel_error.store(true, std::memory_order_relaxed);
                    }
                }
            }
        }

        if (has_parallel_error.load(std::memory_order_relaxed)) {
            throw MatchError(parallel_error_message.empty() ? "Parallel raster matching failed"
                                                            : parallel_error_message);
        }

        stat_total_queries = parallel_total_queries;
        stat_db_only       = parallel_db_only;
        stat_model_used    = parallel_model_used;
        stat_model_queries = parallel_model_queries;
        stat_sum_db_de     = parallel_sum_db_de;
        stat_sum_model_de  = parallel_sum_model_de;
    };

    if (cfg.cluster_method == ClusterMethod::Slic) {
        const int requested_superpixels = std::max(0, cfg.slic_target_superpixels);
        const int target_superpixels =
            std::min(requested_superpixels, static_cast<int>(valid_indices.size()));
        const bool use_slic = (requested_superpixels > 1 && target_superpixels > 1 &&
                               static_cast<std::size_t>(target_superpixels) < valid_indices.size());
        spdlog::debug(
            "MatchFromRaster: valid_pixels={}, cluster_method=slic, clustering={} (target={})",
            valid_indices.size(), use_slic ? "yes" : "no", target_superpixels);

        if (!use_slic) {
            run_per_pixel_matching();
            write_stats();
            return result;
        }

        detail::SlicConfig slic_cfg;
        slic_cfg.target_superpixels = target_superpixels;
        slic_cfg.compactness        = std::max(0.001f, cfg.slic_compactness);
        slic_cfg.iterations         = std::max(1, cfg.slic_iterations);
        slic_cfg.min_region_ratio   = std::clamp(cfg.slic_min_region_ratio, 0.0f, 1.0f);

        detail::SlicResult slic = detail::SegmentBySlic(target, img.mask, slic_cfg);
        const int slic_clusters = static_cast<int>(slic.centers.size());
        if (slic_clusters <= 1) {
            run_per_pixel_matching();
            write_stats();
            return result;
        }

        std::vector<detail::CandidateResult> cluster_candidates(
            static_cast<std::size_t>(slic_clusters));
        for (int i = 0; i < slic_clusters; ++i) {
            const cv::Vec3f center_color             = slic.centers[static_cast<std::size_t>(i)];
            const detail::CandidateDecision decision = detail::SelectCandidate(
                center_color, use_lab, prepared_dbs, profile, cfg,
                prepared_model ? &prepared_model.value() : nullptr, model_only);
            if (!decision.selected.valid) {
                throw MatchError("SLIC center has no valid match candidate");
            }
            cluster_candidates[static_cast<std::size_t>(i)] = decision.selected;
            accumulate_stats(decision);
        }

        std::atomic<bool> has_label_error{false};
        std::string label_error_message;
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < static_cast<int>(valid_indices.size()); ++i) {
            if (has_label_error.load(std::memory_order_relaxed)) { continue; }

            try {
                const std::size_t idx = valid_indices[static_cast<std::size_t>(i)];
                const int r           = static_cast<int>(idx / static_cast<std::size_t>(img.width));
                const int c           = static_cast<int>(idx % static_cast<std::size_t>(img.width));
                const int label       = slic.labels.at<int>(r, c);
                if (label < 0 || label >= slic_clusters) { throw InputError("Invalid SLIC label"); }

                const detail::CandidateResult& best =
                    cluster_candidates[static_cast<std::size_t>(label)];
                result.mapped_color[idx] = best.mapped_lab;
                detail::WriteRecipe(result, idx, best.recipe);
                detail::WriteSourceMask(result, idx, best.from_model);
            } catch (const std::exception& e) {
#ifdef _OPENMP
#    pragma omp critical(match_from_raster_label_error)
#endif
                {
                    if (!has_label_error.load(std::memory_order_relaxed)) {
                        label_error_message = e.what();
                        has_label_error.store(true, std::memory_order_relaxed);
                    }
                }
            } catch (...) {
#ifdef _OPENMP
#    pragma omp critical(match_from_raster_label_error)
#endif
                {
                    if (!has_label_error.load(std::memory_order_relaxed)) {
                        label_error_message = "Unknown error in SLIC label writeback";
                        has_label_error.store(true, std::memory_order_relaxed);
                    }
                }
            }
        }

        if (has_label_error.load(std::memory_order_relaxed)) {
            throw InputError(label_error_message.empty() ? "SLIC label writeback failed"
                                                         : label_error_message);
        }

        write_stats();
        return result;
    }

    // cluster_count == 0 → auto-detect; == 1 → per-pixel; >= 2 → manual
    const bool auto_k            = (cfg.cluster_count == 0);
    const int requested_clusters = auto_k ? 0 : std::max(0, cfg.cluster_count);
    const int num_valid          = static_cast<int>(valid_indices.size());

    // Build sample matrix (needed for both auto-K detection and actual clustering)
    const cv::Mat target_flat = target.reshape(1, static_cast<int>(pixel_count));
    cv::Mat samples(num_valid, 3, CV_32FC1);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < num_valid; ++i) {
        target_flat.row(static_cast<int>(valid_indices[static_cast<std::size_t>(i)]))
            .copyTo(samples.row(i));
    }

    const int resolved_k =
        auto_k ? EstimateClusterCount(samples) : std::min(requested_clusters, num_valid);
    const bool use_cluster =
        (resolved_k > 1 && static_cast<std::size_t>(resolved_k) < valid_indices.size());

    spdlog::debug("MatchFromRaster: valid_pixels={}, cluster_method=kmeans, clustering={} (k={}{})",
                  valid_indices.size(), use_cluster ? "yes" : "no", resolved_k,
                  auto_k ? ", auto" : "");

    if (!use_cluster) {
        run_per_pixel_matching();
        write_stats();
        return result;
    }

    cv::Mat labels;
    cv::Mat centers;
    const cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                                    kKMeansMaxIter, kKMeansEps);
    cv::kmeans(samples, resolved_k, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    std::vector<detail::CandidateResult> cluster_candidates(static_cast<std::size_t>(resolved_k));
    for (int i = 0; i < resolved_k; ++i) {
        const cv::Vec3f center_color(centers.at<float>(i, 0), centers.at<float>(i, 1),
                                     centers.at<float>(i, 2));
        const detail::CandidateDecision decision =
            detail::SelectCandidate(center_color, use_lab, prepared_dbs, profile, cfg,
                                    prepared_model ? &prepared_model.value() : nullptr, model_only);
        if (!decision.selected.valid) {
            throw MatchError("Cluster center has no valid match candidate");
        }
        cluster_candidates[static_cast<std::size_t>(i)] = decision.selected;
        accumulate_stats(decision);
    }

    std::atomic<bool> has_label_error{false};
    std::string label_error_message;
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < static_cast<int>(valid_indices.size()); ++i) {
        if (has_label_error.load(std::memory_order_relaxed)) { continue; }

        try {
            const int label = labels.at<int>(i, 0);
            if (label < 0 || label >= resolved_k) { throw InputError("Invalid kmeans label"); }
            const detail::CandidateResult& best =
                cluster_candidates[static_cast<std::size_t>(label)];
            const std::size_t idx    = valid_indices[static_cast<std::size_t>(i)];
            result.mapped_color[idx] = best.mapped_lab;
            detail::WriteRecipe(result, idx, best.recipe);
            detail::WriteSourceMask(result, idx, best.from_model);
        } catch (const std::exception& e) {
#ifdef _OPENMP
#    pragma omp critical(match_from_raster_label_error)
#endif
            {
                if (!has_label_error.load(std::memory_order_relaxed)) {
                    label_error_message = e.what();
                    has_label_error.store(true, std::memory_order_relaxed);
                }
            }
        } catch (...) {
#ifdef _OPENMP
#    pragma omp critical(match_from_raster_label_error)
#endif
            {
                if (!has_label_error.load(std::memory_order_relaxed)) {
                    label_error_message = "Unknown error in kmeans label writeback";
                    has_label_error.store(true, std::memory_order_relaxed);
                }
            }
        }
    }

    if (has_label_error.load(std::memory_order_relaxed)) {
        throw InputError(label_error_message.empty() ? "kmeans label writeback failed"
                                                     : label_error_message);
    }

    write_stats();
    return result;
}

} // namespace ChromaPrint3D
