#include "detail/dither.h"
#include "detail/candidate_select.h"
#include "detail/match_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ChromaPrint3D {
namespace detail {

namespace {

constexpr float kBlueNoiseLScale  = 8.0f;
constexpr float kBlueNoiseABScale = 6.0f;

inline void AccumulateStats(DitherStats& stats, const CandidateDecision& decision) {
    ++stats.total_queries;
    stats.sum_db_de += static_cast<double>(decision.db_de);
    if (decision.model_queried) {
        ++stats.model_queries;
        stats.sum_model_de += static_cast<double>(decision.model_de);
    }
    if (decision.selected.from_model) {
        ++stats.model_used;
    } else {
        ++stats.db_only;
    }
}

} // namespace

void MatchWithBlueNoiseDither(RecipeMap& result, const cv::Mat& target, const cv::Mat& mask,
                              bool use_lab, const std::vector<PreparedDB>& prepared_dbs,
                              const PrintProfile& profile, const MatchConfig& cfg,
                              const PreparedModel* prepared_model, bool model_only, float strength,
                              DitherStats& stats) {
    const int W         = result.width;
    const int H         = result.height;
    const bool has_mask = !mask.empty();

    spdlog::debug("MatchWithBlueNoiseDither: {}x{}, strength={:.2f}", W, H, strength);

    int local_total      = 0;
    int local_db_only    = 0;
    int local_model_used = 0;
    int local_model_q    = 0;
    double local_sum_db  = 0.0;
    double local_sum_mdl = 0.0;

#pragma omp parallel for schedule(dynamic, 16)                                                     \
    reduction(+ : local_total, local_db_only, local_model_used, local_model_q, local_sum_db,       \
                  local_sum_mdl)
    for (int r = 0; r < H; ++r) {
        const uint8_t* mask_row = has_mask ? mask.ptr<uint8_t>(r) : nullptr;
        for (int c = 0; c < W; ++c) {
            const std::size_t idx = static_cast<std::size_t>(r) * static_cast<std::size_t>(W) +
                                    static_cast<std::size_t>(c);

            if (has_mask && mask_row[c] == 0) { continue; }

            const float noise =
                static_cast<float>(kBlueNoise[r % kBlueNoiseSize][c % kBlueNoiseSize]) / 255.0f;
            const float bias = (noise - 0.5f) * strength;

            cv::Vec3f adjusted = target.at<cv::Vec3f>(r, c);
            if (use_lab) {
                adjusted[0] += bias * kBlueNoiseLScale;
                adjusted[1] += bias * kBlueNoiseABScale;
                adjusted[2] += bias * kBlueNoiseABScale;
            } else {
                adjusted[0] += bias * 0.05f;
                adjusted[1] += bias * 0.05f;
                adjusted[2] += bias * 0.05f;
            }

            const CandidateDecision decision = SelectCandidate(
                adjusted, use_lab, prepared_dbs, profile, cfg, prepared_model, model_only);
            if (!decision.selected.valid) { continue; }

            result.mapped_color[idx] = decision.selected.mapped_lab;
            WriteRecipe(result, idx, decision.selected.recipe);
            WriteSourceMask(result, idx, decision.selected.from_model);

            ++local_total;
            local_sum_db += static_cast<double>(decision.db_de);
            if (decision.model_queried) {
                ++local_model_q;
                local_sum_mdl += static_cast<double>(decision.model_de);
            }
            if (decision.selected.from_model) {
                ++local_model_used;
            } else {
                ++local_db_only;
            }
        }
    }

    stats.total_queries = local_total;
    stats.db_only       = local_db_only;
    stats.model_used    = local_model_used;
    stats.model_queries = local_model_q;
    stats.sum_db_de     = local_sum_db;
    stats.sum_model_de  = local_sum_mdl;
}

void MatchWithFloydSteinberg(RecipeMap& result, const cv::Mat& target, const cv::Mat& mask,
                             bool use_lab, const std::vector<PreparedDB>& prepared_dbs,
                             const PrintProfile& profile, const MatchConfig& cfg,
                             const PreparedModel* prepared_model, bool model_only, float strength,
                             DitherStats& stats) {
    const int W         = result.width;
    const int H         = result.height;
    const bool has_mask = !mask.empty();

    spdlog::debug("MatchWithFloydSteinberg: {}x{}, strength={:.2f}", W, H, strength);

    // Two-row rolling error buffer: current row and next row.
    const std::size_t row_size = static_cast<std::size_t>(W);
    std::vector<Vec3f> error_cur(row_size, Vec3f());
    std::vector<Vec3f> error_next(row_size, Vec3f());

    for (int r = 0; r < H; ++r) {
        std::fill(error_next.begin(), error_next.end(), Vec3f());
        const uint8_t* mask_row = has_mask ? mask.ptr<uint8_t>(r) : nullptr;

        for (int c = 0; c < W; ++c) {
            const std::size_t idx = static_cast<std::size_t>(r) * static_cast<std::size_t>(W) +
                                    static_cast<std::size_t>(c);

            if (has_mask && mask_row[c] == 0) { continue; }

            const cv::Vec3f original = target.at<cv::Vec3f>(r, c);

            cv::Vec3f adjusted;
            adjusted[0] = original[0] + error_cur[static_cast<std::size_t>(c)].x * strength;
            adjusted[1] = original[1] + error_cur[static_cast<std::size_t>(c)].y * strength;
            adjusted[2] = original[2] + error_cur[static_cast<std::size_t>(c)].z * strength;

            const CandidateDecision decision = SelectCandidate(
                adjusted, use_lab, prepared_dbs, profile, cfg, prepared_model, model_only);
            if (!decision.selected.valid) { continue; }

            result.mapped_color[idx] = decision.selected.mapped_lab;
            WriteRecipe(result, idx, decision.selected.recipe);
            WriteSourceMask(result, idx, decision.selected.from_model);
            AccumulateStats(stats, decision);

            // Compute quantization error in the matching color space.
            // Use original target (not adjusted) to prevent error amplification.
            Lab original_lab = use_lab ? Lab(original[0], original[1], original[2])
                                       : Rgb(original[0], original[1], original[2]).ToLab();
            Lab matched_lab  = decision.selected.mapped_lab;
            Vec3f quant_error(original_lab.l() - matched_lab.l(),
                              original_lab.a() - matched_lab.a(),
                              original_lab.b() - matched_lab.b());

            // When matching in RGB space, convert error back to the working space.
            if (!use_lab) {
                Rgb original_rgb(original[0], original[1], original[2]);
                Rgb matched_rgb = matched_lab.ToRgb();
                quant_error =
                    Vec3f(original_rgb.r() - matched_rgb.r(), original_rgb.g() - matched_rgb.g(),
                          original_rgb.b() - matched_rgb.b());
            }

            // Floyd-Steinberg distribution: right 7/16, bottom-left 3/16, bottom 5/16, bottom-right
            // 1/16
            auto distribute = [&](int dc, int dr, float weight) {
                const int nc = c + dc;
                const int nr = r + dr;
                if (nc < 0 || nc >= W || nr < 0 || nr >= H) { return; }
                Vec3f& dst = (nr == r) ? error_cur[static_cast<std::size_t>(nc)]
                                       : error_next[static_cast<std::size_t>(nc)];
                dst.x += quant_error.x * weight;
                dst.y += quant_error.y * weight;
                dst.z += quant_error.z * weight;
            };

            distribute(+1, 0, 7.0f / 16.0f);
            distribute(-1, +1, 3.0f / 16.0f);
            distribute(0, +1, 5.0f / 16.0f);
            distribute(+1, +1, 1.0f / 16.0f);
        }

        std::swap(error_cur, error_next);
    }
}

} // namespace detail
} // namespace ChromaPrint3D
