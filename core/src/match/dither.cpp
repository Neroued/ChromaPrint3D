#include "detail/dither.h"
#include "detail/candidate_select.h"
#include "detail/match_target.h"
#include "detail/match_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace ChromaPrint3D {
namespace detail {

namespace {

constexpr float kBlueNoiseLScale  = 8.0f;
constexpr float kBlueNoiseABScale = 6.0f;

// Blue-noise bias vector applied to the typed target color. For Lab the L*
// component receives a stronger nudge than the a*/b* components (matching
// human luminance sensitivity); for linear RGB all three channels share the
// same small scale.
template <typename T>
inline T BlueNoiseBias(const T& src, float bias) {
    static_assert(std::is_same_v<T, Lab> || std::is_same_v<T, Rgb>,
                  "BlueNoiseBias supports only Lab or Rgb target types");
    if constexpr (std::is_same_v<T, Lab>) {
        return Lab(src.l() + bias * kBlueNoiseLScale, src.a() + bias * kBlueNoiseABScale,
                   src.b() + bias * kBlueNoiseABScale);
    } else {
        return Rgb(src.r() + bias * 0.05f, src.g() + bias * 0.05f, src.b() + bias * 0.05f);
    }
}

// Add the Floyd-Steinberg quantization error (in the matching color space)
// to a typed target color.
template <typename T>
inline T ApplyError(const T& src, const Vec3f& err, float strength) {
    static_assert(std::is_same_v<T, Lab> || std::is_same_v<T, Rgb>,
                  "ApplyError supports only Lab or Rgb target types");
    if constexpr (std::is_same_v<T, Lab>) {
        return Lab(src.l() + err.x * strength, src.a() + err.y * strength,
                   src.b() + err.z * strength);
    } else {
        return Rgb(src.r() + err.x * strength, src.g() + err.y * strength,
                   src.b() + err.z * strength);
    }
}

// Compute the quantization error in the matching color space (typed target
// minus the matched candidate, projected back to the same space).
template <typename T>
inline Vec3f QuantError(const T& original, const Lab& matched_lab) {
    static_assert(std::is_same_v<T, Lab> || std::is_same_v<T, Rgb>,
                  "QuantError supports only Lab or Rgb target types");
    if constexpr (std::is_same_v<T, Lab>) {
        return Vec3f(original.l() - matched_lab.l(), original.a() - matched_lab.a(),
                     original.b() - matched_lab.b());
    } else {
        const Rgb matched_rgb = matched_lab.ToRgb();
        return Vec3f(original.r() - matched_rgb.r(), original.g() - matched_rgb.g(),
                     original.b() - matched_rgb.b());
    }
}

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

template <typename T>
void MatchWithBlueNoiseDither(RecipeMap& result, const cv::Mat& target, const cv::Mat& mask,
                              const std::vector<PreparedDB>& prepared_dbs,
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

            const T original = MakeTarget<T>(target.at<cv::Vec3f>(r, c));
            const T adjusted = BlueNoiseBias<T>(original, bias);

            const CandidateDecision decision = SelectCandidate<T>(adjusted, prepared_dbs, profile,
                                                                  cfg, prepared_model, model_only);
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

template <typename T>
void MatchWithFloydSteinberg(RecipeMap& result, const cv::Mat& target, const cv::Mat& mask,
                             const std::vector<PreparedDB>& prepared_dbs,
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

            const T original = MakeTarget<T>(target.at<cv::Vec3f>(r, c));
            const T adjusted =
                ApplyError<T>(original, error_cur[static_cast<std::size_t>(c)], strength);

            const CandidateDecision decision = SelectCandidate<T>(adjusted, prepared_dbs, profile,
                                                                  cfg, prepared_model, model_only);
            if (!decision.selected.valid) { continue; }

            result.mapped_color[idx] = decision.selected.mapped_lab;
            WriteRecipe(result, idx, decision.selected.recipe);
            WriteSourceMask(result, idx, decision.selected.from_model);
            AccumulateStats(stats, decision);

            // Error is computed against the *original* (un-adjusted) target
            // to prevent error amplification across pixels.
            const Vec3f quant_error = QuantError<T>(original, decision.selected.mapped_lab);

            // Floyd-Steinberg distribution: right 7/16, bottom-left 3/16,
            // bottom 5/16, bottom-right 1/16.
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

// ── Explicit instantiations ─────────────────────────────────────────────────

template void MatchWithBlueNoiseDither<Lab>(RecipeMap&, const cv::Mat&, const cv::Mat&,
                                            const std::vector<PreparedDB>&, const PrintProfile&,
                                            const MatchConfig&, const PreparedModel*, bool, float,
                                            DitherStats&);
template void MatchWithBlueNoiseDither<Rgb>(RecipeMap&, const cv::Mat&, const cv::Mat&,
                                            const std::vector<PreparedDB>&, const PrintProfile&,
                                            const MatchConfig&, const PreparedModel*, bool, float,
                                            DitherStats&);

template void MatchWithFloydSteinberg<Lab>(RecipeMap&, const cv::Mat&, const cv::Mat&,
                                           const std::vector<PreparedDB>&, const PrintProfile&,
                                           const MatchConfig&, const PreparedModel*, bool, float,
                                           DitherStats&);
template void MatchWithFloydSteinberg<Rgb>(RecipeMap&, const cv::Mat&, const cv::Mat&,
                                           const std::vector<PreparedDB>&, const PrintProfile&,
                                           const MatchConfig&, const PreparedModel*, bool, float,
                                           DitherStats&);

} // namespace detail
} // namespace ChromaPrint3D
