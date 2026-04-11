#include "chromaprint3d/vector_recipe_map.h"
#include "chromaprint3d/vector_proc.h"
#include "chromaprint3d/color_db.h"
#include "chromaprint3d/print_profile.h"
#include "chromaprint3d/recipe_map.h"
#include "chromaprint3d/error.h"
#include "detail/candidate_select.h"

#include <spdlog/spdlog.h>

#include <atomic>
#include <cstddef>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ChromaPrint3D {

namespace {

struct RgbHash {
    size_t operator()(const Rgb& c) const {
        auto h = std::hash<float>{}(c.x);
        h ^= std::hash<float>{}(c.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(c.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct RgbEqual {
    bool operator()(const Rgb& a, const Rgb& b) const {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
};

struct CachedMatch {
    std::vector<uint8_t> recipe;
    Lab matched_lab;
    float de        = 0.0f;
    bool from_model = false;
};

} // namespace

VectorRecipeMap VectorRecipeMap::Match(const VectorProcResult& result, std::span<const ColorDB> dbs,
                                       const PrintProfile& profile, const MatchConfig& cfg,
                                       const ModelPackage* model_package,
                                       const ModelGateConfig& model_gate, MatchStats* out_stats) {
    if (dbs.empty() && !model_gate.model_only) {
        throw InputError("VectorRecipeMap::Match: no ColorDBs provided");
    }
    profile.Validate();

    VectorRecipeMap map;
    map.color_layers = profile.color_layers;
    map.num_channels = profile.NumChannels();
    map.layer_order  = profile.layer_order;

    if (result.shapes.empty()) { return map; }

    const bool use_lab    = (cfg.color_space == ColorSpace::Lab);
    const bool model_only = model_gate.model_only;
    std::vector<detail::PreparedDB> prepared_dbs;
    if (!model_only) { prepared_dbs = detail::PrepareDBs(dbs, profile); }

    const std::optional<detail::PreparedModel> prepared_model =
        detail::PrepareModel(model_package, model_gate, profile);
    if (model_only && !prepared_model.has_value()) {
        throw ConfigError("Model-only matching requires a compatible model package");
    }
    if (!model_only && prepared_dbs.empty() && !prepared_model.has_value()) {
        throw ConfigError("No compatible ColorDB entries for color_layers=" +
                          std::to_string(profile.color_layers) +
                          "; try using the default layer count (5 or 10)");
    }

    std::vector<int> shape_to_unique_idx(result.shapes.size(), -1);
    std::vector<Rgb> unique_colors;
    unique_colors.reserve(result.shapes.size());
    std::unordered_map<Rgb, int, RgbHash, RgbEqual> unique_index_by_color;
    unique_index_by_color.reserve(result.shapes.size());

    std::size_t solid_shape_count = 0;
    for (std::size_t i = 0; i < result.shapes.size(); ++i) {
        const VectorShape& shape = result.shapes[i];
        if (shape.fill_type != FillType::Solid) { continue; }
        ++solid_shape_count;
        const auto [it, inserted] =
            unique_index_by_color.emplace(shape.fill_color, static_cast<int>(unique_colors.size()));
        if (inserted) { unique_colors.push_back(shape.fill_color); }
        shape_to_unique_idx[i] = it->second;
    }

    int stat_total_queries   = 0;
    int stat_db_only         = 0;
    int stat_model_used      = 0;
    int stat_model_queries   = 0;
    double stat_sum_db_de    = 0.0;
    double stat_sum_model_de = 0.0;

    std::vector<CachedMatch> unique_matches(unique_colors.size());
    std::vector<uint8_t> unique_match_ready(unique_colors.size(), static_cast<uint8_t>(0));
    std::atomic<bool> has_parallel_error{false};
    std::string parallel_error_message;

    int parallel_total_queries   = 0;
    int parallel_db_only         = 0;
    int parallel_model_used      = 0;
    int parallel_model_queries   = 0;
    double parallel_sum_db_de    = 0.0;
    double parallel_sum_model_de = 0.0;

#ifdef _OPENMP
#    pragma omp parallel for schedule(guided)                                                      \
        reduction(+ : parallel_total_queries, parallel_db_only, parallel_model_used,               \
                      parallel_model_queries, parallel_sum_db_de, parallel_sum_model_de)
#endif
    for (int i = 0; i < static_cast<int>(unique_colors.size()); ++i) {
        if (has_parallel_error.load(std::memory_order_relaxed)) { continue; }

        try {
            const Rgb& color     = unique_colors[static_cast<std::size_t>(i)];
            const Lab target_lab = Lab::FromRgb(color);
            const cv::Vec3f target_color =
                use_lab ? cv::Vec3f(target_lab.l(), target_lab.a(), target_lab.b())
                        : cv::Vec3f(color.r(), color.g(), color.b());
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

            unique_matches[static_cast<std::size_t>(i)] =
                CachedMatch{decision.selected.recipe, decision.selected.mapped_lab,
                            decision.selected.from_model ? decision.model_de : decision.db_de,
                            decision.selected.from_model};
            unique_match_ready[static_cast<std::size_t>(i)] = static_cast<uint8_t>(1);
        } catch (const std::exception& e) {
#ifdef _OPENMP
#    pragma omp critical(vector_match_error)
#endif
            {
                if (!has_parallel_error.load(std::memory_order_relaxed)) {
                    parallel_error_message = e.what();
                    has_parallel_error.store(true, std::memory_order_relaxed);
                }
            }
        } catch (...) {
#ifdef _OPENMP
#    pragma omp critical(vector_match_error)
#endif
            {
                if (!has_parallel_error.load(std::memory_order_relaxed)) {
                    parallel_error_message = "Unknown error in parallel vector matching";
                    has_parallel_error.store(true, std::memory_order_relaxed);
                }
            }
        }
    }

    if (has_parallel_error.load(std::memory_order_relaxed)) {
        throw MatchError(parallel_error_message.empty() ? "Parallel vector matching failed"
                                                        : parallel_error_message);
    }

    stat_total_queries = parallel_total_queries;
    stat_db_only       = parallel_db_only;
    stat_model_used    = parallel_model_used;
    stat_model_queries = parallel_model_queries;
    stat_sum_db_de     = parallel_sum_db_de;
    stat_sum_model_de  = parallel_sum_model_de;

    map.entries.reserve(solid_shape_count);
    for (std::size_t i = 0; i < result.shapes.size(); ++i) {
        const int unique_idx = shape_to_unique_idx[i];
        if (unique_idx < 0) { continue; }

        const std::size_t idx = static_cast<std::size_t>(unique_idx);
        if (idx >= unique_matches.size() || unique_match_ready[idx] == static_cast<uint8_t>(0)) {
            throw MatchError("VectorRecipeMap::Match cache entry is missing");
        }
        const CachedMatch& cache = unique_matches[idx];

        ShapeEntry entry;
        entry.shape_idx     = static_cast<int>(i);
        entry.recipe        = cache.recipe;
        entry.matched_color = cache.matched_lab;
        entry.from_model    = cache.from_model;
        map.entries.push_back(std::move(entry));
    }

    if (out_stats) {
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
    }

    spdlog::debug(
        "VectorRecipeMap::Match: {} shapes, {} unique colors, db_only={}, model_fallback={}, "
        "avg_db_de={:.2f}, avg_model_de={:.2f}",
        map.entries.size(), unique_colors.size(), stat_db_only, stat_model_used,
        stat_total_queries > 0 ? stat_sum_db_de / static_cast<double>(stat_total_queries) : 0.0,
        stat_model_queries > 0 ? stat_sum_model_de / static_cast<double>(stat_model_queries) : 0.0);

    return map;
}

void VectorRecipeMap::ReplaceRecipeForEntries(const std::vector<int>& entry_indices,
                                              const std::vector<uint8_t>& new_recipe,
                                              const Lab& new_mapped_color, bool new_from_model) {
    if (static_cast<int>(new_recipe.size()) != color_layers) {
        throw InputError("new_recipe size does not match color_layers");
    }

    for (int idx : entry_indices) {
        if (idx < 0 || static_cast<std::size_t>(idx) >= entries.size()) { continue; }
        auto& entry         = entries[static_cast<std::size_t>(idx)];
        entry.recipe        = new_recipe;
        entry.matched_color = new_mapped_color;
        entry.from_model    = new_from_model;
    }
}

void VectorRecipeMap::UpgradeColorLayers(int new_color_layers, uint8_t pad_channel) {
    if (new_color_layers <= color_layers) return;
    const int delta = new_color_layers - color_layers;
    for (auto& entry : entries) {
        if (layer_order == LayerOrder::Top2Bottom) {
            entry.recipe.insert(entry.recipe.end(), delta, pad_channel);
        } else {
            entry.recipe.insert(entry.recipe.begin(), delta, pad_channel);
        }
    }
    color_layers = new_color_layers;
}

} // namespace ChromaPrint3D
