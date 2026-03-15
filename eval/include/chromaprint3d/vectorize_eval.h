#pragma once

/// \file vectorize_eval.h
/// \brief Public API for vectorization quality evaluation pipeline.

#include "chromaprint3d/vectorizer.h"

#include <optional>
#include <string>
#include <vector>

namespace ChromaPrint3D {

// ── Field-level vectorizer config overlay ────────────────────────────────────

/// Each field is optional; only fields that are set will override the base
/// VectorizerConfig when MergeInto() is called.
struct PartialVectorizerConfig {
    std::optional<int> num_colors;
    std::optional<int> min_region_area;
    std::optional<float> curve_fit_error;
    std::optional<float> corner_angle_threshold;
    std::optional<float> smoothing_spatial;
    std::optional<float> smoothing_color;
    std::optional<int> upscale_short_edge;
    std::optional<int> max_working_pixels;
    std::optional<int> slic_region_size;
    std::optional<float> slic_compactness;
    std::optional<float> edge_sensitivity;
    std::optional<int> refine_passes;
    std::optional<float> max_merge_color_dist;
    std::optional<bool> enable_subpixel_refine;
    std::optional<float> subpixel_max_displacement;
    std::optional<float> thin_line_max_radius;
    std::optional<bool> svg_enable_stroke;
    std::optional<float> svg_stroke_width;
    std::optional<float> min_contour_area;
    std::optional<float> min_hole_area;
    std::optional<float> contour_simplify;
    std::optional<bool> enable_coverage_fix;
    std::optional<float> min_coverage_ratio;
    std::optional<float> smoothness;
    std::optional<float> detail_level;
    std::optional<float> merge_segment_tolerance;
    std::optional<bool> enable_antialias_detect;
    std::optional<float> aa_tolerance;

    /// Apply set fields onto \p base, returning the merged config.
    VectorizerConfig MergeInto(const VectorizerConfig& base) const;
};

// ── Expectations ─────────────────────────────────────────────────────────────

/// Per-image quality thresholds. Check() returns failure descriptions for any
/// threshold that is violated.
struct Expectations {
    std::optional<double> min_coverage;
    std::optional<double> max_delta_e_mean;
    std::optional<int> max_shapes;
    std::optional<double> min_ssim;
    std::optional<double> min_psnr;
    std::optional<double> max_chamfer_distance;

    std::vector<std::string> Check(const struct VectorizeMetrics& m) const;
};

// ── Metrics ──────────────────────────────────────────────────────────────────

struct VectorizeMetrics {
    // Pixel fidelity
    double psnr                = 0;
    double ssim                = 0;
    double coverage            = 0;
    double overlap             = 0;
    double delta_e_mean        = 0;
    double delta_e_p95         = 0;
    double delta_e_max         = 0;
    double border_delta_e_mean = 0;

    // Edge fidelity
    double edge_f1          = 0;
    double chamfer_distance = 0;

    // Path structure
    int total_shapes           = 0;
    int unique_colors          = 0;
    double mergeable_ratio     = 1.0;
    double tiny_fragment_rate  = 0;
    double gini_coefficient    = 0;
    int path_complexity_median = 0;
    int path_complexity_p95    = 0;
    double circularity_p95     = 0;
    int sliver_count           = 0;
    int island_count           = 0;
    int same_color_gap_pixels  = 0;
    double color_compression   = 0;

    // Timing
    double vectorize_time_ms = 0;
    double eval_time_ms      = 0;

    int width  = 0;
    int height = 0;

    std::string ToJson(int indent = 2) const;
};

// ── Evaluation config ────────────────────────────────────────────────────────

struct EvalConfig {
    std::string svg_output_dir;
    double tiny_area_threshold = 50.0;
    double sliver_threshold    = 0.02;
    int edge_tolerance_px      = 2;
    PartialVectorizerConfig vectorizer_overrides;
};

// ── Manifest ─────────────────────────────────────────────────────────────────

struct ImageEntry {
    std::string path;
    std::string name;
    std::string category;
    PartialVectorizerConfig vectorizer_overrides;
    Expectations expectations;
};

struct Manifest {
    std::vector<ImageEntry> images;
    static Manifest LoadFromJson(const std::string& path);
};

// ── Results ──────────────────────────────────────────────────────────────────

struct ImageResult {
    std::string name;
    std::string category;
    std::string svg_path;
    std::string original_path;
    VectorizeMetrics metrics;
    double score = 0;
    std::vector<std::string> expectation_failures;
};

// ── Core evaluation entry points ─────────────────────────────────────────────

VectorizeMetrics EvaluateImage(const std::string& image_path, const EvalConfig& config = {});

std::vector<ImageResult> EvaluateBatch(const Manifest& manifest, const EvalConfig& config = {});

// ── Scoring ──────────────────────────────────────────────────────────────────

struct ScoreWeights {
    double fidelity        = 40;
    double structure       = 30;
    double edge            = 15;
    double efficiency      = 15;
    double delta_e_ceiling = 40;
};

double ComputeScore(const VectorizeMetrics& m, const ScoreWeights& w = {});

// ── Baseline management ──────────────────────────────────────────────────────

struct BaselineVerdict {
    std::string name;

    enum Status { OK, IMPROVED, REGRESSED, NEW_IMAGE } status = OK;

    double baseline_score = 0;
    double current_score  = 0;
};

std::vector<BaselineVerdict> CompareBaseline(const std::string& baseline_dir,
                                             const std::vector<ImageResult>& results,
                                             double threshold = 1.0);

void SaveBaseline(const std::string& baseline_dir, const std::vector<ImageResult>& results);

// ── History ──────────────────────────────────────────────────────────────────

void AppendHistory(const std::string& history_path, const std::string& run_id,
                   const std::vector<ImageResult>& results, const std::string& note = "");

// ── Reporting ────────────────────────────────────────────────────────────────

void PrintReport(const std::vector<ImageResult>& results,
                 const std::vector<BaselineVerdict>& verdicts);

// ── Full benchmark JSON output ───────────────────────────────────────────────

struct BenchmarkReport {
    std::string run_id;
    std::string timestamp;
    std::string git_commit;
    std::string note;

    struct Summary {
        int total_images    = 0;
        double score_avg    = 0;
        double coverage_avg = 0;
        double delta_e_avg  = 0;
        double ssim_avg     = 0;
        int regressions     = 0;
        int improvements    = 0;
        int failures        = 0;
    } summary;

    std::vector<ImageResult> results;
    std::vector<BaselineVerdict> verdicts;

    std::string ToJson(int indent = 2) const;
};

namespace eval {

std::string MakeRunId();
std::string GitShortHash();
std::string CurrentTimestamp();

} // namespace eval

} // namespace ChromaPrint3D
