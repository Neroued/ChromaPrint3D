#include "chromaprint3d/logging.h"
#include "chromaprint3d/vectorize_eval.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace ChromaPrint3D;

namespace {

struct Options {
    std::string image_path;
    std::string manifest_path;
    std::string svg_dir;
    std::string json_path;
    std::string baseline_dir;
    std::string history_path;
    std::string category_filter;
    std::string note;
    bool set_baseline     = false;
    std::string log_level = "info";

    PartialVectorizerConfig vec_overrides;
};

void PrintUsage(const char* exe) {
    std::printf("Usage: %s --image input.png   [options]   (single-image mode)\n"
                "       %s --manifest m.json   [options]   (batch / benchmark mode)\n"
                "\nEvaluation options:\n"
                "  --svg-dir DIR             SVG output directory\n"
                "  --json FILE               Write metrics/report JSON to file\n"
                "  --baseline-dir DIR        Baseline directory for regression comparison\n"
                "  --set-baseline            Save current results as new baseline\n"
                "  --history FILE            CSV history file to append run summary\n"
                "  --category CAT            Only run images of this category (batch mode)\n"
                "  --note TEXT               Annotation stored in history / report\n"
                "\nVectorizer parameter overrides (same as raster_to_svg):\n"
                "  --colors N                Number of quantization colors\n"
                "  --min-region N            Min region area in pixels\n"
                "  --curve-fit-error F       Curve fitting error threshold\n"
                "  --corner-angle F          Corner angle threshold in degrees\n"
                "  --smoothing-spatial F     Mean Shift spatial radius\n"
                "  --smoothing-color F       Mean Shift color radius\n"
                "  --upscale-short-edge N    Auto-upscale short-edge threshold\n"
                "  --max-working-pixels N    Auto-downscale pixel count threshold\n"
                "  --slic-region-size N      SLIC region size\n"
                "  --slic-compactness F      SLIC compactness\n"
                "  --contour-simplify F      Contour simplification strength\n"
                "  --edge-sensitivity F      Edge-aware SLIC sensitivity [0,1]\n"
                "  --refine-passes N         Boundary label refinement iterations\n"
                "  --max-merge-color-dist F  Max LAB dE^2 for small-region merging\n"
                "  --disable-subpixel-refine Disable sub-pixel boundary refinement\n"
                "  --disable-coverage-fix    Disable coverage patching\n"
                "  --min-coverage-ratio F    Coverage fix trigger ratio\n"
                "  --smoothness F            Contour smoothness [0,1]\n"
                "  --detail-level F          Unified detail control [0,1]\n"
                "  --merge-tolerance F       Near-linear segment merge tolerance\n"
                "  --enable-antialias        Enable AA mixed-edge detection\n"
                "  --aa-tolerance F          AA blend detection LAB tolerance\n"
                "  --log-level LEVEL         trace/debug/info/warn/error/off (default info)\n",
                exe, exe);
}

bool ParseInt(const char* s, int& out) {
    if (!s) return false;
    try {
        size_t idx = 0;
        out        = std::stoi(s, &idx, 10);
        return idx == std::string(s).size();
    } catch (...) { return false; }
}

bool ParseFloat(const char* s, float& out) {
    if (!s) return false;
    try {
        size_t idx = 0;
        out        = std::stof(s, &idx);
        return idx == std::string(s).size();
    } catch (...) { return false; }
}

#define TRY_INT_OPT(FLAG, FIELD)                                                                   \
    if (arg == FLAG && i + 1 < argc) {                                                             \
        int v;                                                                                     \
        if (!ParseInt(argv[++i], v)) {                                                             \
            std::fprintf(stderr, "Invalid %s\n", FLAG);                                            \
            return false;                                                                          \
        }                                                                                          \
        opt.vec_overrides.FIELD = v;                                                               \
        continue;                                                                                  \
    }

#define TRY_FLOAT_OPT(FLAG, FIELD)                                                                 \
    if (arg == FLAG && i + 1 < argc) {                                                             \
        float v;                                                                                   \
        if (!ParseFloat(argv[++i], v)) {                                                           \
            std::fprintf(stderr, "Invalid %s\n", FLAG);                                            \
            return false;                                                                          \
        }                                                                                          \
        opt.vec_overrides.FIELD = v;                                                               \
        continue;                                                                                  \
    }

bool ParseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return false;
        }
        if (arg == "--image" && i + 1 < argc) {
            opt.image_path = argv[++i];
            continue;
        }
        if (arg == "--manifest" && i + 1 < argc) {
            opt.manifest_path = argv[++i];
            continue;
        }
        if (arg == "--svg-dir" && i + 1 < argc) {
            opt.svg_dir = argv[++i];
            continue;
        }
        if (arg == "--json" && i + 1 < argc) {
            opt.json_path = argv[++i];
            continue;
        }
        if (arg == "--baseline-dir" && i + 1 < argc) {
            opt.baseline_dir = argv[++i];
            continue;
        }
        if (arg == "--set-baseline") {
            opt.set_baseline = true;
            continue;
        }
        if (arg == "--history" && i + 1 < argc) {
            opt.history_path = argv[++i];
            continue;
        }
        if (arg == "--category" && i + 1 < argc) {
            opt.category_filter = argv[++i];
            continue;
        }
        if (arg == "--note" && i + 1 < argc) {
            opt.note = argv[++i];
            continue;
        }
        if (arg == "--log-level" && i + 1 < argc) {
            opt.log_level = argv[++i];
            continue;
        }

        TRY_INT_OPT("--colors", num_colors)
        TRY_INT_OPT("--min-region", min_region_area)
        TRY_FLOAT_OPT("--curve-fit-error", curve_fit_error)
        TRY_FLOAT_OPT("--corner-angle", corner_angle_threshold)
        TRY_FLOAT_OPT("--smoothing-spatial", smoothing_spatial)
        TRY_FLOAT_OPT("--smoothing-color", smoothing_color)
        TRY_INT_OPT("--upscale-short-edge", upscale_short_edge)
        TRY_INT_OPT("--max-working-pixels", max_working_pixels)
        TRY_INT_OPT("--slic-region-size", slic_region_size)
        TRY_FLOAT_OPT("--slic-compactness", slic_compactness)
        TRY_FLOAT_OPT("--contour-simplify", contour_simplify)
        TRY_FLOAT_OPT("--edge-sensitivity", edge_sensitivity)
        TRY_INT_OPT("--refine-passes", refine_passes)
        TRY_FLOAT_OPT("--max-merge-color-dist", max_merge_color_dist)
        TRY_FLOAT_OPT("--min-coverage-ratio", min_coverage_ratio)
        TRY_FLOAT_OPT("--smoothness", smoothness)
        TRY_FLOAT_OPT("--detail-level", detail_level)
        TRY_FLOAT_OPT("--merge-tolerance", merge_segment_tolerance)
        TRY_FLOAT_OPT("--aa-tolerance", aa_tolerance)

        if (arg == "--disable-subpixel-refine") {
            opt.vec_overrides.enable_subpixel_refine = false;
            continue;
        }
        if (arg == "--disable-coverage-fix") {
            opt.vec_overrides.enable_coverage_fix = false;
            continue;
        }
        if (arg == "--enable-antialias") {
            opt.vec_overrides.enable_antialias_detect = true;
            continue;
        }

        std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
        PrintUsage(argv[0]);
        return false;
    }
    return true;
}

#undef TRY_INT_OPT
#undef TRY_FLOAT_OPT

} // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!ParseArgs(argc, argv, opt)) return 1;

    InitLogging(ParseLogLevel(opt.log_level));

    if (opt.image_path.empty() && opt.manifest_path.empty()) {
        std::fprintf(stderr, "Error: --image or --manifest is required\n");
        PrintUsage(argv[0]);
        return 1;
    }

    try {
        EvalConfig eval_cfg;
        eval_cfg.vectorizer_overrides = opt.vec_overrides;
        if (!opt.svg_dir.empty()) eval_cfg.svg_output_dir = opt.svg_dir;

        // ── Single image mode ────────────────────────────────────────────
        if (!opt.image_path.empty()) {
            if (eval_cfg.svg_output_dir.empty()) eval_cfg.svg_output_dir = ".";

            auto metrics = EvaluateImage(opt.image_path, eval_cfg);
            double score = ComputeScore(metrics);

            std::string json_str = metrics.ToJson();
            if (!opt.json_path.empty()) {
                std::ofstream f(opt.json_path);
                if (!f) throw std::runtime_error("Cannot write: " + opt.json_path);
                f << json_str;
                spdlog::info("Metrics written to {}", opt.json_path);
            } else {
                std::printf("%s\n", json_str.c_str());
            }

            spdlog::info("Score: {:.1f}", score);
            return 0;
        }

        // ── Batch / benchmark mode ───────────────────────────────────────
        auto manifest = Manifest::LoadFromJson(opt.manifest_path);

        // Category filter
        if (!opt.category_filter.empty()) {
            auto& imgs = manifest.images;
            imgs.erase(std::remove_if(
                           imgs.begin(), imgs.end(),
                           [&](const ImageEntry& e) { return e.category != opt.category_filter; }),
                       imgs.end());
            spdlog::info("Filtered to {} images in category '{}'", imgs.size(),
                         opt.category_filter);
        }

        if (manifest.images.empty()) {
            spdlog::warn("No images to evaluate");
            return 0;
        }

        // Auto SVG dir
        std::string run_id = eval::MakeRunId();
        if (eval_cfg.svg_output_dir.empty()) {
            auto manifest_dir       = std::filesystem::path(opt.manifest_path).parent_path();
            eval_cfg.svg_output_dir = (manifest_dir / "results" / run_id).string();
        }

        auto results = EvaluateBatch(manifest, eval_cfg);

        // Score
        for (auto& r : results) r.score = ComputeScore(r.metrics);

        // Baseline comparison
        std::vector<BaselineVerdict> verdicts;
        if (!opt.baseline_dir.empty()) { verdicts = CompareBaseline(opt.baseline_dir, results); }

        // Terminal report
        PrintReport(results, verdicts);

        // Save baseline
        if (opt.set_baseline) {
            std::string bl_dir =
                opt.baseline_dir.empty() ? "test_data/baselines/current" : opt.baseline_dir;
            SaveBaseline(bl_dir, results);
        }

        // Append history
        if (!opt.history_path.empty()) {
            AppendHistory(opt.history_path, run_id, results, opt.note);
        }

        // JSON report
        BenchmarkReport report;
        report.run_id     = run_id;
        report.timestamp  = eval::CurrentTimestamp();
        report.git_commit = eval::GitShortHash();
        report.note       = opt.note;
        report.results    = results;
        report.verdicts   = verdicts;

        // Summary
        auto& s        = report.summary;
        s.total_images = static_cast<int>(results.size());
        for (auto& r : results) {
            s.score_avg += r.score;
            s.coverage_avg += r.metrics.coverage;
            s.delta_e_avg += r.metrics.delta_e_mean;
            s.ssim_avg += r.metrics.ssim;
            if (!r.expectation_failures.empty()) s.failures++;
        }
        if (s.total_images > 0) {
            s.score_avg /= s.total_images;
            s.coverage_avg /= s.total_images;
            s.delta_e_avg /= s.total_images;
            s.ssim_avg /= s.total_images;
        }
        for (auto& v : verdicts) {
            if (v.status == BaselineVerdict::REGRESSED) s.regressions++;
            if (v.status == BaselineVerdict::IMPROVED) s.improvements++;
        }

        if (!opt.json_path.empty()) {
            std::ofstream f(opt.json_path);
            if (!f) throw std::runtime_error("Cannot write: " + opt.json_path);
            f << report.ToJson();
            spdlog::info("Report written to {}", opt.json_path);
        }

        if (s.regressions > 0) {
            spdlog::warn("{} regressions detected!", s.regressions);
            return 1;
        }

    } catch (const std::exception& e) {
        spdlog::error("Failed: {}", e.what());
        return 1;
    }

    return 0;
}
