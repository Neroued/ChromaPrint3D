#include "chromaprint3d/vectorize_eval.h"

#include "benchmark.h"
#include "edge_metrics.h"
#include "path_metrics.h"
#include "pixel_metrics.h"
#include "svg_rasterizer.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#include "chromaprint3d/image_io.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace ChromaPrint3D {

// ── PartialVectorizerConfig ──────────────────────────────────────────────────

VectorizerConfig PartialVectorizerConfig::MergeInto(const VectorizerConfig& base) const {
    VectorizerConfig out = base;
#define MERGE_FIELD(F)                                                                             \
    if (F) out.F = *F
    MERGE_FIELD(num_colors);
    MERGE_FIELD(min_region_area);
    MERGE_FIELD(curve_fit_error);
    MERGE_FIELD(corner_angle_threshold);
    MERGE_FIELD(smoothing_spatial);
    MERGE_FIELD(smoothing_color);
    MERGE_FIELD(upscale_short_edge);
    MERGE_FIELD(max_working_pixels);
    MERGE_FIELD(slic_region_size);
    MERGE_FIELD(slic_compactness);
    MERGE_FIELD(edge_sensitivity);
    MERGE_FIELD(refine_passes);
    MERGE_FIELD(max_merge_color_dist);
    MERGE_FIELD(enable_subpixel_refine);
    MERGE_FIELD(subpixel_max_displacement);
    MERGE_FIELD(thin_line_max_radius);
    MERGE_FIELD(svg_enable_stroke);
    MERGE_FIELD(svg_stroke_width);
    MERGE_FIELD(min_contour_area);
    MERGE_FIELD(min_hole_area);
    MERGE_FIELD(contour_simplify);
    MERGE_FIELD(enable_coverage_fix);
    MERGE_FIELD(min_coverage_ratio);
    MERGE_FIELD(smoothness);
    MERGE_FIELD(detail_level);
    MERGE_FIELD(merge_segment_tolerance);
    MERGE_FIELD(enable_antialias_detect);
    MERGE_FIELD(aa_tolerance);
#undef MERGE_FIELD
    return out;
}

// ── VectorizeMetrics JSON ────────────────────────────────────────────────────

std::string VectorizeMetrics::ToJson(int indent) const {
    std::ostringstream ss;
    std::string ind(indent, ' ');
    ss << "{\n";
    ss << std::fixed;
    ss << ind << "\"psnr\": " << std::setprecision(2) << psnr << ",\n";
    ss << ind << "\"ssim\": " << std::setprecision(4) << ssim << ",\n";
    ss << ind << "\"coverage\": " << std::setprecision(4) << coverage << ",\n";
    ss << ind << "\"overlap\": " << std::setprecision(4) << overlap << ",\n";
    ss << ind << "\"delta_e_mean\": " << std::setprecision(2) << delta_e_mean << ",\n";
    ss << ind << "\"delta_e_p95\": " << std::setprecision(2) << delta_e_p95 << ",\n";
    ss << ind << "\"delta_e_max\": " << std::setprecision(2) << delta_e_max << ",\n";
    ss << ind << "\"border_delta_e_mean\": " << std::setprecision(2) << border_delta_e_mean
       << ",\n";
    ss << ind << "\"edge_f1\": " << std::setprecision(4) << edge_f1 << ",\n";
    ss << ind << "\"chamfer_distance\": " << std::setprecision(2) << chamfer_distance << ",\n";
    ss << ind << "\"total_shapes\": " << total_shapes << ",\n";
    ss << ind << "\"unique_colors\": " << unique_colors << ",\n";
    ss << ind << "\"mergeable_ratio\": " << std::setprecision(2) << mergeable_ratio << ",\n";
    ss << ind << "\"tiny_fragment_rate\": " << std::setprecision(4) << tiny_fragment_rate << ",\n";
    ss << ind << "\"gini_coefficient\": " << std::setprecision(4) << gini_coefficient << ",\n";
    ss << ind << "\"path_complexity_median\": " << path_complexity_median << ",\n";
    ss << ind << "\"path_complexity_p95\": " << path_complexity_p95 << ",\n";
    ss << ind << "\"circularity_p95\": " << std::setprecision(4) << circularity_p95 << ",\n";
    ss << ind << "\"sliver_count\": " << sliver_count << ",\n";
    ss << ind << "\"island_count\": " << island_count << ",\n";
    ss << ind << "\"same_color_gap_pixels\": " << same_color_gap_pixels << ",\n";
    ss << ind << "\"color_compression\": " << std::setprecision(4) << color_compression << ",\n";
    ss << ind << "\"vectorize_time_ms\": " << std::setprecision(1) << vectorize_time_ms << ",\n";
    ss << ind << "\"eval_time_ms\": " << std::setprecision(1) << eval_time_ms << ",\n";
    ss << ind << "\"width\": " << width << ",\n";
    ss << ind << "\"height\": " << height << "\n";
    ss << "}";
    return ss.str();
}

// ── Manifest loading ─────────────────────────────────────────────────────────

namespace {

// Minimal JSON parser sufficient for manifest format.
// Supports nested objects/arrays, string/number/bool values.
struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray  = std::vector<JsonValue>;

struct JsonValue {
    enum Type { NUL, STRING, NUMBER, BOOL, OBJECT, ARRAY };

    Type type = NUL;
    std::string str_val;
    double num_val = 0;
    bool bool_val  = false;
    JsonObject obj;
    JsonArray arr;

    double AsNumber(double def = 0) const { return type == NUMBER ? num_val : def; }

    int AsInt(int def = 0) const { return type == NUMBER ? static_cast<int>(num_val) : def; }

    bool AsBool(bool def = false) const { return type == BOOL ? bool_val : def; }

    const std::string& AsString() const { return str_val; }

    bool Has(const std::string& key) const { return type == OBJECT && obj.count(key) > 0; }

    const JsonValue& operator[](const std::string& key) const {
        static JsonValue nil;
        if (type != OBJECT) return nil;
        auto it = obj.find(key);
        return it != obj.end() ? it->second : nil;
    }
};

// Forward declarations for recursive parser
size_t SkipWs(const std::string& s, size_t i);
JsonValue ParseValue(const std::string& s, size_t& i);

size_t SkipWs(const std::string& s, size_t i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    return i;
}

std::string ParseString(const std::string& s, size_t& i) {
    i = SkipWs(s, i);
    if (i >= s.size() || s[i] != '"') return {};
    ++i;
    std::string result;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
            ++i;
            switch (s[i]) {
            case '"':
            case '\\':
            case '/':
                result += s[i];
                break;
            case 'n':
                result += '\n';
                break;
            case 't':
                result += '\t';
                break;
            default:
                result += s[i];
                break;
            }
        } else {
            result += s[i];
        }
        ++i;
    }
    if (i < s.size()) ++i; // skip closing "
    return result;
}

JsonValue ParseValue(const std::string& s, size_t& i) {
    i = SkipWs(s, i);
    JsonValue v;
    if (i >= s.size()) return v;

    if (s[i] == '"') {
        v.type    = JsonValue::STRING;
        v.str_val = ParseString(s, i);
    } else if (s[i] == '{') {
        v.type = JsonValue::OBJECT;
        ++i;
        i = SkipWs(s, i);
        while (i < s.size() && s[i] != '}') {
            std::string key = ParseString(s, i);
            i               = SkipWs(s, i);
            if (i < s.size() && s[i] == ':') ++i;
            v.obj[key] = ParseValue(s, i);
            i          = SkipWs(s, i);
            if (i < s.size() && s[i] == ',') ++i;
            i = SkipWs(s, i);
        }
        if (i < s.size()) ++i;
    } else if (s[i] == '[') {
        v.type = JsonValue::ARRAY;
        ++i;
        i = SkipWs(s, i);
        while (i < s.size() && s[i] != ']') {
            v.arr.push_back(ParseValue(s, i));
            i = SkipWs(s, i);
            if (i < s.size() && s[i] == ',') ++i;
            i = SkipWs(s, i);
        }
        if (i < s.size()) ++i;
    } else if (s[i] == 't' || s[i] == 'f') {
        v.type = JsonValue::BOOL;
        if (s.substr(i, 4) == "true") {
            v.bool_val = true;
            i += 4;
        } else if (s.substr(i, 5) == "false") {
            v.bool_val = false;
            i += 5;
        }
    } else if (s[i] == 'n') {
        i += 4; // null
    } else {
        // Number
        v.type       = JsonValue::NUMBER;
        size_t start = i;
        if (s[i] == '-') ++i;
        while (i < s.size() && (std::isdigit(s[i]) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' ||
                                s[i] == '+' || s[i] == '-'))
            ++i;
        v.num_val = std::stod(s.substr(start, i - start));
    }
    return v;
}

PartialVectorizerConfig ParsePartialConfig(const JsonValue& v) {
    PartialVectorizerConfig p;
    if (v.type != JsonValue::OBJECT) return p;

#define OPT_INT(F)                                                                                 \
    if (v.Has(#F)) p.F = v[#F].AsInt()
#define OPT_FLOAT(F)                                                                               \
    if (v.Has(#F)) p.F = static_cast<float>(v[#F].AsNumber())
#define OPT_BOOL(F)                                                                                \
    if (v.Has(#F)) p.F = v[#F].AsBool()

    OPT_INT(num_colors);
    OPT_INT(min_region_area);
    OPT_FLOAT(curve_fit_error);
    OPT_FLOAT(corner_angle_threshold);
    OPT_FLOAT(smoothing_spatial);
    OPT_FLOAT(smoothing_color);
    OPT_INT(upscale_short_edge);
    OPT_INT(max_working_pixels);
    OPT_INT(slic_region_size);
    OPT_FLOAT(slic_compactness);
    OPT_FLOAT(edge_sensitivity);
    OPT_INT(refine_passes);
    OPT_FLOAT(max_merge_color_dist);
    OPT_BOOL(enable_subpixel_refine);
    OPT_FLOAT(subpixel_max_displacement);
    OPT_FLOAT(thin_line_max_radius);
    OPT_BOOL(svg_enable_stroke);
    OPT_FLOAT(svg_stroke_width);
    OPT_FLOAT(min_contour_area);
    OPT_FLOAT(min_hole_area);
    OPT_FLOAT(contour_simplify);
    OPT_BOOL(enable_coverage_fix);
    OPT_FLOAT(min_coverage_ratio);
    OPT_FLOAT(smoothness);
    OPT_FLOAT(detail_level);
    OPT_FLOAT(merge_segment_tolerance);
    OPT_BOOL(enable_antialias_detect);
    OPT_FLOAT(aa_tolerance);

#undef OPT_INT
#undef OPT_FLOAT
#undef OPT_BOOL
    return p;
}

Expectations ParseExpectations(const JsonValue& v) {
    Expectations e;
    if (v.type != JsonValue::OBJECT) return e;
    if (v.Has("min_coverage")) e.min_coverage = v["min_coverage"].AsNumber();
    if (v.Has("max_delta_e_mean")) e.max_delta_e_mean = v["max_delta_e_mean"].AsNumber();
    if (v.Has("max_shapes")) e.max_shapes = v["max_shapes"].AsInt();
    if (v.Has("min_ssim")) e.min_ssim = v["min_ssim"].AsNumber();
    if (v.Has("min_psnr")) e.min_psnr = v["min_psnr"].AsNumber();
    if (v.Has("max_chamfer_distance"))
        e.max_chamfer_distance = v["max_chamfer_distance"].AsNumber();
    return e;
}

VectorizerConfig ResolveConfig(const EvalConfig& eval_cfg, const ImageEntry& entry) {
    VectorizerConfig cfg;
    cfg = eval_cfg.vectorizer_overrides.MergeInto(cfg);
    cfg = entry.vectorizer_overrides.MergeInto(cfg);
    return cfg;
}

} // namespace

Manifest Manifest::LoadFromJson(const std::string& path) {
    Manifest m;
    std::ifstream f(path);
    if (!f) {
        spdlog::error("Cannot open manifest: {}", path);
        return m;
    }
    std::string content((std::istreambuf_iterator<char>(f)), {});
    size_t pos = 0;
    auto root  = ParseValue(content, pos);
    if (!root.Has("images") || root["images"].type != JsonValue::ARRAY) {
        spdlog::error("Manifest must contain \"images\" array");
        return m;
    }

    auto manifest_dir = std::filesystem::path(path).parent_path();
    for (auto& item : root["images"].arr) {
        ImageEntry entry;
        entry.path = (manifest_dir / item["path"].AsString()).string();
        entry.name = item.Has("name") ? item["name"].AsString() : "";
        if (entry.name.empty()) { entry.name = std::filesystem::path(entry.path).stem().string(); }
        entry.category = item.Has("category") ? item["category"].AsString() : "";
        if (item.Has("vectorizer_overrides"))
            entry.vectorizer_overrides = ParsePartialConfig(item["vectorizer_overrides"]);
        if (item.Has("expectations")) entry.expectations = ParseExpectations(item["expectations"]);
        m.images.push_back(std::move(entry));
    }
    spdlog::info("Loaded manifest with {} images from {}", m.images.size(), path);
    return m;
}

// ── Single image evaluation ──────────────────────────────────────────────────

namespace {

ImageResult EvaluateSingleEntry(const ImageEntry& entry, const EvalConfig& config) {
    ImageResult result;
    result.name          = entry.name;
    result.category      = entry.category;
    result.original_path = entry.path;

    auto cfg = ResolveConfig(config, entry);

    // Vectorize
    auto t0         = std::chrono::high_resolution_clock::now();
    auto vec_result = Vectorize(entry.path, cfg);
    auto t1         = std::chrono::high_resolution_clock::now();
    double vec_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (vec_result.svg_content.empty()) {
        spdlog::error("Vectorization failed for '{}'", entry.path);
        return result;
    }

    // Save SVG
    std::string svg_dir = config.svg_output_dir;
    if (svg_dir.empty()) svg_dir = ".";
    std::filesystem::create_directories(svg_dir);
    result.svg_path = (std::filesystem::path(svg_dir) / (result.name + ".svg")).string();
    {
        std::ofstream f(result.svg_path);
        f << vec_result.svg_content;
    }

    // Load original with ICC color management (same pipeline as vectorizer)
    cv::Mat loaded;
    try {
        loaded = LoadImageWithIcc(entry.path);
    } catch (const std::exception& e) {
        spdlog::error("Cannot load image '{}': {}", entry.path, e.what());
        return result;
    }
    if (loaded.empty()) {
        spdlog::error("Cannot load original image: {}", entry.path);
        return result;
    }

    // Normalize bit depth to 8-bit
    if (loaded.depth() != CV_8U) {
        double scale = (loaded.depth() == CV_16U || loaded.depth() == CV_16S) ? 1.0 / 256.0 : 1.0;
        loaded.convertTo(loaded, CV_8U, scale);
    }

    cv::Mat original;
    cv::Mat alpha_mask;
    if (loaded.channels() == 4) {
        // Extract alpha mask before compositing
        cv::Mat alpha;
        cv::extractChannel(loaded, alpha, 3);
        cv::threshold(alpha, alpha_mask, 0, 255, cv::THRESH_BINARY);

        // Alpha-composite onto white background (matches vectorizer's EnsureBgr)
        original = cv::Mat(loaded.rows, loaded.cols, CV_8UC3);
        for (int r = 0; r < loaded.rows; ++r) {
            const cv::Vec4b* src_row = loaded.ptr<cv::Vec4b>(r);
            cv::Vec3b* dst_row       = original.ptr<cv::Vec3b>(r);
            for (int c = 0; c < loaded.cols; ++c) {
                int a = src_row[c][3];
                if (a <= 0) {
                    dst_row[c] = cv::Vec3b(255, 255, 255);
                } else if (a >= 255) {
                    dst_row[c] = cv::Vec3b(src_row[c][0], src_row[c][1], src_row[c][2]);
                } else {
                    int inv_a = 255 - a;
                    dst_row[c][0] =
                        static_cast<uint8_t>((src_row[c][0] * a + 255 * inv_a + 127) / 255);
                    dst_row[c][1] =
                        static_cast<uint8_t>((src_row[c][1] * a + 255 * inv_a + 127) / 255);
                    dst_row[c][2] =
                        static_cast<uint8_t>((src_row[c][2] * a + 255 * inv_a + 127) / 255);
                }
            }
        }
    } else if (loaded.channels() == 1) {
        cv::cvtColor(loaded, original, cv::COLOR_GRAY2BGR);
    } else {
        original = loaded;
    }

    int w = original.cols, h = original.rows;

    auto t2 = std::chrono::high_resolution_clock::now();

    // Rasterize SVG
    auto rast = eval::RasterizeSvg(vec_result.svg_content, w, h);

    // Save rasterized image and original for visual comparison
    if (!svg_dir.empty()) {
        auto rast_path = std::filesystem::path(svg_dir) / (result.name + "_rasterized.png");
        cv::imwrite(rast_path.string(), rast.bgr);

        auto orig_path = std::filesystem::path(svg_dir) / (result.name + "_original.png");
        cv::imwrite(orig_path.string(), original);
    }

    // Compute metrics — pass alpha_mask so transparent pixels are excluded
    auto px =
        eval::ComputePixelMetrics(original, rast.bgr, rast.coverage, rast.shape_count, alpha_mask);
    auto em = eval::ComputeEdgeMetrics(original, rast.bgr, config.edge_tolerance_px, alpha_mask);
    auto pm = eval::ComputePathMetrics(vec_result.svg_content, w, h, config.tiny_area_threshold,
                                       config.sliver_threshold);

    auto t3        = std::chrono::high_resolution_clock::now();
    double eval_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Assemble
    auto& m                  = result.metrics;
    m.psnr                   = px.psnr;
    m.ssim                   = px.ssim;
    m.coverage               = px.coverage;
    m.overlap                = px.overlap;
    m.delta_e_mean           = px.delta_e_mean;
    m.delta_e_p95            = px.delta_e_p95;
    m.delta_e_max            = px.delta_e_max;
    m.border_delta_e_mean    = px.border_delta_e_mean;
    m.edge_f1                = em.edge_f1;
    m.chamfer_distance       = em.chamfer_distance;
    m.total_shapes           = pm.total_shapes;
    m.unique_colors          = pm.unique_colors;
    m.mergeable_ratio        = pm.mergeable_ratio;
    m.tiny_fragment_rate     = pm.tiny_fragment_rate;
    m.gini_coefficient       = pm.gini_coefficient;
    m.path_complexity_median = pm.path_complexity_median;
    m.path_complexity_p95    = pm.path_complexity_p95;
    m.circularity_p95        = pm.circularity_p95;
    m.sliver_count           = pm.sliver_count;
    m.island_count           = pm.island_count;
    m.same_color_gap_pixels  = pm.same_color_gap_pixels;
    m.color_compression      = pm.color_compression;
    m.vectorize_time_ms      = vec_ms;
    m.eval_time_ms           = eval_ms;
    m.width                  = w;
    m.height                 = h;

    result.score                = ComputeScore(m);
    result.expectation_failures = entry.expectations.Check(m);

    return result;
}

} // namespace

VectorizeMetrics EvaluateImage(const std::string& image_path, const EvalConfig& config) {
    ImageEntry entry;
    entry.path  = image_path;
    entry.name  = std::filesystem::path(image_path).stem().string();
    auto result = EvaluateSingleEntry(entry, config);
    return result.metrics;
}

std::vector<ImageResult> EvaluateBatch(const Manifest& manifest, const EvalConfig& config) {
    std::vector<ImageResult> results(manifest.images.size());

#if defined(_OPENMP) && _OPENMP >= 200805
    int prev_levels = omp_get_max_active_levels();
    omp_set_max_active_levels(1);
#endif

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(manifest.images.size()); ++i) {
        results[i] = EvaluateSingleEntry(manifest.images[i], config);
    }

#if defined(_OPENMP) && _OPENMP >= 200805
    omp_set_max_active_levels(prev_levels);
#endif

    return results;
}

} // namespace ChromaPrint3D
