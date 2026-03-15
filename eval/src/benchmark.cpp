#include "benchmark.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>

namespace ChromaPrint3D {

// ── Scoring ──────────────────────────────────────────────────────────────────

double ComputeScore(const VectorizeMetrics& m, const ScoreWeights& w) {
    double fidelity   = std::max(0.0, 1.0 - m.delta_e_mean / w.delta_e_ceiling);
    double structure  = std::clamp(m.ssim, 0.0, 1.0);
    double edge       = std::clamp(m.edge_f1, 0.0, 1.0);
    double efficiency = std::clamp((1.0 - m.tiny_fragment_rate) * m.coverage, 0.0, 1.0);

    return w.fidelity * fidelity + w.structure * structure + w.edge * edge +
           w.efficiency * efficiency;
}

// ── Expectations ─────────────────────────────────────────────────────────────

std::vector<std::string> Expectations::Check(const VectorizeMetrics& m) const {
    std::vector<std::string> fails;
    auto fmt = [](double v, int prec = 4) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(prec) << v;
        return ss.str();
    };
    if (min_coverage && m.coverage < *min_coverage)
        fails.push_back("coverage " + fmt(m.coverage) + " < min " + fmt(*min_coverage));
    if (max_delta_e_mean && m.delta_e_mean > *max_delta_e_mean)
        fails.push_back("delta_e_mean " + fmt(m.delta_e_mean, 2) + " > max " +
                        fmt(*max_delta_e_mean, 2));
    if (max_shapes && m.total_shapes > *max_shapes)
        fails.push_back("shapes " + std::to_string(m.total_shapes) + " > max " +
                        std::to_string(*max_shapes));
    if (min_ssim && m.ssim < *min_ssim)
        fails.push_back("ssim " + fmt(m.ssim) + " < min " + fmt(*min_ssim));
    if (min_psnr && m.psnr < *min_psnr)
        fails.push_back("psnr " + fmt(m.psnr, 1) + " < min " + fmt(*min_psnr, 1));
    if (max_chamfer_distance && m.chamfer_distance > *max_chamfer_distance)
        fails.push_back("chamfer " + fmt(m.chamfer_distance, 2) + " > max " +
                        fmt(*max_chamfer_distance, 2));
    return fails;
}

// ── Baseline ─────────────────────────────────────────────────────────────────

namespace {

std::string CsvEscape(const std::string& s) {
    bool needs_quoting = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quoting = true;
            break;
        }
    }
    if (!needs_quoting) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"')
            out += "\"\"";
        else
            out += c;
    }
    out += '"';
    return out;
}

std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out;
}

std::string ReadFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f), {}};
}

double ExtractScore(const std::string& json) {
    // Minimal JSON number extraction for "score"
    auto pos = json.find("\"score\"");
    if (pos == std::string::npos) return -1;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return -1;
    return std::stod(json.substr(pos + 1));
}

} // namespace

std::vector<BaselineVerdict> CompareBaseline(const std::string& baseline_dir,
                                             const std::vector<ImageResult>& results,
                                             double threshold) {
    std::vector<BaselineVerdict> verdicts;
    namespace fs = std::filesystem;
    for (auto& r : results) {
        BaselineVerdict v;
        v.name          = r.name;
        v.current_score = r.score;

        auto base_json = fs::path(baseline_dir) / (r.name + ".json");
        if (!fs::exists(base_json)) {
            v.status = BaselineVerdict::NEW_IMAGE;
            verdicts.push_back(v);
            continue;
        }
        auto content     = ReadFile(base_json.string());
        v.baseline_score = ExtractScore(content);
        double diff      = v.current_score - v.baseline_score;
        if (diff > threshold)
            v.status = BaselineVerdict::IMPROVED;
        else if (diff < -threshold)
            v.status = BaselineVerdict::REGRESSED;
        else
            v.status = BaselineVerdict::OK;
        verdicts.push_back(v);
    }
    return verdicts;
}

void SaveBaseline(const std::string& baseline_dir, const std::vector<ImageResult>& results) {
    namespace fs = std::filesystem;
    fs::create_directories(baseline_dir);
    int copied = 0;
    for (auto& r : results) {
        auto json_path = fs::path(baseline_dir) / (r.name + ".json");
        {
            std::ofstream f(json_path);
            if (f) {
                f << "{\n  \"score\": " << std::fixed << std::setprecision(1) << r.score
                  << ",\n  \"metrics\": " << r.metrics.ToJson(2) << "\n}\n";
                copied++;
            }
        }

        // Copy SVG
        if (!r.svg_path.empty() && fs::exists(r.svg_path)) {
            auto dst = fs::path(baseline_dir) / (r.name + ".svg");
            fs::copy_file(r.svg_path, dst, fs::copy_options::overwrite_existing);
            copied++;
        } else {
            spdlog::warn("SaveBaseline: no SVG for '{}', skipping", r.name);
        }
    }
    spdlog::info("Baseline updated: {} files -> {}", copied, baseline_dir);
}

// ── History ──────────────────────────────────────────────────────────────────

void AppendHistory(const std::string& history_path, const std::string& run_id,
                   const std::vector<ImageResult>& results, const std::string& note) {
    bool write_header = !std::filesystem::exists(history_path);
    std::ofstream f(history_path, std::ios::app);
    if (!f) {
        spdlog::error("Cannot open history file: {}", history_path);
        return;
    }

    if (write_header) {
        f << "run_id,timestamp,git_commit,images,score_avg,coverage_avg,delta_e_avg,ssim_avg,"
             "note\n";
    }

    double score_sum = 0, cov_sum = 0, de_sum = 0, ssim_sum = 0;
    for (auto& r : results) {
        score_sum += r.score;
        cov_sum += r.metrics.coverage;
        de_sum += r.metrics.delta_e_mean;
        ssim_sum += r.metrics.ssim;
    }
    int n      = static_cast<int>(results.size());
    double inv = (n > 0) ? 1.0 / n : 0;

    f << run_id << "," << eval::CurrentTimestamp() << "," << eval::GitShortHash() << "," << n << ","
      << std::fixed << std::setprecision(1) << score_sum * inv << "," << std::setprecision(4)
      << cov_sum * inv << "," << std::setprecision(1) << de_sum * inv << "," << std::setprecision(4)
      << ssim_sum * inv << "," << CsvEscape(note) << "\n";
}

// ── Report ───────────────────────────────────────────────────────────────────

void PrintReport(const std::vector<ImageResult>& results,
                 const std::vector<BaselineVerdict>& verdicts) {
    std::map<std::string, const BaselineVerdict*> vmap;
    for (auto& v : verdicts) vmap[v.name] = &v;

    std::string header = std::string("Category   ") + "Image            " + "Shapes " + "  Cvg% " +
                         " dE_mean " + "  SSIM " + " Score ";
    if (!verdicts.empty()) header += "    Verdict";
    std::printf("\n%s\n", header.c_str());
    std::printf("%s\n", std::string(header.size(), '-').c_str());

    int ok = 0, improved = 0, regressed = 0, new_img = 0, failed = 0;
    for (auto& r : results) {
        auto& m = r.metrics;
        std::printf("%-10s %-16s %6d %5.2f%% %8.1f %6.4f %6.1f", r.category.c_str(), r.name.c_str(),
                    m.total_shapes, m.coverage * 100.0, m.delta_e_mean, m.ssim, r.score);
        if (auto it = vmap.find(r.name); it != vmap.end()) {
            auto s              = it->second->status;
            const char* label[] = {"OK", "IMPROVED", "REGRESSED", "NEW"};
            std::printf(" %10s", label[s]);
            if (s == BaselineVerdict::REGRESSED) {
                regressed++;
                std::printf(" !");
            } else if (s == BaselineVerdict::IMPROVED)
                improved++;
            else if (s == BaselineVerdict::NEW_IMAGE)
                new_img++;
            else
                ok++;
        }
        if (!r.expectation_failures.empty()) {
            failed++;
            std::printf("  FAIL:");
            for (auto& f : r.expectation_failures) std::printf(" %s;", f.c_str());
        }
        std::printf("\n");
    }

    double score_avg = 0;
    for (auto& r : results) score_avg += r.score;
    if (!results.empty()) score_avg /= results.size();

    std::printf("\n--- Summary ---\n");
    std::printf("Total: %zu images", results.size());
    if (!verdicts.empty())
        std::printf(" | OK: %d | IMPROVED: %d | REGRESSED: %d | NEW: %d", ok, improved, regressed,
                    new_img);
    if (failed) std::printf(" | FAIL: %d", failed);
    std::printf("\nAggregate score: %.1f\n", score_avg);
}

// ── BenchmarkReport JSON ─────────────────────────────────────────────────────

std::string BenchmarkReport::ToJson(int indent) const {
    std::ostringstream ss;
    std::string ind(indent, ' ');
    std::string ind2(indent * 2, ' ');
    std::string ind3(indent * 3, ' ');

    ss << "{\n";
    ss << ind << "\"run_id\": \"" << JsonEscape(run_id) << "\",\n";
    ss << ind << "\"timestamp\": \"" << JsonEscape(timestamp) << "\",\n";
    ss << ind << "\"git_commit\": \"" << JsonEscape(git_commit) << "\",\n";
    ss << ind << "\"note\": \"" << JsonEscape(note) << "\",\n";
    ss << ind << "\"summary\": {\n";
    ss << ind2 << "\"total_images\": " << summary.total_images << ",\n";
    ss << ind2 << std::fixed << std::setprecision(1) << "\"score_avg\": " << summary.score_avg
       << ",\n";
    ss << ind2 << std::setprecision(4) << "\"coverage_avg\": " << summary.coverage_avg << ",\n";
    ss << ind2 << std::setprecision(1) << "\"delta_e_avg\": " << summary.delta_e_avg << ",\n";
    ss << ind2 << std::setprecision(4) << "\"ssim_avg\": " << summary.ssim_avg << ",\n";
    ss << ind2 << "\"regressions\": " << summary.regressions << ",\n";
    ss << ind2 << "\"improvements\": " << summary.improvements << ",\n";
    ss << ind2 << "\"failures\": " << summary.failures << "\n";
    ss << ind << "},\n";

    ss << ind << "\"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        auto& r = results[i];
        ss << ind2 << "{\n";
        ss << ind3 << "\"name\": \"" << JsonEscape(r.name) << "\",\n";
        ss << ind3 << "\"category\": \"" << JsonEscape(r.category) << "\",\n";
        ss << ind3 << "\"svg_path\": \"" << JsonEscape(r.svg_path) << "\",\n";
        ss << ind3 << std::setprecision(1) << "\"score\": " << r.score << ",\n";
        ss << ind3 << "\"metrics\": " << r.metrics.ToJson(indent) << ",\n";
        ss << ind3 << "\"expectation_failures\": [";
        for (size_t j = 0; j < r.expectation_failures.size(); ++j) {
            ss << "\"" << JsonEscape(r.expectation_failures[j]) << "\"";
            if (j + 1 < r.expectation_failures.size()) ss << ", ";
        }
        ss << "]\n";
        ss << ind2 << "}" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    ss << ind << "],\n";

    ss << ind << "\"verdicts\": [\n";
    for (size_t i = 0; i < verdicts.size(); ++i) {
        auto& v              = verdicts[i];
        const char* labels[] = {"OK", "IMPROVED", "REGRESSED", "NEW"};
        ss << ind2 << "{\"name\": \"" << JsonEscape(v.name) << "\", \"status\": \""
           << labels[v.status] << "\", \"baseline_score\": " << std::setprecision(1)
           << v.baseline_score << ", \"current_score\": " << v.current_score << "}";
        if (i + 1 < verdicts.size()) ss << ",";
        ss << "\n";
    }
    ss << ind << "]\n";

    ss << "}\n";
    return ss.str();
}

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace eval {

std::string GitShortHash() {
    std::array<char, 128> buf;
    std::string result = "unknown";
#if defined(_WIN32)
    FILE* pipe = _popen("git rev-parse --short HEAD 2>NUL", "r");
#else
    FILE* pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
#endif
    if (pipe) {
        if (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
            result = buf.data();
            while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
                result.pop_back();
        }
#if defined(_WIN32)
        _pclose(pipe);
#else
        pclose(pipe);
#endif
    }
    return result;
}

std::string CurrentTimestamp() {
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return buf;
}

std::string MakeRunId() {
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf) + "_" + GitShortHash();
}

} // namespace eval

} // namespace ChromaPrint3D
