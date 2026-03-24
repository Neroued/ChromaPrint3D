#include "chromaprint3d/matting.h"
#include "chromaprint3d/error.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace ChromaPrint3D {

using nlohmann::json;

// ── MattingModelConfig ──────────────────────────────────────────────────────

MattingModelConfig MattingModelConfig::LoadFromJson(const std::string& json_path) {
    std::ifstream in(json_path);
    if (!in.is_open()) { throw IOError("Failed to open matting model config: " + json_path); }

    json j;
    try {
        in >> j;
    } catch (const json::parse_error& e) {
        throw FormatError("Failed to parse matting config " + json_path + ": " + e.what());
    }

    MattingModelConfig cfg;
    cfg.name        = j.value("name", "");
    cfg.description = j.value("description", "");

    if (j.contains("preprocess")) {
        const auto& pre = j["preprocess"];

        if (pre.contains("input_size")) {
            int sz           = pre["input_size"].get<int>();
            cfg.input_width  = sz;
            cfg.input_height = sz;
        }
        if (pre.contains("input_width")) cfg.input_width = pre["input_width"].get<int>();
        if (pre.contains("input_height")) cfg.input_height = pre["input_height"].get<int>();

        cfg.channel_order = pre.value("channel_order", cfg.channel_order);
        cfg.layout        = pre.value("layout", cfg.layout);

        if (pre.contains("normalize_scale"))
            cfg.normalize_scale = pre["normalize_scale"].get<float>();
        if (pre.contains("normalize_mean")) {
            auto arr = pre["normalize_mean"];
            for (int i = 0; i < 3 && i < static_cast<int>(arr.size()); ++i)
                cfg.normalize_mean[i] = arr[i].get<float>();
        }
        if (pre.contains("normalize_std")) {
            auto arr = pre["normalize_std"];
            for (int i = 0; i < 3 && i < static_cast<int>(arr.size()); ++i)
                cfg.normalize_std[i] = arr[i].get<float>();
        }
    }

    if (j.contains("postprocess")) {
        const auto& post = j["postprocess"];
        if (post.contains("output_index")) cfg.output_index = post["output_index"].get<int>();
        if (post.contains("threshold")) cfg.threshold = post["threshold"].get<float>();
    }

    spdlog::debug("Loaded matting config '{}' from {}", cfg.name, json_path);
    return cfg;
}

// ── ThresholdMattingProvider ────────────────────────────────────────────────

ThresholdMattingProvider::ThresholdMattingProvider(float distance_threshold)
    : distance_threshold_(distance_threshold) {}

std::string ThresholdMattingProvider::Name() const { return "opencv-threshold"; }

std::string ThresholdMattingProvider::Description() const {
    return "基于 Lab 色彩空间的背景色阈值分割，适用于纯色背景图片";
}

MattingOutput ThresholdMattingProvider::Run(const cv::Mat& bgr, MattingTimingInfo* timing) const {
    if (bgr.empty()) return {};

    using Clock = std::chrono::steady_clock;
    auto t0     = Clock::now();

    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    lab.convertTo(lab, CV_32F);

    auto t1 = Clock::now();

    const int rows              = lab.rows;
    const int cols              = lab.cols;
    constexpr int kSampleRadius = 5;
    const int sr                = std::min(kSampleRadius, std::min(rows, cols) / 4);

    auto sample_mean = [&](int r0, int c0, int r1, int c1) -> cv::Vec3f {
        r0              = std::clamp(r0, 0, rows);
        r1              = std::clamp(r1, 0, rows);
        c0              = std::clamp(c0, 0, cols);
        c1              = std::clamp(c1, 0, cols);
        cv::Mat roi     = lab(cv::Range(r0, r1), cv::Range(c0, c1));
        cv::Scalar mean = cv::mean(roi);
        return {static_cast<float>(mean[0]), static_cast<float>(mean[1]),
                static_cast<float>(mean[2])};
    };

    cv::Vec3f tl = sample_mean(0, 0, sr, sr);
    cv::Vec3f tr = sample_mean(0, cols - sr, sr, cols);
    cv::Vec3f bl = sample_mean(rows - sr, 0, rows, sr);
    cv::Vec3f br = sample_mean(rows - sr, cols - sr, rows, cols);
    cv::Vec3f bg = (tl + tr + bl + br) * 0.25f;

    cv::Mat mask(rows, cols, CV_8UC1);
    const float thr2 = distance_threshold_ * distance_threshold_;

    for (int r = 0; r < rows; ++r) {
        const auto* row = lab.ptr<cv::Vec3f>(r);
        auto* out       = mask.ptr<uint8_t>(r);
        for (int c = 0; c < cols; ++c) {
            cv::Vec3f diff = row[c] - bg;
            float dist2    = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
            out[c]         = (dist2 > thr2) ? 255 : 0;
        }
    }

    auto t2 = Clock::now();

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    auto t3 = Clock::now();

    auto ms        = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
    double pre_ms  = ms(t1 - t0);
    double inf_ms  = ms(t2 - t1);
    double post_ms = ms(t3 - t2);
    double tot_ms  = ms(t3 - t0);

    spdlog::debug("ThresholdMatting: preprocess={:.1f}ms, threshold={:.1f}ms, morphology={:.1f}ms, "
                  "total={:.1f}ms",
                  pre_ms, inf_ms, post_ms, tot_ms);

    if (timing) {
        timing->preprocess_ms  = pre_ms;
        timing->inference_ms   = inf_ms;
        timing->postprocess_ms = post_ms;
        timing->total_ms       = tot_ms;
    }

    return {mask, {}};
}

// ── MattingRegistry ─────────────────────────────────────────────────────────

void MattingRegistry::Register(std::string key, MattingProviderPtr provider) {
    spdlog::debug("MattingRegistry: registered provider '{}'", key);
    providers_[std::move(key)] = std::move(provider);
}

IMattingProvider* MattingRegistry::Get(const std::string& key) const {
    auto it = providers_.find(key);
    return (it != providers_.end()) ? it->second.get() : nullptr;
}

MattingProviderPtr MattingRegistry::GetShared(const std::string& key) const {
    auto it = providers_.find(key);
    return (it != providers_.end()) ? it->second : nullptr;
}

bool MattingRegistry::Has(const std::string& key) const { return providers_.count(key) > 0; }

std::vector<std::string> MattingRegistry::Available() const {
    std::vector<std::string> keys;
    keys.reserve(providers_.size());
    for (const auto& [k, _] : providers_) keys.push_back(k);
    std::sort(keys.begin(), keys.end());
    return keys;
}

// ── Model discovery ─────────────────────────────────────────────────────────

std::vector<std::pair<std::string, MattingModelConfig>>
DiscoverMattingModels(const std::string& directory) {
    namespace fs = std::filesystem;

    std::vector<std::pair<std::string, MattingModelConfig>> results;

    if (!fs::is_directory(directory)) {
        spdlog::warn("Matting model directory does not exist: {}", directory);
        return results;
    }

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;

        const auto& path = entry.path();
        auto ext         = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext != ".onnx") continue;

        fs::path json_path = path;
        json_path.replace_extension(".json");
        if (!fs::is_regular_file(json_path)) {
            spdlog::warn("Matting model {} has no JSON sidecar config, skipping",
                         path.filename().string());
            continue;
        }

        try {
            auto config      = MattingModelConfig::LoadFromJson(json_path.string());
            std::string stem = path.stem().string();
            if (config.name.empty()) config.name = stem;
            results.emplace_back(path.string(), std::move(config));
            spdlog::info("Discovered matting model: {} ({})", stem, path.string());
        } catch (const std::exception& e) {
            spdlog::error("Failed to load config for {}: {}", path.filename().string(), e.what());
        }
    }

    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return std::filesystem::path(a.first).stem().string() <
               std::filesystem::path(b.first).stem().string();
    });

    return results;
}

} // namespace ChromaPrint3D
