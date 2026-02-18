#include "chromaprint3d/imgproc.h"
#include "chromaprint3d/color.h"
#include "chromaprint3d/error.h"
#include "detail/cv_utils.h"

#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>

#include <cmath>
#include <algorithm>
#include <filesystem>

namespace ChromaPrint3D {

namespace {

constexpr int kMinKernelSize = 3;

static auto ToCvMethod(ResizeMethod method) {
    switch (method) {
    case ResizeMethod::Nearest:
        return cv::INTER_NEAREST;
    case ResizeMethod::Area:
        return cv::INTER_AREA;
    case ResizeMethod::Linear:
        return cv::INTER_LINEAR;
    case ResizeMethod::Cubic:
        return cv::INTER_CUBIC;
    }
    return cv::INTER_AREA;
}

static int NormalizeOddKernel(int k) {
    if (k < kMinKernelSize) { return kMinKernelSize; }
    return (k % 2 == 0) ? (k + 1) : k;
}

static const cv::Mat& SrgbToLinearLut() {
    static const cv::Mat lut = []() {
        cv::Mat table(1, 256, CV_32FC1);
        float* ptr = table.ptr<float>();
        for (int i = 0; i < 256; ++i) {
            ptr[i] = SrgbToLinear(static_cast<float>(i) / 255.0f);
        }
        return table;
    }();
    return lut;
}

static cv::Mat BgrToRgbLinear(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat channels[3];
    cv::split(rgb, channels);

    const cv::Mat& lut = SrgbToLinearLut();
    for (int c = 0; c < 3; ++c) {
        cv::LUT(channels[c], lut, channels[c]);
    }

    cv::Mat rgb_linear;
    cv::merge(channels, 3, rgb_linear);
    return rgb_linear;
}

static std::string PathStem(const std::string& path) {
    if (path.empty()) { return {}; }
    std::filesystem::path p(path);
    std::string stem = p.stem().string();
    if (!stem.empty()) { return stem; }
    return p.filename().string();
}
} // namespace

ImgProc::ImgProc(const ImgProcConfig& config) : config_(config) {}

ImgProcResult ImgProc::Run(const std::string& path) const {
    spdlog::info("ImgProc: loading image from file: {}", path);
    cv::Mat input = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (input.empty()) { throw IOError("Failed to read image: " + path); }
    spdlog::info("ImgProc: loaded {}x{}, {} channel(s)", input.cols, input.rows, input.channels());
    return Run(input, PathStem(path));
}

ImgProcResult ImgProc::Run(const cv::Mat& input, const std::string& name) const {
    if (input.empty()) { throw InputError("ImgProc::Run: input image is empty"); }

    cv::Mat resized;
    Resize(input, resized);

    cv::Mat mask;
    ExtractAlphaMask(input, resized.size(), mask);

    cv::Mat bgr = detail::EnsureBgr(resized);
    cv::Mat denoised;
    Denoise(bgr, denoised);

    ImgProcResult result;
    result.name   = name;
    result.width  = resized.cols;
    result.height = resized.rows;
    result.rgb    = BgrToRgbLinear(denoised);
    result.mask   = mask;
    result.lab    = detail::BgrToLab(denoised);
    return result;
}

ImgProcResult ImgProc::RunFromBuffer(const std::vector<uint8_t>& buffer,
                                     const std::string& name) const {
    if (buffer.empty()) { throw InputError("ImgProc::RunFromBuffer: buffer is empty"); }
    spdlog::info("ImgProc: decoding image from buffer ({} bytes, name={})", buffer.size(), name);
    cv::Mat input = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
    if (input.empty()) {
        throw IOError("ImgProc::RunFromBuffer: failed to decode image");
    }
    spdlog::info("ImgProc: decoded {}x{}, {} channel(s)", input.cols, input.rows, input.channels());
    return Run(input, name);
}

void ImgProc::Resize(const cv::Mat& input, cv::Mat& resized) const {
    int orig_width  = input.cols;
    int orig_height = input.rows;

    int target_width  = orig_width;
    int target_height = orig_height;

    float scale = (config_.scale > 0.0f) ? config_.scale : 1.0f;

    if (config_.max_width > 0) {
        float max_scale_w = static_cast<float>(config_.max_width) / orig_width;
        scale             = std::min(max_scale_w, scale);
    }
    if (config_.max_height > 0) {
        float max_scale_h = static_cast<float>(config_.max_height) / orig_height;
        scale             = std::min(max_scale_h, scale);
    }

    target_width  = static_cast<int>(std::lround(target_width * scale));
    target_height = static_cast<int>(std::lround(target_height * scale));

    target_width  = std::max(1, target_width);
    target_height = std::max(1, target_height);

    bool is_downsample  = scale < 1.0f;
    ResizeMethod method = is_downsample ? config_.downsample_method : config_.upsample_method;
    cv::Size target_size(target_width, target_height);
    cv::resize(input, resized, target_size, 0.0, 0.0, ToCvMethod(method));
    spdlog::info("ImgProc: resize {}x{} -> {}x{} (scale={:.3f})", orig_width, orig_height,
                 target_width, target_height, scale);
}

void ImgProc::ExtractAlphaMask(const cv::Mat& input, const cv::Size& target_size,
                               cv::Mat& mask) const {
    mask = cv::Mat(target_size, CV_8UC1, cv::Scalar(255));
    if (config_.use_alpha_mask && input.channels() == 4) {
        cv::Mat alpha;
        cv::extractChannel(input, alpha, 3);

        cv::Mat alpha_resized;
        cv::resize(alpha, alpha_resized, target_size, 0.0, 0.0, cv::INTER_NEAREST);
        mask = alpha_resized > config_.alpha_threshold;
    }
}

void ImgProc::Denoise(const cv::Mat& input, cv::Mat& denoised) const {
    switch (config_.denoise_method) {
    case DenoiseMethod::None:
        denoised = input;
        return;
    case DenoiseMethod::Median: {
        const int k = NormalizeOddKernel(config_.denoise_kernel);
        if (k <= 1) {
            denoised = input;
            return;
        }
        cv::medianBlur(input, denoised, k);
        return;
    }
    case DenoiseMethod::Bilateral: {
        const int diameter       = std::max(1, config_.bilateral_diameter);
        const double sigma_color = std::max(0.0f, config_.bilateral_sigma_color);
        const double sigma_space = std::max(0.0f, config_.bilateral_sigma_space);
        cv::bilateralFilter(input, denoised, diameter, sigma_color, sigma_space);
        return;
    }
    }
}

} // namespace ChromaPrint3D
