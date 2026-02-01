#include "imgproc.h"
#include "vec3.h"

#include <opencv2/opencv.hpp>

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace ChromaPrint3D {

// helper functions
namespace {
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

static cv::Mat EnsureBgr(const cv::Mat& src) {
    if (src.channels() == 3) { return src; }
    if (src.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
        return bgr.clone();
    }
    if (src.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
        return bgr.clone();
    }
    throw std::runtime_error("Unsupported image channel count");
}

static int NormalizeOddKernel(int k) {
    if (k < 3) { return 3; }
    return (k % 2 == 0) ? (k + 1) : k;
}

static cv::Mat BgrToLab(const cv::Mat& bgr) {
    cv::Mat bgr_float;
    bgr.convertTo(bgr_float, CV_32F, 1.0 / 255.0);
    cv::Mat lab;
    cv::cvtColor(bgr_float, lab, cv::COLOR_BGR2Lab);
    return lab;
}

static cv::Mat BgrToRgbLinear(const cv::Mat& bgr) {
    // 1. BGR -> RGB (swap channels)
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // 2. Convert to float [0, 1]
    cv::Mat rgb_float;
    rgb.convertTo(rgb_float, CV_32F, 1.0 / 255.0);

    // 3. sRGB gamma correction -> linear RGB
    cv::Mat rgb_linear(rgb_float.size(), CV_32FC3);
    for (int i = 0; i < rgb_float.rows; i++) {
        for (int j = 0; j < rgb_float.cols; j++) {
            cv::Vec3f pixel = rgb_float.at<cv::Vec3f>(i, j);
            for (int c = 0; c < 3; c++) {
                float val = pixel[c];
                pixel[c]  = SrgbToLinear(val);
            }
            rgb_linear.at<cv::Vec3f>(i, j) = pixel;
        }
    }

    return rgb_linear;
}
} // namespace

ImgProcResult ImgProc::Run(const std::string& path) const {
    cv::Mat input = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (input.empty()) { throw std::runtime_error("Failed to read image: " + path); }

    // 1. Apply resize policy
    cv::Mat resized;
    Resize(input, resized);

    // 若需要背景去除，在这里插入

    // 2. Extract alpha mask
    cv::Mat mask;
    ExtractAlphaMask(input, resized.size(), mask);

    // 3. Denoise
    cv::Mat bgr = EnsureBgr(resized);
    cv::Mat denoised;
    Denoise(bgr, denoised);

    ImgProcResult result;
    result.width  = resized.cols;
    result.height = resized.rows;
    result.rgb    = BgrToRgbLinear(denoised);
    result.mask   = mask;
    result.lab    = BgrToLab(denoised);
    return result;
}

// 1. 不改变原图比例
// 2. 尺寸不超过 max_size
// 3. 根据 request_scale 做保持比例变化
void ImgProc::Resize(const cv::Mat& input, cv::Mat& resized) const {
    // 原始大小
    int orig_width  = input.cols;
    int orig_height = input.rows;

    // 目标大小
    int target_width  = orig_width;
    int target_height = orig_height;

    float scale = (request_scale > 0.0f) ? request_scale : 1.0f;

    // 限制最大尺寸
    if (max_width > 0) {
        float max_scale_w = static_cast<float>(max_width) / orig_width;
        scale             = std::min(max_scale_w, scale);
    }
    if (max_height > 0) {
        float max_scale_h = static_cast<float>(max_height) / orig_height;
        scale             = std::min(max_scale_h, scale);
    }

    target_width  = static_cast<int>(std::lround(target_width * scale));
    target_height = static_cast<int>(std::lround(target_height * scale));

    target_width  = std::max(1, target_width);
    target_height = std::max(1, target_height);

    bool is_downsample  = scale < 1.0f;
    ResizeMethod method = (is_downsample) ? downsample_method : upsample_method;
    cv::Size target_size(target_width, target_height);
    cv::resize(input, resized, target_size, 0.0, 0.0, ToCvMethod(method));
    return;
}

void ImgProc::ExtractAlphaMask(const cv::Mat& input, const cv::Size& target_size,
                               cv::Mat& mask) const {
    mask = cv::Mat(target_size, CV_8UC1, cv::Scalar(255));
    if (use_alpha_mask && input.channels() == 4) {
        // 使用原图 alpha 通道
        cv::Mat alpha;
        cv::extractChannel(input, alpha, 3);

        // 对于 mask 固定使用 Nearest 插值
        cv::Mat alpha_resized;
        cv::resize(alpha, alpha_resized, target_size, 0.0, 0.0, cv::INTER_NEAREST);
        mask = alpha_resized > alpha_threshold;
    }
    return;
}

void ImgProc::Denoise(const cv::Mat& input, cv::Mat& denoised) const {
    switch (denoise_method) {
    case DenoiseMethod::None:
        denoised = input;
        return;
    case DenoiseMethod::Median: {
        const int k = NormalizeOddKernel(denoise_kernel);
        if (k <= 1) {
            denoised = input;
            return;
        }
        cv::medianBlur(input, denoised, k);
        return;
    }
    case DenoiseMethod::Bilateral: {
        const int diameter       = std::max(1, bilateral_diameter);
        const double sigma_color = std::max(0.0f, bilateral_sigma_color);
        const double sigma_space = std::max(0.0f, bilateral_sigma_space);
        cv::bilateralFilter(input, denoised, diameter, sigma_color, sigma_space);
        return;
    }
    }
}
} // namespace ChromaPrint3D