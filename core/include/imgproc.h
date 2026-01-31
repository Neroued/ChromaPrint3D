#pragma once

#include "common.h"

#include <string>

#include <opencv2/opencv.hpp>

namespace ChromaPrint3D {

struct ImgProcResult {
    int width  = 0;
    int height = 0;

    cv::Mat bgr;  // H x W, CV_8UC3
    cv::Mat mask; // H x W, CV_8UC1
    cv::Mat lab;  // H x W, CV_32FC3
};

class ImgProc {
public:
    float request_scale = 1.0f;

    int max_width  = 0; // 最大尺寸，非0则会保持长宽比进行缩放
    int max_height = 0;

    bool use_alpha_mask     = true;
    uint8_t alpha_threshold = 1; // alpha <= threshold 则不打印

    ResizeMethod upsample_method   = ResizeMethod::Nearest; // 升/降采样使用不同方法
    ResizeMethod downsample_method = ResizeMethod::Area;

    DenoiseMethod denoise_method = DenoiseMethod::None;
    int denoise_kernel           = 3; // Median kernel size (odd, >= 3)
    int bilateral_diameter       = 5; // Bilateral filter diameter
    float bilateral_sigma_color  = 25.0f;
    float bilateral_sigma_space  = 5.0f;

    ImgProcResult Run(const std::string& path) const;

private:
    void Resize(const cv::Mat& input, cv::Mat& resized) const;

    void ExtractAlphaMask(const cv::Mat& input, const cv::Size& target_size, cv::Mat& mask) const;

    void Denoise(const cv::Mat& input, cv::Mat& denoised) const;
};


} // namespace ChromaPrint3D