/// \file detail/cv_utils.h
/// \brief Internal OpenCV utility functions shared across core modules.

#pragma once

#include "chromaprint3d/error.h"

#include <opencv2/imgproc.hpp>

namespace ChromaPrint3D::detail {

/// Ensure the input image is in BGR CV_8U format.
/// Handles BGRA (4-channel), grayscale (1-channel), and BGR (3-channel) inputs.
/// Converts higher bit-depth images (e.g. 16-bit PNG from iPhone) to 8-bit.
/// Returns an empty Mat if input is empty.
inline cv::Mat EnsureBgr(const cv::Mat& src) {
    if (src.empty()) { return cv::Mat(); }

    cv::Mat img = src;
    if (img.depth() != CV_8U) {
        double scale = (img.depth() == CV_16U || img.depth() == CV_16S) ? 1.0 / 256.0 : 1.0;
        img.convertTo(img, CV_8U, scale);
    }

    if (img.channels() == 3) { return img; }
    if (img.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    if (img.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    throw InputError("Unsupported image channel count: " + std::to_string(img.channels()));
}

/// Convert a BGR (uint8) image to CIE L*a*b* (float32).
/// Returns an empty Mat if input is empty.
inline cv::Mat BgrToLab(const cv::Mat& bgr) {
    if (bgr.empty()) { return cv::Mat(); }
    cv::Mat bgr_float;
    bgr.convertTo(bgr_float, CV_32F, 1.0 / 255.0);
    cv::Mat lab;
    cv::cvtColor(bgr_float, lab, cv::COLOR_BGR2Lab);
    return lab;
}

} // namespace ChromaPrint3D::detail
