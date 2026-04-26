/// \file detail/cv_utils.h
/// \brief Internal OpenCV utility functions shared across core modules.

#pragma once

#include "chromaprint3d/error.h"

#include <opencv2/imgproc.hpp>

#include <cstdint>

namespace ChromaPrint3D::detail {

enum class BgraPolicy : uint8_t {
    CompositeWhite,
    DropAlpha,
};

/// Ensure the input image is in BGR CV_8U format.
/// Handles BGRA (4-channel), grayscale (1-channel), and BGR (3-channel) inputs.
/// Converts higher bit-depth images (e.g. 16-bit PNG from iPhone) to 8-bit.
/// BGRA handling is configurable: alpha-composite onto white (default) or drop alpha without blend.
/// Returns an empty Mat if input is empty.
inline cv::Mat EnsureBgr(const cv::Mat& src, BgraPolicy bgra_policy = BgraPolicy::CompositeWhite) {
    if (src.empty()) { return cv::Mat(); }

    cv::Mat img = src;
    if (img.depth() != CV_8U) {
        double scale = (img.depth() == CV_16U || img.depth() == CV_16S) ? 1.0 / 256.0 : 1.0;
        img.convertTo(img, CV_8U, scale);
    }

    if (img.channels() == 3) { return img; }
    if (img.channels() == 4) {
        if (bgra_policy == BgraPolicy::DropAlpha) {
            cv::Mat bgr;
            cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);
            return bgr;
        }
        cv::Mat bgr(img.rows, img.cols, CV_8UC3);
        for (int r = 0; r < img.rows; ++r) {
            const cv::Vec4b* src_row = img.ptr<cv::Vec4b>(r);
            cv::Vec3b* dst_row       = bgr.ptr<cv::Vec3b>(r);
            for (int c = 0; c < img.cols; ++c) {
                const cv::Vec4b& px = src_row[c];
                const int a         = static_cast<int>(px[3]);
                if (a <= 0) {
                    dst_row[c] = cv::Vec3b(255, 255, 255);
                    continue;
                }
                if (a >= 255) {
                    dst_row[c] = cv::Vec3b(px[0], px[1], px[2]);
                    continue;
                }

                const int inv_a = 255 - a;
                dst_row[c][0]   = static_cast<uint8_t>((px[0] * a + 255 * inv_a + 127) / 255);
                dst_row[c][1]   = static_cast<uint8_t>((px[1] * a + 255 * inv_a + 127) / 255);
                dst_row[c][2]   = static_cast<uint8_t>((px[2] * a + 255 * inv_a + 127) / 255);
            }
        }
        return bgr;
    }
    if (img.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    throw InputError("Unsupported image channel count: " + std::to_string(img.channels()));
}

/// Extract an opaque-mask from alpha channel.
/// Non-BGRA input returns an all-opaque mask.
inline cv::Mat ExtractOpaqueMask(const cv::Mat& src, uint8_t alpha_threshold = 0) {
    if (src.empty()) { return cv::Mat(); }

    cv::Mat img = src;
    if (img.depth() != CV_8U) {
        double scale = (img.depth() == CV_16U || img.depth() == CV_16S) ? 1.0 / 256.0 : 1.0;
        img.convertTo(img, CV_8U, scale);
    }

    cv::Mat mask(img.rows, img.cols, CV_8UC1, cv::Scalar(255));
    if (img.channels() != 4) { return mask; }

    cv::Mat alpha;
    cv::extractChannel(img, alpha, 3);
    cv::threshold(alpha, mask, static_cast<double>(alpha_threshold), 255, cv::THRESH_BINARY);
    return mask;
}

// detail::BgrToLab removed: superseded by ChromaPrint3D::BgrToLab in
// `chromaprint3d/color/image.h`, which self-implements the project D65 Lab
// pipeline (no OpenCV cvtColor(BGR2Lab) call).

} // namespace ChromaPrint3D::detail
