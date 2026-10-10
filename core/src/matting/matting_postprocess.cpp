#include "chromaprint3d/matting_postprocess.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#    include <opencv2/geometry.hpp> // OpenCV 5 moved DistanceTypes/boundingRect to the geometry module
#endif

#include <algorithm>
#include <cmath>
#include <vector>

namespace ChromaPrint3D {

namespace {

cv::Mat EllipseKernel(int radius) {
    int d = std::max(radius * 2 + 1, 1);
    return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(d, d));
}

int ResolveAdaptiveOutlineWidth(const cv::Size& mask_size, int requested_width) {
    if (requested_width <= 0) return 0;
    if (mask_size.width <= 0 || mask_size.height <= 0) return requested_width;

    constexpr double kReferenceShortSide = 1024.0;
    constexpr int kMaxAdaptiveWidth      = 96;
    const int short_side                 = std::min(mask_size.width, mask_size.height);
    const double scale = std::max(1.0, static_cast<double>(short_side) / kReferenceShortSide);
    const int scaled = static_cast<int>(std::lround(static_cast<double>(requested_width) * scale));
    return std::clamp(scaled, requested_width, kMaxAdaptiveWidth);
}

cv::Mat CompositeForeground(const cv::Mat& bgr, const cv::Mat& mask) {
    cv::Mat bgra;
    cv::cvtColor(bgr, bgra, cv::COLOR_BGR2BGRA);
    std::vector<cv::Mat> channels(4);
    cv::split(bgra, channels);
    channels[3] = mask;
    cv::merge(channels, bgra);
    return bgra;
}

cv::Mat DrawOutline(const cv::Mat& mask, int width, const std::array<uint8_t, 3>& color_bgr,
                    OutlineMode mode) {
    cv::Mat outline(mask.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));
    const int effective_width = ResolveAdaptiveOutlineWidth(mask.size(), width);
    if (effective_width <= 0) return outline;

    cv::Mat band;

    switch (mode) {
    case OutlineMode::Outer: {
        cv::Mat dilated;
        cv::dilate(mask, dilated, EllipseKernel(effective_width));
        cv::subtract(dilated, mask, band);
        break;
    }
    case OutlineMode::Inner: {
        cv::Mat eroded;
        cv::erode(mask, eroded, EllipseKernel(effective_width));
        cv::subtract(mask, eroded, band);
        break;
    }
    case OutlineMode::Center:
    default: {
        cv::Mat dilated, eroded;
        cv::dilate(mask, dilated, EllipseKernel((effective_width + 1) / 2));
        cv::erode(mask, eroded, EllipseKernel(effective_width / 2));
        cv::subtract(dilated, eroded, band);
        break;
    }
    }

    outline.setTo(cv::Scalar(color_bgr[0], color_bgr[1], color_bgr[2], 255), band);
    return outline;
}

bool HasForegroundPixels(const cv::Mat& mask) {
    return !mask.empty() && cv::countNonZero(mask) > 0;
}

int ResolveMorphCloseExpandPadding(const MattingPostprocessParams& params) {
    if (params.morph_close_size <= 0) return 0;
    const int ksize      = params.morph_close_size | 1;
    const int radius     = ksize / 2;
    const int iterations = std::max(params.morph_close_iterations, 1);
    return radius * iterations;
}

int ResolveOutlineReservePadding(const cv::Size& mask_size,
                                 const MattingPostprocessParams& params) {
    if (!params.outline_enabled || params.outline_width <= 0) return 0;
    const int effective_width = ResolveAdaptiveOutlineWidth(mask_size, params.outline_width);
    switch (params.outline_mode) {
    case OutlineMode::Inner:
        return 0;
    case OutlineMode::Center:
        return (effective_width + 1) / 2;
    case OutlineMode::Outer:
    default:
        return effective_width;
    }
}

int ResolveSafeReframeBorder(const cv::Size& mask_size, const MattingPostprocessParams& params) {
    if (!params.reframe_enabled) return 0;
    const int reframe_padding = std::max(params.reframe_padding_px, 0);
    const int morph_padding   = ResolveMorphCloseExpandPadding(params);
    const int outline_padding = ResolveOutlineReservePadding(mask_size, params);
    return reframe_padding + morph_padding + outline_padding + 2;
}

cv::Rect ExpandRectInBounds(const cv::Rect& rect, int padding, const cv::Size& bounds) {
    const int safe_padding = std::max(padding, 0);
    const int left         = std::max(0, rect.x - safe_padding);
    const int top          = std::max(0, rect.y - safe_padding);
    const int right        = std::min(bounds.width, rect.x + rect.width + safe_padding);
    const int bottom       = std::min(bounds.height, rect.y + rect.height + safe_padding);
    return cv::Rect(left, top, std::max(0, right - left), std::max(0, bottom - top));
}

void FilterSmallRegions(cv::Mat& mask, int min_area) {
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids);

    cv::Mat filtered = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (int i = 1; i < n; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= min_area) { filtered.setTo(255, labels == i); }
    }
    mask = filtered;
}

} // namespace

MattingPostprocessResult ApplyMattingPostprocess(const cv::Mat& alpha, const cv::Mat& fallback_mask,
                                                 const cv::Mat& original,
                                                 const MattingPostprocessParams& params) {
    MattingPostprocessResult result;

    if (!alpha.empty()) {
        double thresh = std::clamp(static_cast<double>(params.threshold), 0.0, 1.0) * 255.0;
        cv::threshold(alpha, result.mask, thresh, 255.0, cv::THRESH_BINARY);
    } else if (!fallback_mask.empty()) {
        result.mask = fallback_mask.clone();
    } else {
        return result;
    }

    const cv::Size source_size = result.mask.size();
    cv::Mat working_mask       = result.mask;
    cv::Mat working_original   = original;
    int reframe_border         = 0;
    if (params.reframe_enabled && HasForegroundPixels(working_mask)) {
        reframe_border = ResolveSafeReframeBorder(working_mask.size(), params);
        if (reframe_border > 0) {
            cv::copyMakeBorder(working_mask, working_mask, reframe_border, reframe_border,
                               reframe_border, reframe_border, cv::BORDER_CONSTANT, cv::Scalar(0));
            if (!working_original.empty()) {
                cv::copyMakeBorder(working_original, working_original, reframe_border,
                                   reframe_border, reframe_border, reframe_border,
                                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            }
        }
    }

    if (params.morph_close_size > 0) {
        int ksize      = params.morph_close_size | 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        int iterations = std::max(params.morph_close_iterations, 1);
        cv::morphologyEx(working_mask, working_mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1),
                         iterations);
    }

    if (params.min_region_area > 0) { FilterSmallRegions(working_mask, params.min_region_area); }

    if (params.reframe_enabled) {
        if (HasForegroundPixels(working_mask)) {
            std::vector<cv::Point> foreground_pixels;
            cv::findNonZero(working_mask, foreground_pixels);
            const cv::Rect bbox       = cv::boundingRect(foreground_pixels);
            const int reframe_padding = std::max(params.reframe_padding_px, 0);
            const int outline_padding = ResolveOutlineReservePadding(working_mask.size(), params);
            const cv::Rect roi =
                ExpandRectInBounds(bbox, reframe_padding + outline_padding, working_mask.size());
            if (roi.width > 0 && roi.height > 0) {
                working_mask = working_mask(roi).clone();
                if (!working_original.empty()) { working_original = working_original(roi).clone(); }
            }
        } else if (reframe_border > 0 && source_size.width > 0 && source_size.height > 0) {
            const cv::Rect restore_roi(reframe_border, reframe_border, source_size.width,
                                       source_size.height);
            const bool restore_valid = restore_roi.x + restore_roi.width <= working_mask.cols &&
                                       restore_roi.y + restore_roi.height <= working_mask.rows;
            if (restore_valid) {
                working_mask = working_mask(restore_roi).clone();
                if (!working_original.empty()) {
                    working_original = working_original(restore_roi).clone();
                }
            }
        }
    }

    result.mask = std::move(working_mask);

    if (!working_original.empty()) {
        result.foreground = CompositeForeground(working_original, result.mask);
    }

    if (params.outline_enabled && params.outline_width > 0) {
        result.outline = DrawOutline(result.mask, params.outline_width, params.outline_color,
                                     params.outline_mode);
    }

    return result;
}

} // namespace ChromaPrint3D
