#include "imgproc/vectorize_preprocess.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <cmath>

namespace ChromaPrint3D::detail {

PreprocessResult PreprocessForVectorize(const cv::Mat& bgr, bool enable_color_smoothing,
                                        float smoothing_spatial, float smoothing_color,
                                        int upscale_short_edge, int max_working_pixels) {
    PreprocessResult result;
    result.bgr   = bgr;
    result.scale = 1.0f;
    spdlog::debug(
        "Vectorize preprocess start: input={}x{}, smoothing_enabled={}, sp={:.2f}, sr={:.2f}, "
        "upscale_short_edge={}, max_working_pixels={}",
        bgr.cols, bgr.rows, enable_color_smoothing, smoothing_spatial, smoothing_color,
        upscale_short_edge, max_working_pixels);

    const int h                   = bgr.rows;
    const int w                   = bgr.cols;
    const int short_edge          = std::min(h, w);
    const std::int64_t total_px   = static_cast<std::int64_t>(h) * static_cast<std::int64_t>(w);
    const bool enable_downscale   = max_working_pixels > 0;
    const bool enable_upscale     = upscale_short_edge > 0;
    bool downscale_applied        = false;
    constexpr float kScaleEpsilon = 1e-6f;

    if (enable_downscale && total_px > static_cast<std::int64_t>(max_working_pixels)) {
        float factor = std::sqrt(static_cast<float>(max_working_pixels) /
                                 static_cast<float>(std::max<std::int64_t>(1, total_px)));
        int new_h    = std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * factor)));
        int new_w    = std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * factor)));
        if (new_h < h || new_w < w) {
            cv::resize(result.bgr, result.bgr, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
            const float fx = static_cast<float>(new_w) / static_cast<float>(w);
            const float fy = static_cast<float>(new_h) / static_cast<float>(h);
            result.scale *= std::min(fx, fy);
            downscale_applied = true;
            spdlog::debug("Vectorize preprocess downscale applied: {}x{} -> {}x{}, factor={:.3f}, "
                          "target_max_pixels={}",
                          w, h, new_w, new_h, result.scale, max_working_pixels);
        }
    }

    if (!downscale_applied && enable_upscale && short_edge < upscale_short_edge &&
        total_px < 1000000) {
        float target_factor =
            static_cast<float>(upscale_short_edge) / static_cast<float>(std::max(1, short_edge));
        float factor = std::min(2.0f, target_factor);
        int new_h    = std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * factor)));
        int new_w    = std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * factor)));
        if (static_cast<std::int64_t>(new_h) * static_cast<std::int64_t>(new_w) > 4000000LL) {
            factor =
                std::sqrt(4000000.0f / static_cast<float>(std::max<std::int64_t>(1, total_px)));
            new_h = std::max(1, static_cast<int>(std::lround(static_cast<float>(h) * factor)));
            new_w = std::max(1, static_cast<int>(std::lround(static_cast<float>(w) * factor)));
        }
        if (factor > 1.05f) {
            cv::resize(bgr, result.bgr, cv::Size(new_w, new_h), 0, 0, cv::INTER_LANCZOS4);
            result.scale *= factor;
            spdlog::debug("Vectorize preprocess upscale applied: {}x{} -> {}x{}, factor={:.3f}", w,
                          h, new_w, new_h, factor);
        }
    }

    result.unsmoothed_bgr = result.bgr.clone();

    const float sp = std::max(0.0f, smoothing_spatial);
    const float sr = std::max(0.0f, smoothing_color);
    if (enable_color_smoothing && sp > 0.0f && sr > 0.0f) {
        cv::Mat filtered;
        cv::pyrMeanShiftFiltering(result.bgr, filtered, sp, sr);
        result.bgr = filtered;
        spdlog::debug("Vectorize preprocess mean-shift applied: sp={:.2f}, sr={:.2f}", sp, sr);
    } else {
        spdlog::debug("Vectorize preprocess mean-shift skipped: enabled={}, sp={:.2f}, sr={:.2f}",
                      enable_color_smoothing, sp, sr);
    }

    if (std::abs(result.scale - 1.0f) <= kScaleEpsilon) result.scale = 1.0f;
    spdlog::debug("Vectorize preprocess done: output={}x{}, scale={:.3f}", result.bgr.cols,
                  result.bgr.rows, result.scale);
    return result;
}

} // namespace ChromaPrint3D::detail
