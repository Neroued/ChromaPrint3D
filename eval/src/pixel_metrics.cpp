#include "pixel_metrics.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ChromaPrint3D::eval {

namespace {

double ComputePsnr(const cv::Mat& a, const cv::Mat& b, const cv::Mat& mask) {
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff.convertTo(diff, CV_64F);
    cv::Mat sq;
    cv::multiply(diff, diff, sq);

    cv::Scalar m = mask.empty() ? cv::mean(sq) : cv::mean(sq, mask);
    double mse   = (a.channels() == 3) ? (m[0] + m[1] + m[2]) / 3.0 : m[0];
    if (mse < 1e-10) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

double ComputeSsim(const cv::Mat& a, const cv::Mat& b, const cv::Mat& mask) {
    // For SSIM with mask: neutralize outside-mask areas to prevent GaussianBlur
    // boundary artifacts, then take mean of SSIM map only over masked pixels.
    cv::Mat a_in = a, b_in = b;
    if (!mask.empty()) {
        a_in = a.clone();
        b_in = b.clone();
        cv::Mat inv;
        cv::bitwise_not(mask, inv);
        a_in.setTo(cv::Scalar(128, 128, 128), inv);
        b_in.setTo(cv::Scalar(128, 128, 128), inv);
    }

    cv::Mat a32, b32;
    a_in.convertTo(a32, CV_32F);
    b_in.convertTo(b32, CV_32F);

    constexpr double C1 = 6.5025;  // (0.01*255)^2
    constexpr double C2 = 58.5225; // (0.03*255)^2

    cv::Mat mu_a, mu_b;
    cv::GaussianBlur(a32, mu_a, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(b32, mu_b, cv::Size(11, 11), 1.5);

    cv::Mat mu_a2, mu_b2, mu_ab;
    cv::multiply(mu_a, mu_a, mu_a2);
    cv::multiply(mu_b, mu_b, mu_b2);
    cv::multiply(mu_a, mu_b, mu_ab);

    cv::Mat a2, b2, ab;
    cv::multiply(a32, a32, a2);
    cv::multiply(b32, b32, b2);
    cv::multiply(a32, b32, ab);

    cv::Mat sigma_a2, sigma_b2, sigma_ab;
    cv::GaussianBlur(a2, sigma_a2, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(b2, sigma_b2, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(ab, sigma_ab, cv::Size(11, 11), 1.5);
    sigma_a2 -= mu_a2;
    sigma_b2 -= mu_b2;
    sigma_ab -= mu_ab;

    cv::Mat num, den, ssim_map;
    num = (2.0 * mu_ab + C1).mul(2.0 * sigma_ab + C2);
    den = (mu_a2 + mu_b2 + C1).mul(sigma_a2 + sigma_b2 + C2);
    cv::divide(num, den, ssim_map);

    cv::Scalar s = mask.empty() ? cv::mean(ssim_map) : cv::mean(ssim_map, mask);
    if (ssim_map.channels() == 3) return (s[0] + s[1] + s[2]) / 3.0;
    return s[0];
}

struct DeltaEStats {
    double mean    = 0;
    double p95     = 0;
    double max_val = 0;
};

DeltaEStats ComputeDeltaE(const cv::Mat& orig_bgr, const cv::Mat& rend_bgr, const cv::Mat& mask) {
    cv::Mat orig_f, rend_f;
    orig_bgr.convertTo(orig_f, CV_32FC3, 1.0 / 255.0);
    rend_bgr.convertTo(rend_f, CV_32FC3, 1.0 / 255.0);

    cv::Mat orig32, rend32;
    cv::cvtColor(orig_f, orig32, cv::COLOR_BGR2Lab);
    cv::cvtColor(rend_f, rend32, cv::COLOR_BGR2Lab);

    cv::Mat diff;
    cv::subtract(orig32, rend32, diff);
    cv::Mat diff_sq;
    cv::multiply(diff, diff, diff_sq);

    std::vector<cv::Mat> channels(3);
    cv::split(diff_sq, channels);

    cv::Mat sum_sq = channels[0] + channels[1] + channels[2];
    cv::Mat de;
    cv::sqrt(sum_sq, de);

    DeltaEStats stats;

    if (!mask.empty()) {
        cv::minMaxLoc(de, nullptr, &stats.max_val, nullptr, nullptr, mask);
        stats.mean = cv::mean(de, mask)[0];
    } else {
        double min_val;
        cv::minMaxLoc(de, &min_val, &stats.max_val);
        stats.mean = cv::mean(de)[0];
    }

    // p95 via partial sort (only over masked pixels)
    std::vector<float> vals;
    vals.reserve(de.rows * de.cols);
    const uchar* mask_data = mask.empty() ? nullptr : mask.ptr<uchar>();
    for (int r = 0; r < de.rows; ++r) {
        const float* row      = de.ptr<float>(r);
        const uchar* mask_row = mask_data ? mask.ptr<uchar>(r) : nullptr;
        for (int c = 0; c < de.cols; ++c) {
            if (!mask_row || mask_row[c]) vals.push_back(row[c]);
        }
    }
    if (!vals.empty()) {
        size_t idx95 = static_cast<size_t>(vals.size() * 0.95);
        if (idx95 >= vals.size()) idx95 = vals.size() - 1;
        std::nth_element(vals.begin(), vals.begin() + static_cast<long>(idx95), vals.end());
        stats.p95 = vals[idx95];
    }
    return stats;
}

double ComputeBorderDeltaE(const cv::Mat& orig_bgr, const cv::Mat& rend_bgr,
                           const cv::Mat& coverage, const cv::Mat& alpha_mask) {
    cv::Mat gray;
    cv::cvtColor(rend_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    cv::Mat border_mask;
    cv::dilate(edges, border_mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
    cv::bitwise_and(border_mask, coverage, border_mask);
    if (!alpha_mask.empty()) cv::bitwise_and(border_mask, alpha_mask, border_mask);

    if (cv::countNonZero(border_mask) == 0) return 0.0;

    cv::Mat orig_f, rend_f;
    orig_bgr.convertTo(orig_f, CV_32FC3, 1.0 / 255.0);
    rend_bgr.convertTo(rend_f, CV_32FC3, 1.0 / 255.0);

    cv::Mat orig32, rend32;
    cv::cvtColor(orig_f, orig32, cv::COLOR_BGR2Lab);
    cv::cvtColor(rend_f, rend32, cv::COLOR_BGR2Lab);

    cv::Mat diff;
    cv::subtract(orig32, rend32, diff);
    cv::Mat diff_sq;
    cv::multiply(diff, diff, diff_sq);
    std::vector<cv::Mat> ch(3);
    cv::split(diff_sq, ch);
    cv::Mat sum_sq = ch[0] + ch[1] + ch[2];
    cv::Mat de;
    cv::sqrt(sum_sq, de);

    return cv::mean(de, border_mask)[0];
}

} // namespace

PixelMetricsResult ComputePixelMetrics(const cv::Mat& original, const cv::Mat& rendered,
                                       const cv::Mat& coverage, const cv::Mat& shape_count,
                                       const cv::Mat& alpha_mask) {
    PixelMetricsResult r;

    bool has_mask   = !alpha_mask.empty();
    int total_valid = has_mask ? cv::countNonZero(alpha_mask) : original.rows * original.cols;
    if (total_valid == 0) return r;

    r.psnr = ComputePsnr(original, rendered, alpha_mask);
    r.ssim = ComputeSsim(original, rendered, alpha_mask);

    // Coverage: fraction of valid (opaque) pixels that are covered by SVG shapes
    if (has_mask) {
        cv::Mat effective_coverage;
        cv::bitwise_and(coverage, alpha_mask, effective_coverage);
        r.coverage = static_cast<double>(cv::countNonZero(effective_coverage)) /
                     static_cast<double>(total_valid);
    } else {
        r.coverage =
            static_cast<double>(cv::countNonZero(coverage)) / static_cast<double>(total_valid);
    }

    // Overlap: fraction of valid pixels with >1 overlapping shapes
    cv::Mat overlap_mask;
    cv::compare(shape_count, 1, overlap_mask, cv::CMP_GT);
    if (has_mask) cv::bitwise_and(overlap_mask, alpha_mask, overlap_mask);
    r.overlap =
        static_cast<double>(cv::countNonZero(overlap_mask)) / static_cast<double>(total_valid);

    auto de        = ComputeDeltaE(original, rendered, alpha_mask);
    r.delta_e_mean = de.mean;
    r.delta_e_p95  = de.p95;
    r.delta_e_max  = de.max_val;

    r.border_delta_e_mean = ComputeBorderDeltaE(original, rendered, coverage, alpha_mask);

    return r;
}

} // namespace ChromaPrint3D::eval
