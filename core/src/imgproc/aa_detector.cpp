#include "imgproc/aa_detector.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace ChromaPrint3D::detail {

namespace {

float LabDist(const cv::Vec3f& a, const cv::Vec3f& b) {
    float dL = a[0] - b[0];
    float da = a[1] - b[1];
    float db = a[2] - b[2];
    return std::sqrt(dL * dL + da * da + db * db);
}

bool IsBoundaryPixel(const cv::Mat& labels, int r, int c) {
    int lbl = labels.at<int>(r, c);
    if (lbl < 0) return false;
    const int rows            = labels.rows;
    const int cols            = labels.cols;
    static constexpr int dx[] = {-1, 1, 0, 0};
    static constexpr int dy[] = {0, 0, -1, 1};
    for (int d = 0; d < 4; ++d) {
        int nr = r + dy[d];
        int nc = c + dx[d];
        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
        int nb = labels.at<int>(nr, nc);
        if (nb >= 0 && nb != lbl) return true;
    }
    return false;
}

} // namespace

AAMap DetectAAPixels(const cv::Mat& lab, const cv::Mat& labels,
                     const std::vector<cv::Vec3f>& centers_lab, const AADetectConfig& cfg) {
    AAMap result;
    const int rows = lab.rows;
    const int cols = lab.cols;

    result.is_aa   = cv::Mat::zeros(rows, cols, CV_8UC1);
    result.alpha   = cv::Mat::zeros(rows, cols, CV_32FC1);
    result.label_a = cv::Mat::zeros(rows, cols, CV_32SC1);
    result.label_b = cv::Mat::zeros(rows, cols, CV_32SC1);

    if (lab.empty() || labels.empty() || centers_lab.empty()) return result;

    const auto start = std::chrono::steady_clock::now();
    const float tol  = std::max(1.0f, cfg.tolerance);
    int detected     = 0;

    static constexpr int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
    static constexpr int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};

    for (int r = 0; r < rows; ++r) {
        const int* lrow        = labels.ptr<int>(r);
        const cv::Vec3f* lbrow = lab.ptr<cv::Vec3f>(r);

        for (int c = 0; c < cols; ++c) {
            int my_label = lrow[c];
            if (my_label < 0) continue;
            if (!IsBoundaryPixel(labels, r, c)) continue;

            const cv::Vec3f& px = lbrow[c];
            const cv::Vec3f& ca =
                centers_lab[std::min(my_label, static_cast<int>(centers_lab.size()) - 1)];

            float best_residual = tol + 1.0f;
            float best_alpha    = 0.0f;
            int best_nb_label   = -1;

            for (int d = 0; d < 8; ++d) {
                int nr = r + dy[d];
                int nc = c + dx[d];
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                int nb = labels.at<int>(nr, nc);
                if (nb < 0 || nb == my_label) continue;

                const cv::Vec3f& cb =
                    centers_lab[std::min(nb, static_cast<int>(centers_lab.size()) - 1)];

                float dist_ab = LabDist(ca, cb);
                if (dist_ab < 3.0f) continue;

                // Solve for alpha: px ≈ alpha * ca + (1 - alpha) * cb
                cv::Vec3f diff_ab = ca - cb;
                cv::Vec3f diff_px = px - cb;
                float dot_num =
                    diff_px[0] * diff_ab[0] + diff_px[1] * diff_ab[1] + diff_px[2] * diff_ab[2];
                float dot_den =
                    diff_ab[0] * diff_ab[0] + diff_ab[1] * diff_ab[1] + diff_ab[2] * diff_ab[2];
                if (dot_den < 1e-6f) continue;

                float a = dot_num / dot_den;
                a       = std::clamp(a, 0.0f, 1.0f);

                cv::Vec3f predicted = cb + diff_ab * a;
                float residual      = LabDist(px, predicted);

                if (residual < best_residual) {
                    best_residual = residual;
                    best_alpha    = a;
                    best_nb_label = nb;
                }
            }

            if (best_residual <= tol && best_nb_label >= 0 && best_alpha > 0.05f &&
                best_alpha < 0.95f) {
                result.is_aa.at<uint8_t>(r, c) = 255;
                result.alpha.at<float>(r, c)   = best_alpha;
                result.label_a.at<int>(r, c)   = my_label;
                result.label_b.at<int>(r, c)   = best_nb_label;
                ++detected;
            }
        }
    }

    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    spdlog::info("DetectAAPixels done: detected={}/{} boundary pixels, tolerance={:.1f}, "
                 "elapsed_ms={:.2f}",
                 detected, rows * cols, tol, elapsed_ms);
    return result;
}

} // namespace ChromaPrint3D::detail
