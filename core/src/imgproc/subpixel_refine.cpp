#include "imgproc/subpixel_refine.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace ChromaPrint3D::detail {

namespace {

cv::Vec3f BilinearSampleLab(const cv::Mat& lab, float fx, float fy) {
    const int max_c = lab.cols - 1;
    const int max_r = lab.rows - 1;

    int x0  = static_cast<int>(std::floor(fx));
    int y0  = static_cast<int>(std::floor(fy));
    float u = fx - static_cast<float>(x0);
    float v = fy - static_cast<float>(y0);

    x0     = std::clamp(x0, 0, max_c);
    y0     = std::clamp(y0, 0, max_r);
    int x1 = std::min(x0 + 1, max_c);
    int y1 = std::min(y0 + 1, max_r);

    const cv::Vec3f& p00 = lab.at<cv::Vec3f>(y0, x0);
    const cv::Vec3f& p01 = lab.at<cv::Vec3f>(y0, x1);
    const cv::Vec3f& p10 = lab.at<cv::Vec3f>(y1, x0);
    const cv::Vec3f& p11 = lab.at<cv::Vec3f>(y1, x1);

    float w00 = (1.0f - u) * (1.0f - v);
    float w01 = u * (1.0f - v);
    float w10 = (1.0f - u) * v;
    float w11 = u * v;

    return p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11;
}

float LabDeltaE(const cv::Vec3f& a, const cv::Vec3f& b) {
    float dL = a[0] - b[0];
    float da = a[1] - b[1];
    float db = a[2] - b[2];
    return std::sqrt(dL * dL + da * da + db * db);
}

} // namespace

void RefineEdgesSubpixel(BoundaryGraph& graph, const cv::Mat& lab,
                         const SubpixelRefineConfig& cfg) {
    if (lab.empty() || lab.type() != CV_32FC3 || graph.edges.empty()) {
        spdlog::debug("RefineEdgesSubpixel skipped: lab_empty={}, edges={}", lab.empty(),
                      graph.edges.size());
        return;
    }

    const auto start = std::chrono::steady_clock::now();

    const int num_samples = std::max(3, cfg.num_samples | 1); // ensure odd
    const float range     = std::max(0.1f, cfg.sample_range);
    const float step      = 2.0f * range / static_cast<float>(num_samples - 1);
    const float max_disp  = std::max(0.0f, cfg.max_displacement);
    const float min_grad  = std::max(0.0f, cfg.min_gradient);
    const int tw          = std::max(1, cfg.tangent_window);

    std::vector<float> offsets(static_cast<size_t>(num_samples));
    for (int j = 0; j < num_samples; ++j) {
        offsets[static_cast<size_t>(j)] = -range + static_cast<float>(j) * step;
    }

    const int num_grads = num_samples - 1;
    std::vector<float> grad_centers(static_cast<size_t>(num_grads));
    for (int j = 0; j < num_grads; ++j) {
        grad_centers[static_cast<size_t>(j)] =
            0.5f * (offsets[static_cast<size_t>(j)] + offsets[static_cast<size_t>(j + 1)]);
    }

    int total_points  = 0;
    int shifted_count = 0;
    int skipped_short = 0;

    std::vector<cv::Vec3f> samples(static_cast<size_t>(num_samples));
    std::vector<float> grads(static_cast<size_t>(num_grads));

    for (auto& edge : graph.edges) {
        const int n = static_cast<int>(edge.points.size());
        if (n < 5) {
            ++skipped_short;
            continue;
        }

        for (int i = 1; i < n - 1; ++i) {
            ++total_points;
            const Vec2f& p = edge.points[static_cast<size_t>(i)];

            // 1. Local tangent from neighbour window
            int k         = std::min(tw, std::min(i, n - 1 - i));
            Vec2f ahead   = edge.points[static_cast<size_t>(i + k)];
            Vec2f behind  = edge.points[static_cast<size_t>(i - k)];
            Vec2f tangent = ahead - behind;
            float tlen    = tangent.Length();
            if (tlen < 1e-6f) continue;
            tangent = tangent / tlen;

            // 2. Normal (90-deg CCW rotation of tangent)
            Vec2f normal{-tangent.y, tangent.x};

            // 3. Sample LAB image along normal
            for (int j = 0; j < num_samples; ++j) {
                float t                         = offsets[static_cast<size_t>(j)];
                float sx                        = p.x + normal.x * t;
                float sy                        = p.y + normal.y * t;
                samples[static_cast<size_t>(j)] = BilinearSampleLab(lab, sx, sy);
            }

            // 4. Gradient profile (deltaE between consecutive samples)
            int max_j    = 0;
            float max_gv = 0.0f;
            for (int j = 0; j < num_grads; ++j) {
                float g = LabDeltaE(samples[static_cast<size_t>(j)],
                                    samples[static_cast<size_t>(j + 1)]) /
                          step;
                grads[static_cast<size_t>(j)] = g;
                if (g > max_gv) {
                    max_gv = g;
                    max_j  = j;
                }
            }

            if (max_gv < min_grad) continue;

            // 5. Parabolic interpolation around the peak
            float peak_t = grad_centers[static_cast<size_t>(max_j)];
            if (max_j > 0 && max_j < num_grads - 1) {
                float a     = grads[static_cast<size_t>(max_j - 1)];
                float b     = grads[static_cast<size_t>(max_j)];
                float c     = grads[static_cast<size_t>(max_j + 1)];
                float denom = a - 2.0f * b + c;
                if (std::abs(denom) > 1e-6f) {
                    float delta = 0.5f * (a - c) / denom;
                    peak_t += delta * step;
                }
            }

            // 6. Clamp and apply displacement
            peak_t = std::clamp(peak_t, -max_disp, max_disp);
            if (std::abs(peak_t) < 1e-4f) continue;

            edge.points[static_cast<size_t>(i)] =
                Vec2f{p.x + normal.x * peak_t, p.y + normal.y * peak_t};
            ++shifted_count;
        }
    }

    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    spdlog::info("RefineEdgesSubpixel done: edges={}, total_pts={}, shifted={}, skipped_short={}, "
                 "elapsed_ms={:.2f}",
                 graph.edges.size(), total_points, shifted_count, skipped_short, elapsed_ms);
}

void RefineEdgesSubpixelAA(BoundaryGraph& graph, const cv::Mat& lab, const cv::Mat& aa_is_aa,
                           const cv::Mat& aa_alpha, const SubpixelRefineConfig& cfg) {
    if (lab.empty() || lab.type() != CV_32FC3 || graph.edges.empty()) return;
    const bool have_aa = !aa_is_aa.empty() && !aa_alpha.empty();

    const auto start      = std::chrono::steady_clock::now();
    const int num_samples = std::max(3, cfg.num_samples | 1);
    const float range     = std::max(0.1f, cfg.sample_range);
    const float step      = 2.0f * range / static_cast<float>(num_samples - 1);
    const float max_disp  = std::max(0.0f, cfg.max_displacement);
    const float min_grad  = std::max(0.0f, cfg.min_gradient);
    const int tw          = std::max(1, cfg.tangent_window);

    std::vector<float> offsets(static_cast<size_t>(num_samples));
    for (int j = 0; j < num_samples; ++j) {
        offsets[static_cast<size_t>(j)] = -range + static_cast<float>(j) * step;
    }
    const int num_grads = num_samples - 1;
    std::vector<float> grad_centers(static_cast<size_t>(num_grads));
    for (int j = 0; j < num_grads; ++j) {
        grad_centers[static_cast<size_t>(j)] =
            0.5f * (offsets[static_cast<size_t>(j)] + offsets[static_cast<size_t>(j + 1)]);
    }

    int total_pts    = 0;
    int shifted      = 0;
    int aa_shifted   = 0;
    int grad_shifted = 0;

    std::vector<cv::Vec3f> samples(static_cast<size_t>(num_samples));
    std::vector<float> grads(static_cast<size_t>(num_grads));

    for (auto& edge : graph.edges) {
        const int n = static_cast<int>(edge.points.size());
        if (n < 5) continue;

        for (int i = 1; i < n - 1; ++i) {
            ++total_pts;
            const Vec2f& p = edge.points[static_cast<size_t>(i)];

            int k         = std::min(tw, std::min(i, n - 1 - i));
            Vec2f ahead   = edge.points[static_cast<size_t>(i + k)];
            Vec2f behind  = edge.points[static_cast<size_t>(i - k)];
            Vec2f tangent = ahead - behind;
            float tlen    = tangent.Length();
            if (tlen < 1e-6f) continue;
            tangent = tangent / tlen;
            Vec2f normal{-tangent.y, tangent.x};

            // Try AA blend-ratio positioning first
            if (have_aa) {
                int px = std::clamp(static_cast<int>(std::round(p.x)), 0, aa_is_aa.cols - 1);
                int py = std::clamp(static_cast<int>(std::round(p.y)), 0, aa_is_aa.rows - 1);
                if (aa_is_aa.at<uint8_t>(py, px) != 0) {
                    float a      = aa_alpha.at<float>(py, px);
                    float offset = (a - 0.5f) * 0.8f;
                    offset       = std::clamp(offset, -max_disp, max_disp);
                    if (std::abs(offset) > 1e-4f) {
                        edge.points[static_cast<size_t>(i)] =
                            Vec2f{p.x + normal.x * offset, p.y + normal.y * offset};
                        ++shifted;
                        ++aa_shifted;
                        continue;
                    }
                }
            }

            // Fallback: gradient peak
            for (int j = 0; j < num_samples; ++j) {
                float t                         = offsets[static_cast<size_t>(j)];
                float sx                        = p.x + normal.x * t;
                float sy                        = p.y + normal.y * t;
                samples[static_cast<size_t>(j)] = BilinearSampleLab(lab, sx, sy);
            }

            int max_j    = 0;
            float max_gv = 0.0f;
            for (int j = 0; j < num_grads; ++j) {
                float g = LabDeltaE(samples[static_cast<size_t>(j)],
                                    samples[static_cast<size_t>(j + 1)]) /
                          step;
                grads[static_cast<size_t>(j)] = g;
                if (g > max_gv) {
                    max_gv = g;
                    max_j  = j;
                }
            }
            if (max_gv < min_grad) continue;

            float peak_t = grad_centers[static_cast<size_t>(max_j)];
            if (max_j > 0 && max_j < num_grads - 1) {
                float a     = grads[static_cast<size_t>(max_j - 1)];
                float b     = grads[static_cast<size_t>(max_j)];
                float c     = grads[static_cast<size_t>(max_j + 1)];
                float denom = a - 2.0f * b + c;
                if (std::abs(denom) > 1e-6f) {
                    float delta = 0.5f * (a - c) / denom;
                    peak_t += delta * step;
                }
            }

            peak_t = std::clamp(peak_t, -max_disp, max_disp);
            if (std::abs(peak_t) < 1e-4f) continue;

            edge.points[static_cast<size_t>(i)] =
                Vec2f{p.x + normal.x * peak_t, p.y + normal.y * peak_t};
            ++shifted;
            ++grad_shifted;
        }
    }

    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    spdlog::info("RefineEdgesSubpixelAA done: total_pts={}, shifted={} (aa={}, grad={}), "
                 "elapsed_ms={:.2f}",
                 total_pts, shifted, aa_shifted, grad_shifted, elapsed_ms);
}

} // namespace ChromaPrint3D::detail
