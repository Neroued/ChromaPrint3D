#include "imgproc/vectorize_potrace_pipeline.h"

#include "detail/cv_utils.h"
#include "imgproc/bezier.h"
#include "imgproc/boundary_graph.h"
#include "imgproc/contour_assembly.h"
#include "imgproc/coverage_guard.h"
#include "imgproc/curve_fitting.h"
#include "imgproc/subpixel_refine.h"
#include "imgproc/aa_detector.h"
#include "imgproc/potrace_adapter.h"
#include "imgproc/thin_line_vectorizer.h"
#include "imgproc/svg_writer.h"
#include "imgproc/vectorize_preprocess.h"
#include "match/slic_segmenter.h"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

namespace ChromaPrint3D::detail {

namespace {

constexpr int kSlicIterations       = 10;
constexpr float kSlicMinRegionRatio = 0.25f;

struct SegmentationResult {
    cv::Mat labels;
    cv::Mat lab;
    std::vector<cv::Vec3f> centers_lab;
};

SegmentationResult SegmentBinary(const cv::Mat& bgr, const cv::Mat& lab) {
    SegmentationResult out;
    out.lab = lab;

    cv::Mat gray, bw;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    out.labels = cv::Mat(bgr.rows, bgr.cols, CV_32SC1);
    out.centers_lab.resize(2, cv::Vec3f(0, 0, 0));
    std::array<cv::Vec3f, 2> sums{cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0)};
    std::array<int, 2> counts{0, 0};

    for (int r = 0; r < bgr.rows; ++r) {
        const uint8_t* bw_row    = bw.ptr<uint8_t>(r);
        const cv::Vec3f* lab_row = lab.ptr<cv::Vec3f>(r);
        int* out_row             = out.labels.ptr<int>(r);
        for (int c = 0; c < bgr.cols; ++c) {
            int lid    = (bw_row[c] > 0) ? 1 : 0;
            out_row[c] = lid;
            sums[lid] += lab_row[c];
            counts[lid]++;
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (counts[i] > 0) out.centers_lab[i] = sums[i] * (1.0f / static_cast<float>(counts[i]));
    }
    return out;
}

SegmentationResult SegmentMultiColor(const cv::Mat& lab, int num_colors, int slic_region_size,
                                     float slic_compactness, const cv::Mat& edge_map,
                                     float edge_sensitivity) {
    SegmentationResult out;
    out.lab    = lab;
    out.labels = cv::Mat(lab.rows, lab.cols, CV_32SC1, cv::Scalar(0));

    SlicConfig slic_cfg;
    slic_cfg.region_size      = std::max(0, slic_region_size);
    slic_cfg.compactness      = std::max(0.001f, slic_compactness);
    slic_cfg.iterations       = kSlicIterations;
    slic_cfg.min_region_ratio = kSlicMinRegionRatio;
    slic_cfg.edge_map         = edge_map;
    slic_cfg.edge_sensitivity = edge_sensitivity;

    auto slic  = SegmentBySlic(lab, cv::Mat(), slic_cfg);
    int num_sp = static_cast<int>(slic.centers.size());
    spdlog::debug(
        "Vectorize segmentation (SLIC): requested_colors={}, region_size={}, superpixels={}",
        num_colors, slic_region_size, num_sp);

    if (num_sp < num_colors) {
        spdlog::warn("Vectorize segmentation fallback: superpixels={} < requested_colors={}, "
                     "switching to pixel kmeans",
                     num_sp, num_colors);
        cv::Mat samples = lab.reshape(1, lab.rows * lab.cols);
        samples.convertTo(samples, CV_32F);
        int K = std::clamp(num_colors, 2, std::max(2, samples.rows));

        cv::Mat km_labels, km_centers;
        cv::kmeans(samples, K, km_labels,
                   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.2), 5,
                   cv::KMEANS_PP_CENTERS, km_centers);

        out.centers_lab.resize(K);
        for (int k = 0; k < K; ++k) {
            out.centers_lab[k] = {km_centers.at<float>(k, 0), km_centers.at<float>(k, 1),
                                  km_centers.at<float>(k, 2)};
        }
        out.labels = km_labels.reshape(1, lab.rows).clone();
        spdlog::debug("Vectorize segmentation (pixel kmeans) done: K={}", K);
        return out;
    }

    cv::Mat sp_samples(num_sp, 3, CV_32FC1);
    for (int i = 0; i < num_sp; ++i) {
        sp_samples.at<float>(i, 0) = slic.centers[i][0];
        sp_samples.at<float>(i, 1) = slic.centers[i][1];
        sp_samples.at<float>(i, 2) = slic.centers[i][2];
    }
    int K = std::clamp(num_colors, 2, num_sp);

    cv::Mat km_labels, km_centers;
    cv::kmeans(sp_samples, K, km_labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.2), 5,
               cv::KMEANS_PP_CENTERS, km_centers);

    for (int r = 0; r < lab.rows; ++r) {
        const int* slic_row = slic.labels.ptr<int>(r);
        int* out_row        = out.labels.ptr<int>(r);
        for (int c = 0; c < lab.cols; ++c) {
            int sp_id = slic_row[c];
            if (sp_id >= 0 && sp_id < num_sp) {
                out_row[c] = km_labels.at<int>(sp_id, 0);
            } else {
                out_row[c] = -1;
            }
        }
    }

    out.centers_lab.resize(K);
    for (int k = 0; k < K; ++k) {
        out.centers_lab[k] = {km_centers.at<float>(k, 0), km_centers.at<float>(k, 1),
                              km_centers.at<float>(k, 2)};
    }
    spdlog::debug("Vectorize segmentation (SLIC+kmeans) done: K={}", K);
    return out;
}

cv::Mat ComputeEdgeMap(const cv::Mat& bgr) {
    cv::Mat bgr_float;
    bgr.convertTo(bgr_float, CV_32F, 1.0 / 255.0);
    cv::Mat lab;
    cv::cvtColor(bgr_float, lab, cv::COLOR_BGR2Lab);

    cv::Mat channels[3];
    cv::split(lab, channels);

    cv::Mat mag_max = cv::Mat::zeros(bgr.size(), CV_32FC1);
    for (int ch = 0; ch < 3; ++ch) {
        cv::Mat gx, gy, mag;
        cv::Sobel(channels[ch], gx, CV_32F, 1, 0, 3);
        cv::Sobel(channels[ch], gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, mag);
        mag_max = cv::max(mag_max, mag);
    }

    double max_val = 0.0;
    cv::minMaxLoc(mag_max, nullptr, &max_val);
    if (max_val > 0.0) mag_max *= (1.0 / max_val);
    return mag_max;
}

void RefineLabelsBoundary(cv::Mat& labels, const cv::Mat& unsmoothed_lab,
                          const std::vector<cv::Vec3f>& centers_lab, int passes) {
    if (labels.empty() || unsmoothed_lab.empty() || centers_lab.empty()) return;

    const int h          = labels.rows;
    const int w          = labels.cols;
    const int num_labels = static_cast<int>(centers_lab.size());

    constexpr int kDr8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    constexpr int kDc8[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    for (int pass = 0; pass < passes; ++pass) {
        cv::Mat snapshot = labels.clone();
        int changed      = 0;

        for (int r = 0; r < h; ++r) {
            const int* snap_row      = snapshot.ptr<int>(r);
            const cv::Vec3f* lab_row = unsmoothed_lab.ptr<cv::Vec3f>(r);
            int* out_row             = labels.ptr<int>(r);

            for (int c = 0; c < w; ++c) {
                const int lid = snap_row[c];
                if (lid < 0 || lid >= num_labels) continue;

                bool has_different_neighbor = false;
                int neighbor_set[8];
                int neighbor_count = 0;

                for (int k = 0; k < 8; ++k) {
                    int nr = r + kDr8[k];
                    int nc = c + kDc8[k];
                    if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
                    int nl = snapshot.at<int>(nr, nc);
                    if (nl < 0 || nl >= num_labels || nl == lid) continue;
                    has_different_neighbor = true;
                    bool already           = false;
                    for (int j = 0; j < neighbor_count; ++j) {
                        if (neighbor_set[j] == nl) {
                            already = true;
                            break;
                        }
                    }
                    if (!already && neighbor_count < 8) { neighbor_set[neighbor_count++] = nl; }
                }
                if (!has_different_neighbor) continue;

                const cv::Vec3f& pixel = lab_row[c];
                const cv::Vec3f& cur   = centers_lab[lid];
                float d_current        = (pixel[0] - cur[0]) * (pixel[0] - cur[0]) +
                                  (pixel[1] - cur[1]) * (pixel[1] - cur[1]) +
                                  (pixel[2] - cur[2]) * (pixel[2] - cur[2]);

                int best_label  = lid;
                float best_dist = d_current;
                for (int j = 0; j < neighbor_count; ++j) {
                    const cv::Vec3f& cand = centers_lab[neighbor_set[j]];
                    float d               = (pixel[0] - cand[0]) * (pixel[0] - cand[0]) +
                              (pixel[1] - cand[1]) * (pixel[1] - cand[1]) +
                              (pixel[2] - cand[2]) * (pixel[2] - cand[2]);
                    if (d < best_dist) {
                        best_dist  = d;
                        best_label = neighbor_set[j];
                    }
                }

                if (best_label != lid) {
                    out_row[c] = best_label;
                    ++changed;
                }
            }
        }

        spdlog::debug("RefineLabelsBoundary pass {}/{}: changed={}", pass + 1, passes, changed);
        if (changed == 0) break;
    }
}

void MergeSmallComponents(cv::Mat& labels, const cv::Mat& lab, std::vector<cv::Vec3f>& centers_lab,
                          int min_region_area, float max_merge_color_dist) {
    if (min_region_area <= 1 || labels.empty()) return;

    const int h                   = labels.rows;
    const int w                   = labels.cols;
    constexpr int kMaxMergeRounds = 3;
    constexpr int dr[4]           = {1, -1, 0, 0};
    constexpr int dc[4]           = {0, 0, 1, -1};

    std::queue<cv::Point> q;
    std::vector<cv::Point> component;
    component.reserve(1024);

    for (int round = 0; round < kMaxMergeRounds; ++round) {
        cv::Mat visited(h, w, CV_8UC1, cv::Scalar(0));
        int merged_count = 0;

        for (int sr = 0; sr < h; ++sr) {
            for (int sc = 0; sc < w; ++sc) {
                if (visited.at<uint8_t>(sr, sc) != 0) continue;

                const int label0            = labels.at<int>(sr, sc);
                visited.at<uint8_t>(sr, sc) = 1;
                if (label0 < 0) continue;
                q.push({sc, sr});
                component.clear();
                component.push_back({sc, sr});

                std::unordered_map<int, int> border_hist;
                cv::Vec3f mean_lab(0, 0, 0);

                while (!q.empty()) {
                    cv::Point p = q.front();
                    q.pop();
                    mean_lab += lab.at<cv::Vec3f>(p.y, p.x);

                    for (int k = 0; k < 4; ++k) {
                        int nr = p.y + dr[k];
                        int nc = p.x + dc[k];
                        if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
                        int nl = labels.at<int>(nr, nc);
                        if (nl == label0) {
                            if (visited.at<uint8_t>(nr, nc) == 0) {
                                visited.at<uint8_t>(nr, nc) = 1;
                                q.push({nc, nr});
                                component.push_back({nc, nr});
                            }
                        } else if (nl >= 0) {
                            border_hist[nl]++;
                        }
                    }
                }

                if (static_cast<int>(component.size()) >= min_region_area || border_hist.empty())
                    continue;
                mean_lab *= (1.0f / static_cast<float>(component.size()));

                int best_label       = label0;
                int best_border_vote = -1;
                float best_dist      = std::numeric_limits<float>::max();
                for (const auto& [candidate, vote] : border_hist) {
                    if (candidate < 0 || candidate >= static_cast<int>(centers_lab.size()))
                        continue;
                    float dl = mean_lab[0] - centers_lab[candidate][0];
                    float da = mean_lab[1] - centers_lab[candidate][1];
                    float db = mean_lab[2] - centers_lab[candidate][2];
                    float d2 = dl * dl + da * da + db * db;
                    if (d2 < best_dist || (d2 == best_dist && vote > best_border_vote)) {
                        best_border_vote = vote;
                        best_dist        = d2;
                        best_label       = candidate;
                    }
                }

                if (best_label == label0) continue;
                if (best_dist > max_merge_color_dist) continue;

                for (const auto& p : component) labels.at<int>(p.y, p.x) = best_label;
                ++merged_count;
            }
        }

        spdlog::debug("MergeSmallComponents round {}/{}: merged={}", round + 1, kMaxMergeRounds,
                      merged_count);
        if (merged_count == 0) break;

        for (int lid = 0; lid < static_cast<int>(centers_lab.size()); ++lid) {
            cv::Vec3d sum(0, 0, 0);
            int count = 0;
            for (int r = 0; r < h; ++r) {
                const int* lrow          = labels.ptr<int>(r);
                const cv::Vec3f* lab_row = lab.ptr<cv::Vec3f>(r);
                for (int c = 0; c < w; ++c) {
                    if (lrow[c] == lid) {
                        sum += cv::Vec3d(lab_row[c][0], lab_row[c][1], lab_row[c][2]);
                        ++count;
                    }
                }
            }
            if (count > 0) {
                double inv = 1.0 / static_cast<double>(count);
                centers_lab[lid] =
                    cv::Vec3f(static_cast<float>(sum[0] * inv), static_cast<float>(sum[1] * inv),
                              static_cast<float>(sum[2] * inv));
            }
        }
    }
}

void MorphologicalCleanup(cv::Mat& labels, int num_labels, int close_radius) {
    if (close_radius <= 0 || labels.empty() || num_labels <= 0) return;

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(2 * close_radius + 1, 2 * close_radius + 1));

    std::vector<std::pair<int, int>> label_areas;
    std::vector<int> counts(num_labels, 0);
    for (int r = 0; r < labels.rows; ++r) {
        const int* row = labels.ptr<int>(r);
        for (int c = 0; c < labels.cols; ++c) {
            int lid = row[c];
            if (lid >= 0 && lid < num_labels) counts[lid]++;
        }
    }
    for (int i = 0; i < num_labels; ++i) {
        if (counts[i] > 0) label_areas.push_back({counts[i], i});
    }
    std::sort(label_areas.begin(), label_areas.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    cv::Mat valid_mask = (labels >= 0);
    valid_mask.convertTo(valid_mask, CV_8UC1, 255);

    for (const auto& [area, lid] : label_areas) {
        cv::Mat mask = (labels == lid);
        mask.convertTo(mask, CV_8UC1, 255);
        cv::Mat closed_mask;
        cv::morphologyEx(mask, closed_mask, cv::MORPH_CLOSE, kernel);

        cv::Mat newly_claimed;
        cv::subtract(closed_mask, mask, newly_claimed);
        cv::bitwise_and(newly_claimed, valid_mask, newly_claimed);

        labels.setTo(cv::Scalar(lid), newly_claimed > 0);
    }
}

int CompactLabels(cv::Mat& labels, std::vector<cv::Vec3f>& centers_lab) {
    int max_label = static_cast<int>(centers_lab.size());
    if (labels.empty() || max_label <= 0) return 0;

    std::vector<int> remap(max_label, -1);
    int next = 0;
    for (int r = 0; r < labels.rows; ++r) {
        int* row = labels.ptr<int>(r);
        for (int c = 0; c < labels.cols; ++c) {
            int lid = row[c];
            if (lid < 0 || lid >= max_label) continue;
            if (remap[lid] < 0) remap[lid] = next++;
            row[c] = remap[lid];
        }
    }

    std::vector<cv::Vec3f> compact(next, cv::Vec3f(0, 0, 0));
    for (int i = 0; i < max_label; ++i) {
        if (remap[i] >= 0) compact[remap[i]] = centers_lab[i];
    }
    centers_lab.swap(compact);
    return next;
}

std::vector<Rgb> ComputePalette(const cv::Mat& bgr, const cv::Mat& labels, int num_labels) {
    std::vector<Rgb> palette(std::max(0, num_labels), Rgb(0, 0, 0));
    if (num_labels <= 0) return palette;

    std::vector<std::array<double, 3>> sums(num_labels, {0.0, 0.0, 0.0});
    std::vector<int> counts(num_labels, 0);

    for (int r = 0; r < bgr.rows; ++r) {
        const cv::Vec3b* brow = bgr.ptr<cv::Vec3b>(r);
        const int* lrow       = labels.ptr<int>(r);
        for (int c = 0; c < bgr.cols; ++c) {
            int lid = lrow[c];
            if (lid < 0 || lid >= num_labels) continue;
            sums[lid][0] += brow[c][2] / 255.0; // R
            sums[lid][1] += brow[c][1] / 255.0; // G
            sums[lid][2] += brow[c][0] / 255.0; // B
            counts[lid]++;
        }
    }

    for (int i = 0; i < num_labels; ++i) {
        if (counts[i] <= 0) continue;
        float r    = static_cast<float>(sums[i][0] / counts[i]);
        float g    = static_cast<float>(sums[i][1] / counts[i]);
        float b    = static_cast<float>(sums[i][2] / counts[i]);
        palette[i] = Rgb(SrgbToLinear(r), SrgbToLinear(g), SrgbToLinear(b));
    }
    return palette;
}

// ── Auto color count estimation ──────────────────────────────────────────────
//
// Selects K that minimizes: reconstruction_error + fragmentation_penalty + complexity_cost.
// Uses dual sampling (pixel sample for clustering, proxy image for spatial analysis).

int EstimateOptimalColors(const cv::Mat& bgr) {
    constexpr int kTargetSamples            = 30000;
    constexpr int kProxyShortEdge           = 300;
    constexpr float kAchromaticP90Threshold = 8.0f;
    constexpr int kCandidates[]             = {2, 3, 4, 6, 8, 12, 16, 24};
    constexpr int kNumCandidates            = 8;
    constexpr float kTinyAreaFrac           = 0.002f;

    constexpr float kW_meanDE   = 1.5f;
    constexpr float kW_p95DE    = 0.5f;
    constexpr float kW_tinyRate = 20.0f;
    constexpr float kW_compDens = 20.0f;
    constexpr float kW_logK     = 3.0f;

    // ── 1. Dual sampling ─────────────────────────────────────────────────────

    const int total_px    = bgr.rows * bgr.cols;
    const float grid_step = std::sqrt(static_cast<float>(std::max(1, total_px)) / kTargetSamples);
    const int row_step    = std::max(1, static_cast<int>(grid_step));
    const int col_step    = std::max(1, static_cast<int>(grid_step));

    int n_samples = 0;
    for (int r = 0; r < bgr.rows; r += row_step)
        for (int c = 0; c < bgr.cols; c += col_step) ++n_samples;

    cv::Mat sample_bgr(n_samples, 1, CV_8UC3);
    {
        int idx = 0;
        for (int r = 0; r < bgr.rows; r += row_step) {
            const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(r);
            for (int c = 0; c < bgr.cols; c += col_step)
                sample_bgr.at<cv::Vec3b>(idx++, 0) = row[c];
        }
    }
    cv::Mat sample_lab8;
    cv::cvtColor(sample_bgr, sample_lab8, cv::COLOR_BGR2Lab);

    const int short_edge = std::min(bgr.rows, bgr.cols);
    cv::Mat proxy;
    if (short_edge > kProxyShortEdge) {
        const float s = static_cast<float>(kProxyShortEdge) / short_edge;
        cv::resize(bgr, proxy, cv::Size(), s, s, cv::INTER_AREA);
    } else {
        proxy = bgr;
    }
    cv::Mat proxy_lab8;
    cv::cvtColor(proxy, proxy_lab8, cv::COLOR_BGR2Lab);
    const int proxy_area = proxy.rows * proxy.cols;

    // ── 2. Achromatic detection (p90 chroma in LAB) ──────────────────────────

    std::vector<float> chromas(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        const cv::Vec3b p = sample_lab8.at<cv::Vec3b>(i, 0);
        const float a     = static_cast<float>(p[1]) - 128.0f;
        const float b     = static_cast<float>(p[2]) - 128.0f;
        chromas[i]        = std::sqrt(a * a + b * b);
    }
    const int p90_idx = std::max(0, static_cast<int>(0.90f * (n_samples - 1)));
    std::nth_element(chromas.begin(), chromas.begin() + p90_idx, chromas.end());
    const bool achromatic = (chromas[p90_idx] < kAchromaticP90Threshold);

    // ── 3. Convert to CIELAB float (L: 0-100, a/b: -128..127) ───────────────

    const int ch = achromatic ? 1 : 3;

    cv::Mat km_samples(n_samples, ch, CV_32F);
    for (int i = 0; i < n_samples; ++i) {
        const cv::Vec3b p          = sample_lab8.at<cv::Vec3b>(i, 0);
        km_samples.at<float>(i, 0) = p[0] * (100.0f / 255.0f);
        if (!achromatic) {
            km_samples.at<float>(i, 1) = static_cast<float>(p[1]) - 128.0f;
            km_samples.at<float>(i, 2) = static_cast<float>(p[2]) - 128.0f;
        }
    }

    cv::Mat proxy_cielab(proxy_area, ch, CV_32F);
    for (int r = 0; r < proxy.rows; ++r) {
        const cv::Vec3b* row = proxy_lab8.ptr<cv::Vec3b>(r);
        for (int c = 0; c < proxy.cols; ++c) {
            const int idx                  = r * proxy.cols + c;
            const cv::Vec3b p              = row[c];
            proxy_cielab.at<float>(idx, 0) = p[0] * (100.0f / 255.0f);
            if (!achromatic) {
                proxy_cielab.at<float>(idx, 1) = static_cast<float>(p[1]) - 128.0f;
                proxy_cielab.at<float>(idx, 2) = static_cast<float>(p[2]) - 128.0f;
            }
        }
    }

    const float tiny_px_threshold = proxy_area * kTinyAreaFrac;

    // ── 4. Evaluate each candidate K ─────────────────────────────────────────

    int best_k       = 16;
    float best_score = std::numeric_limits<float>::max();

    for (int ci = 0; ci < kNumCandidates; ++ci) {
        const int K = kCandidates[ci];
        if (K > n_samples) break;

        cv::Mat km_labels, km_centers;
        cv::kmeans(km_samples, K, km_labels,
                   cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.5), 3,
                   cv::KMEANS_PP_CENTERS, km_centers);

        // Assign proxy pixels to nearest center and compute per-pixel Delta E
        cv::Mat proxy_labels(proxy.rows, proxy.cols, CV_32SC1);
        std::vector<float> dE_vec(proxy_area);

        for (int i = 0; i < proxy_area; ++i) {
            float min_sq = std::numeric_limits<float>::max();
            int lbl      = 0;
            for (int k = 0; k < km_centers.rows; ++k) {
                float sq = 0.0f;
                for (int d = 0; d < ch; ++d) {
                    const float diff = proxy_cielab.at<float>(i, d) - km_centers.at<float>(k, d);
                    sq += diff * diff;
                }
                if (sq < min_sq) {
                    min_sq = sq;
                    lbl    = k;
                }
            }
            proxy_labels.at<int>(i / proxy.cols, i % proxy.cols) = lbl;
            dE_vec[i]                                            = std::sqrt(min_sq);
        }

        // Reconstruction error: mean and p95 Delta E
        float sum_dE = 0.0f;
        for (float d : dE_vec) sum_dE += d;
        const float mean_dE = sum_dE / std::max(1, proxy_area);

        const int p95 = std::max(0, static_cast<int>(0.95f * (proxy_area - 1)));
        std::nth_element(dE_vec.begin(), dE_vec.begin() + p95, dE_vec.end());
        const float p95_dE = dE_vec[p95];

        // Fragmentation: per-label connected components
        int total_comp = 0;
        int tiny_comp  = 0;
        for (int label = 0; label < km_centers.rows; ++label) {
            cv::Mat mask;
            cv::compare(proxy_labels, label, mask, cv::CMP_EQ);
            if (cv::countNonZero(mask) == 0) continue;

            cv::Mat cc_labels;
            int n_cc = cv::connectedComponents(mask, cc_labels, 8, CV_32S) - 1;
            total_comp += n_cc;

            std::vector<int> areas(n_cc + 1, 0);
            for (int r = 0; r < cc_labels.rows; ++r) {
                const int* row = cc_labels.ptr<int>(r);
                for (int c2 = 0; c2 < cc_labels.cols; ++c2)
                    if (row[c2] > 0) areas[row[c2]]++;
            }
            for (int cc_id = 1; cc_id <= n_cc; ++cc_id)
                if (static_cast<float>(areas[cc_id]) < tiny_px_threshold) ++tiny_comp;
        }

        const float tiny_rate =
            (total_comp > 0) ? static_cast<float>(tiny_comp) / total_comp : 0.0f;
        const float comp_density = static_cast<float>(total_comp) / std::max(1, proxy_area);
        const float log2K        = std::log2(static_cast<float>(K));

        const float score = kW_meanDE * mean_dE + kW_p95DE * p95_dE + kW_tinyRate * tiny_rate +
                            kW_compDens * comp_density + kW_logK * log2K;

        spdlog::debug("AutoColor K={:2d}: dE_mean={:.2f} dE_p95={:.2f} tiny={:.3f} "
                      "comp_dens={:.5f} log2K={:.2f} => score={:.2f}",
                      K, mean_dE, p95_dE, tiny_rate, comp_density, log2K, score);

        if (score < best_score) {
            best_score = score;
            best_k     = K;
        }
    }

    spdlog::info("AutoColor: achromatic={}, selected K={} (score={:.2f})", achromatic, best_k,
                 best_score);
    return best_k;
}

} // namespace

VectorizerResult VectorizePotracePipeline(const cv::Mat& bgr, const VectorizerConfig& cfg,
                                          const cv::Mat& opaque_mask) {
    const auto pipeline_start = std::chrono::steady_clock::now();
    const int resolved_colors = (cfg.num_colors == 0) ? EstimateOptimalColors(bgr) : cfg.num_colors;

    spdlog::info("VectorizePotracePipeline start: input={}x{}, num_colors={}{}, "
                 "min_region_area={}, curve_fit_error={:.2f}, contour_simplify={:.2f}, "
                 "svg_stroke={}, coverage_fix={}, max_working_pixels={}",
                 bgr.cols, bgr.rows, resolved_colors, cfg.num_colors == 0 ? " (auto)" : "",
                 cfg.min_region_area, cfg.curve_fit_error, cfg.contour_simplify,
                 cfg.svg_enable_stroke, cfg.enable_coverage_fix, cfg.max_working_pixels);
    const bool multicolor = resolved_colors > 2;
    auto preproc =
        PreprocessForVectorize(bgr, multicolor, cfg.smoothing_spatial, cfg.smoothing_color,
                               cfg.upscale_short_edge, cfg.max_working_pixels);
    cv::Mat working    = preproc.bgr;
    cv::Mat unsmoothed = preproc.unsmoothed_bgr;
    const float scale  = preproc.scale;
    const bool scaled  = std::abs(scale - 1.0f) > 1e-6f;

    cv::Mat working_mask = opaque_mask;
    if (scaled && !opaque_mask.empty()) {
        cv::resize(opaque_mask, working_mask, working.size(), 0, 0, cv::INTER_NEAREST);
    }
    spdlog::debug("Vectorize preprocess done: working={}x{}, scale={:.3f}, mask_present={}",
                  working.cols, working.rows, scale, !working_mask.empty());

    cv::Mat edge_map;
    if (multicolor && !unsmoothed.empty()) {
        edge_map = ComputeEdgeMap(unsmoothed);
        spdlog::debug("Vectorize edge map computed: size={}x{}", edge_map.cols, edge_map.rows);
    }

    cv::Mat lab = BgrToLab(working);
    SegmentationResult seg =
        multicolor ? SegmentMultiColor(lab, resolved_colors, cfg.slic_region_size,
                                       cfg.slic_compactness, edge_map, cfg.edge_sensitivity)
                   : SegmentBinary(working, lab);
    if (seg.labels.empty()) {
        spdlog::error("Vectorize segmentation failed: empty labels");
        throw std::runtime_error("VectorizePotracePipeline: segmentation failed");
    }
    spdlog::info("Vectorize segmentation completed: mode={}, centers={}, label_map={}x{}",
                 multicolor ? "multicolor" : "binary", seg.centers_lab.size(), seg.labels.cols,
                 seg.labels.rows);

    if (!working_mask.empty()) {
        if (working_mask.type() != CV_8UC1 || working_mask.size() != seg.labels.size()) {
            spdlog::error(
                "Vectorize mask invalid: expected type=CV_8UC1 size={}x{}, got type={} size={}x{}",
                seg.labels.cols, seg.labels.rows, working_mask.type(), working_mask.cols,
                working_mask.rows);
            throw std::runtime_error("VectorizePotracePipeline: invalid opaque mask");
        }
        cv::Mat transparent;
        cv::compare(working_mask, 0, transparent, cv::CMP_EQ);
        const int transparent_px = cv::countNonZero(transparent);
        seg.labels.setTo(cv::Scalar(-1), transparent);
        spdlog::debug("Vectorize transparent mask applied: transparent_pixels={}", transparent_px);
    }

    cv::Mat unsmoothed_lab;
    if (multicolor && !unsmoothed.empty()) { unsmoothed_lab = BgrToLab(unsmoothed); }

    if (multicolor && cfg.refine_passes > 0 && !unsmoothed_lab.empty() &&
        !seg.centers_lab.empty()) {
        RefineLabelsBoundary(seg.labels, unsmoothed_lab, seg.centers_lab, cfg.refine_passes);
        spdlog::info("Vectorize label refinement applied: passes={}", cfg.refine_passes);
    }

    MergeSmallComponents(seg.labels, seg.lab, seg.centers_lab, std::max(2, cfg.min_region_area),
                         cfg.max_merge_color_dist);
    if (multicolor) {
        MorphologicalCleanup(seg.labels, static_cast<int>(seg.centers_lab.size()), 1);
    }
    int num_labels = CompactLabels(seg.labels, seg.centers_lab);
    auto palette   = ComputePalette(working, seg.labels, num_labels);
    spdlog::info("Vectorize labels compacted: num_labels={}, palette_size={}", num_labels,
                 palette.size());

    float effective_curve_fit_error  = cfg.curve_fit_error;
    float effective_contour_simplify = cfg.contour_simplify;
    if (cfg.detail_level >= 0.0f) {
        static const VectorizerConfig kDefaults;
        float dl = std::clamp(cfg.detail_level, 0.0f, 1.0f);
        if (cfg.curve_fit_error == kDefaults.curve_fit_error)
            effective_curve_fit_error = 2.0f - 1.7f * dl;
        if (cfg.contour_simplify == kDefaults.contour_simplify)
            effective_contour_simplify = 0.8f - 0.6f * dl;
        spdlog::info("detail_level={:.2f}: derived curve_fit_error={:.2f}, "
                     "contour_simplify={:.2f}",
                     dl, effective_curve_fit_error, effective_contour_simplify);
    }

    const float trace_eps =
        std::max(0.2f, std::clamp(effective_contour_simplify * 0.45f + 0.2f, 0.2f, 2.0f));
    const int turdsize        = std::max(0, static_cast<int>(std::lround(trace_eps * 0.5f)));
    const double opttolerance = std::clamp(static_cast<double>(trace_eps), 0.2, 2.0);
    spdlog::debug("Vectorize trace params: trace_eps={:.3f}, turdsize={}, opttolerance={:.3f}",
                  trace_eps, turdsize, opttolerance);

    std::vector<VectorizedShape> shapes;

    if (multicolor && num_labels > 2) {
        spdlog::info("Vectorize contour mode: BoundaryGraph+CurveFit");
        auto boundary_graph = BuildBoundaryGraph(seg.labels);
        spdlog::debug("BoundaryGraph built: nodes={}, edges={}", boundary_graph.nodes.size(),
                      boundary_graph.edges.size());

        if (cfg.enable_subpixel_refine) {
            const cv::Mat& refine_lab =
                unsmoothed_lab.empty() ? (unsmoothed_lab = BgrToLab(working)) : unsmoothed_lab;
            SubpixelRefineConfig sp_cfg;
            sp_cfg.max_displacement = cfg.subpixel_max_displacement;

            if (cfg.enable_antialias_detect) {
                AADetectConfig aa_cfg;
                aa_cfg.tolerance = cfg.aa_tolerance;
                auto aa_map      = DetectAAPixels(refine_lab, seg.labels, seg.centers_lab, aa_cfg);
                RefineEdgesSubpixelAA(boundary_graph, refine_lab, aa_map.is_aa, aa_map.alpha,
                                      sp_cfg);
            } else {
                RefineEdgesSubpixel(boundary_graph, refine_lab, sp_cfg);
            }
        }

        CurveFitConfig fit_cfg;
        fit_cfg.error_threshold            = std::clamp(effective_curve_fit_error, 0.05f, 10.0f);
        fit_cfg.corner_angle_threshold_deg = std::clamp(cfg.corner_angle_threshold, 60.0f, 179.0f);
        auto smooth_cfg                    = ContourSmoothFromLevel(cfg.smoothness);
        shapes =
            AssembleContoursFromGraph(boundary_graph, num_labels, palette, cfg.min_contour_area,
                                      cfg.min_hole_area, &fit_cfg, smooth_cfg);
        spdlog::info("BoundaryGraph contour assembly done: shapes={}", shapes.size());

        if (cfg.merge_segment_tolerance > 0.0f) {
            int merged_total = 0;
            for (auto& shape : shapes) {
                for (auto& contour : shape.contours) {
                    int before = static_cast<int>(contour.segments.size());
                    MergeNearLinearSegments(contour.segments, cfg.merge_segment_tolerance);
                    merged_total += before - static_cast<int>(contour.segments.size());
                }
            }
            if (merged_total > 0) {
                spdlog::info("MergeNearLinearSegments: removed {} segments", merged_total);
            }
        }

        // Potrace fallback for labels that BoundaryGraph failed to produce shapes for.
        // Determine covered labels by checking pixel coverage against shapes.
        std::vector<double> label_pixel_count(num_labels, 0.0);
        std::vector<double> label_shape_area(num_labels, 0.0);
        for (int r = 0; r < seg.labels.rows; ++r) {
            const int* lrow = seg.labels.ptr<int>(r);
            for (int c = 0; c < seg.labels.cols; ++c) {
                int lid = lrow[c];
                if (lid >= 0 && lid < num_labels) label_pixel_count[lid] += 1.0;
            }
        }
        for (const auto& s : shapes) {
            if (s.is_stroke) continue;
            for (int rid = 0; rid < num_labels; ++rid) {
                if (rid < static_cast<int>(palette.size())) {
                    uint8_t sr, sg, sb, pr, pg, pb;
                    s.color.ToRgb255(sr, sg, sb);
                    palette[rid].ToRgb255(pr, pg, pb);
                    if (sr == pr && sg == pg && sb == pb) { label_shape_area[rid] += s.area; }
                }
            }
        }
        std::vector<bool> label_covered(num_labels, false);
        for (int rid = 0; rid < num_labels; ++rid) {
            if (label_pixel_count[rid] < 1.0) {
                label_covered[rid] = true;
            } else if (label_shape_area[rid] > label_pixel_count[rid] * 0.1) {
                label_covered[rid] = true;
            }
        }
        int uncovered_labels = 0;
        for (int rid = 0; rid < num_labels; ++rid) {
            if (!label_covered[rid]) ++uncovered_labels;
        }
        if (uncovered_labels > 0) {
            spdlog::warn(
                "Vectorize fallback triggered: uncovered_labels={} (BoundaryGraph -> Potrace)",
                uncovered_labels);
        }
        int fallback_labels     = 0;
        int fallback_shapes_add = 0;
        for (int rid = 0; rid < num_labels; ++rid) {
            if (label_covered[rid]) continue;
            ++fallback_labels;
            cv::Mat mask = (seg.labels == rid);
            mask.convertTo(mask, CV_8UC1, 255);
            if (cv::countNonZero(mask) <= 0) continue;

            auto traced = TraceMaskWithPotraceBezier(mask, turdsize, opttolerance);
            for (auto& g : traced) {
                if (g.area < static_cast<double>(cfg.min_contour_area)) continue;
                VectorizedShape shape;
                shape.color = palette[rid];
                shape.area  = g.area;
                shape.contours.push_back(std::move(g.outer));
                for (auto& hole : g.holes) {
                    double hole_area = std::abs(BezierContourSignedArea(hole));
                    if (hole_area < static_cast<double>(cfg.min_hole_area)) continue;
                    shape.contours.push_back(std::move(hole));
                }
                if (!shape.contours.empty()) {
                    shapes.push_back(std::move(shape));
                    ++fallback_shapes_add;
                }
            }
        }
        if (fallback_labels > 0) {
            spdlog::info("Vectorize fallback completed: labels={}, shapes_added={}",
                         fallback_labels, fallback_shapes_add);
        }
    } else {
        spdlog::info("Vectorize contour mode: per-label Potrace");
        int labels_traced       = 0;
        int dilate_retry_count  = 0;
        int direct_shapes_added = 0;
        for (int rid = 0; rid < num_labels; ++rid) {
            cv::Mat mask = (seg.labels == rid);
            mask.convertTo(mask, CV_8UC1, 255);
            int px = cv::countNonZero(mask);
            if (px <= 0) continue;
            ++labels_traced;

            auto traced = TraceMaskWithPotraceBezier(mask, turdsize, opttolerance);
            if (traced.empty()) {
                ++dilate_retry_count;
                cv::Mat dilated;
                cv::dilate(mask, dilated,
                           cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
                traced = TraceMaskWithPotraceBezier(dilated, std::max(0, turdsize - 1),
                                                    std::max(0.2, opttolerance * 0.8));
            }

            for (auto& g : traced) {
                if (g.area < static_cast<double>(cfg.min_contour_area)) continue;
                VectorizedShape shape;
                shape.color = palette[rid];
                shape.area  = g.area;
                shape.contours.push_back(std::move(g.outer));
                for (auto& hole : g.holes) {
                    double hole_area = std::abs(BezierContourSignedArea(hole));
                    if (hole_area < static_cast<double>(cfg.min_hole_area)) continue;
                    shape.contours.push_back(std::move(hole));
                }
                if (!shape.contours.empty()) {
                    shapes.push_back(std::move(shape));
                    ++direct_shapes_added;
                }
            }
        }
        if (dilate_retry_count > 0) {
            spdlog::warn("Vectorize Potrace retry with dilation: labels_retried={}",
                         dilate_retry_count);
        }
        spdlog::info("Vectorize per-label Potrace done: labels_traced={}, shapes_added={}",
                     labels_traced, direct_shapes_added);
    }

    // Thin-line enhancement: detect narrow sub-regions and add stroke paths
    if (cfg.svg_enable_stroke && multicolor && num_labels > 1) {
        const float thin_radius = std::clamp(cfg.thin_line_max_radius, 0.1f, 50.0f);
        int labels_with_thin    = 0;
        int stroke_added        = 0;
        for (int rid = 0; rid < num_labels; ++rid) {
            cv::Mat mask = (seg.labels == rid);
            mask.convertTo(mask, CV_8UC1, 255);
            if (cv::countNonZero(mask) <= 0) continue;

            cv::Mat thin = DetectThinRegion(mask, thin_radius);
            if (cv::countNonZero(thin) < 3) continue;
            ++labels_with_thin;

            cv::Mat dist;
            cv::distanceTransform(mask, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
            cv::Mat skel = ZhangSuenThinning(thin);
            if (cv::countNonZero(skel) < 3) continue;

            auto strokes = ExtractStrokePaths(skel, dist, palette[rid], 3.0f);
            stroke_added += static_cast<int>(strokes.size());
            for (auto& s : strokes) shapes.push_back(std::move(s));
        }
        spdlog::info("Vectorize thin-line enhancement: labels={}, strokes_added={}",
                     labels_with_thin, stroke_added);
    } else if (cfg.svg_enable_stroke) {
        spdlog::debug("Vectorize thin-line enhancement skipped: multicolor={}, num_labels={}",
                      multicolor, num_labels);
    }

    std::sort(shapes.begin(), shapes.end(), [](const auto& a, const auto& b) {
        if (a.is_stroke != b.is_stroke) return !a.is_stroke;
        return a.area > b.area;
    });

    if (cfg.enable_coverage_fix) {
        float effective_coverage_ratio = cfg.min_coverage_ratio;
        if (multicolor && num_labels > 2) {
            effective_coverage_ratio = std::min(cfg.min_coverage_ratio, 0.995f);
        }
        const auto before = shapes.size();
        ApplyCoverageGuard(shapes, seg.labels, palette, effective_coverage_ratio, trace_eps,
                           std::max(1.0f, cfg.min_contour_area * 0.5f));
        const auto added = shapes.size() >= before ? (shapes.size() - before) : 0;
        spdlog::info("Vectorize coverage guard applied: added_shapes={}", added);
    }

    if (scaled) {
        const float inv = 1.0f / scale;
        for (auto& shape : shapes) {
            for (auto& contour : shape.contours) {
                for (auto& s : contour.segments) {
                    s.p0 = s.p0 * inv;
                    s.p1 = s.p1 * inv;
                    s.p2 = s.p2 * inv;
                    s.p3 = s.p3 * inv;
                }
            }
        }
        spdlog::debug("Vectorize output rescaled by inverse factor={:.4f}", inv);
    }

    VectorizerResult result;
    result.width               = bgr.cols;
    result.height              = bgr.rows;
    result.num_shapes          = static_cast<int>(shapes.size());
    result.resolved_num_colors = resolved_colors;
    result.palette             = std::move(palette);
    result.svg_content =
        WriteSvg(shapes, bgr.cols, bgr.rows, cfg.svg_enable_stroke, cfg.svg_stroke_width);
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pipeline_start)
            .count();
    spdlog::info("VectorizePotracePipeline completed: elapsed_ms={:.2f}, width={}, height={}, "
                 "num_shapes={}, palette_size={}, svg_bytes={}",
                 elapsed_ms, result.width, result.height, result.num_shapes, result.palette.size(),
                 result.svg_content.size());
    return result;
}

} // namespace ChromaPrint3D::detail
