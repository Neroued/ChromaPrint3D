#include "edge_metrics.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace ChromaPrint3D::eval {

EdgeMetricsResult ComputeEdgeMetrics(const cv::Mat& original, const cv::Mat& rendered,
                                     int tolerance, const cv::Mat& alpha_mask) {
    EdgeMetricsResult r;

    cv::Mat gray_orig, gray_rend;
    cv::cvtColor(original, gray_orig, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rendered, gray_rend, cv::COLOR_BGR2GRAY);

    cv::Mat edges_orig, edges_rend;
    cv::Canny(gray_orig, edges_orig, 50, 150);
    cv::Canny(gray_rend, edges_rend, 50, 150);

    // Exclude edges in transparent regions
    if (!alpha_mask.empty()) {
        cv::bitwise_and(edges_orig, alpha_mask, edges_orig);
        cv::bitwise_and(edges_rend, alpha_mask, edges_rend);
    }

    int p_orig = cv::countNonZero(edges_orig);
    int p_rend = cv::countNonZero(edges_rend);

    if (p_orig == 0 && p_rend == 0) {
        r.edge_f1          = 1.0;
        r.chamfer_distance = 0.0;
        return r;
    }

    // Edge F1 with dilation tolerance
    cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * tolerance + 1, 2 * tolerance + 1));
    cv::Mat dilated_orig, dilated_rend;
    cv::dilate(edges_orig, dilated_orig, kernel);
    cv::dilate(edges_rend, dilated_rend, kernel);

    cv::Mat tp_rend_mask, tp_orig_mask;
    cv::bitwise_and(edges_rend, dilated_orig, tp_rend_mask);
    cv::bitwise_and(edges_orig, dilated_rend, tp_orig_mask);

    double tp_precision = (p_rend > 0) ? static_cast<double>(cv::countNonZero(tp_rend_mask)) /
                                             static_cast<double>(p_rend)
                                       : 0.0;
    double tp_recall    = (p_orig > 0) ? static_cast<double>(cv::countNonZero(tp_orig_mask)) /
                                          static_cast<double>(p_orig)
                                       : 0.0;

    if (tp_precision + tp_recall > 0)
        r.edge_f1 = 2.0 * tp_precision * tp_recall / (tp_precision + tp_recall);

    // Chamfer distance via distance transform
    cv::Mat not_orig, not_rend;
    cv::bitwise_not(edges_orig, not_orig);
    cv::bitwise_not(edges_rend, not_rend);

    cv::Mat dist_orig, dist_rend;
    cv::distanceTransform(not_orig, dist_orig, cv::DIST_L2, 3);
    cv::distanceTransform(not_rend, dist_rend, cv::DIST_L2, 3);

    double sum_rend_to_orig = 0;
    if (p_rend > 0) sum_rend_to_orig = cv::mean(dist_orig, edges_rend)[0];

    double sum_orig_to_rend = 0;
    if (p_orig > 0) sum_orig_to_rend = cv::mean(dist_rend, edges_orig)[0];

    r.chamfer_distance = (sum_rend_to_orig + sum_orig_to_rend) / 2.0;

    return r;
}

} // namespace ChromaPrint3D::eval
