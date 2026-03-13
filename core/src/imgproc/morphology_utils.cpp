#include "imgproc/morphology_utils.h"

#include <cstdint>

namespace ChromaPrint3D::detail {

cv::Mat ZhangSuenThinning(const cv::Mat& binary_mask) {
    cv::Mat img;
    binary_mask.convertTo(img, CV_8UC1);
    img /= 255;

    cv::Mat prev = cv::Mat::zeros(img.size(), CV_8UC1);
    cv::Mat marker;

    while (true) {
        marker = cv::Mat::zeros(img.size(), CV_8UC1);
        for (int r = 1; r < img.rows - 1; ++r) {
            for (int c = 1; c < img.cols - 1; ++c) {
                if (img.at<uint8_t>(r, c) == 0) continue;
                uint8_t p2 = img.at<uint8_t>(r - 1, c);
                uint8_t p3 = img.at<uint8_t>(r - 1, c + 1);
                uint8_t p4 = img.at<uint8_t>(r, c + 1);
                uint8_t p5 = img.at<uint8_t>(r + 1, c + 1);
                uint8_t p6 = img.at<uint8_t>(r + 1, c);
                uint8_t p7 = img.at<uint8_t>(r + 1, c - 1);
                uint8_t p8 = img.at<uint8_t>(r, c - 1);
                uint8_t p9 = img.at<uint8_t>(r - 1, c - 1);

                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (B < 2 || B > 6) continue;

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + (p4 == 0 && p5 == 1) +
                        (p5 == 0 && p6 == 1) + (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                if (A != 1) continue;

                if (p2 * p4 * p6 != 0) continue;
                if (p4 * p6 * p8 != 0) continue;
                marker.at<uint8_t>(r, c) = 1;
            }
        }
        img -= marker;

        marker = cv::Mat::zeros(img.size(), CV_8UC1);
        for (int r = 1; r < img.rows - 1; ++r) {
            for (int c = 1; c < img.cols - 1; ++c) {
                if (img.at<uint8_t>(r, c) == 0) continue;
                uint8_t p2 = img.at<uint8_t>(r - 1, c);
                uint8_t p3 = img.at<uint8_t>(r - 1, c + 1);
                uint8_t p4 = img.at<uint8_t>(r, c + 1);
                uint8_t p5 = img.at<uint8_t>(r + 1, c + 1);
                uint8_t p6 = img.at<uint8_t>(r + 1, c);
                uint8_t p7 = img.at<uint8_t>(r + 1, c - 1);
                uint8_t p8 = img.at<uint8_t>(r, c - 1);
                uint8_t p9 = img.at<uint8_t>(r - 1, c - 1);

                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (B < 2 || B > 6) continue;

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + (p4 == 0 && p5 == 1) +
                        (p5 == 0 && p6 == 1) + (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                if (A != 1) continue;

                if (p2 * p4 * p8 != 0) continue;
                if (p2 * p6 * p8 != 0) continue;
                marker.at<uint8_t>(r, c) = 1;
            }
        }
        img -= marker;

        if (cv::countNonZero(img != prev) == 0) break;
        img.copyTo(prev);
    }

    img *= 255;
    return img;
}

} // namespace ChromaPrint3D::detail
