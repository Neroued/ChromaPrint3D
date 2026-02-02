#pragma once

#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ChromaPrint3D::pybind_utils {

namespace py = pybind11;

template <typename T>
inline py::array MatToNumpy(const cv::Mat& mat) {
    if (mat.empty()) { return py::array(); }
    cv::Mat contiguous = mat.isContinuous() ? mat : mat.clone();
    const int channels = contiguous.channels();
    std::vector<py::ssize_t> shape;
    if (channels == 1) {
        shape = {static_cast<py::ssize_t>(contiguous.rows),
                 static_cast<py::ssize_t>(contiguous.cols)};
    } else {
        shape = {static_cast<py::ssize_t>(contiguous.rows),
                 static_cast<py::ssize_t>(contiguous.cols), static_cast<py::ssize_t>(channels)};
    }
    py::array array(py::dtype::of<T>(), shape);
    const size_t bytes =
        static_cast<size_t>(contiguous.total()) * static_cast<size_t>(channels) * sizeof(T);
    std::memcpy(array.mutable_data(), contiguous.data, bytes);
    return array;
}

inline py::array MatToNumpyAuto(const cv::Mat& mat) {
    if (mat.empty()) { return py::array(); }
    switch (mat.depth()) {
    case CV_8U:
        return MatToNumpy<uint8_t>(mat);
    case CV_32F:
        return MatToNumpy<float>(mat);
    default:
        throw std::runtime_error("Unsupported cv::Mat depth for numpy export");
    }
}

} // namespace ChromaPrint3D::pybind_utils
