#include "bindings.h"

#include "imgproc.h"
#include "pybind_utils.h"

namespace py = pybind11;

namespace ChromaPrint3D::pybind {

void BindImgProc(py::module_& m) {
    py::class_<ImgProcResult>(m, "ImgProcResult", "Image processing output.")
        .def_readonly("name", &ImgProcResult::name, "Source name.")
        .def_readonly("width", &ImgProcResult::width, "Image width in pixels.")
        .def_readonly("height", &ImgProcResult::height, "Image height in pixels.")
        .def_property_readonly(
            "rgb",
            [](const ImgProcResult& result) { return pybind_utils::MatToNumpyAuto(result.rgb); },
            "Linear RGB image as numpy array (H, W, 3).")
        .def_property_readonly(
            "lab",
            [](const ImgProcResult& result) { return pybind_utils::MatToNumpyAuto(result.lab); },
            "Lab image as numpy array (H, W, 3).")
        .def_property_readonly(
            "mask",
            [](const ImgProcResult& result) { return pybind_utils::MatToNumpyAuto(result.mask); },
            "Mask image as numpy array (H, W).");

    py::class_<ImgProc>(m, "ImgProc", "Image preprocessing pipeline.")
        .def(py::init<>(), "Create a processor with default parameters.")
        .def_readwrite("request_scale", &ImgProc::request_scale, "Requested scale factor.")
        .def_readwrite("max_width", &ImgProc::max_width, "Max width, 0 to disable.")
        .def_readwrite("max_height", &ImgProc::max_height, "Max height, 0 to disable.")
        .def_readwrite("use_alpha_mask", &ImgProc::use_alpha_mask, "Use alpha mask if present.")
        .def_readwrite("alpha_threshold", &ImgProc::alpha_threshold,
                       "Alpha <= threshold is masked.")
        .def_readwrite("upsample_method", &ImgProc::upsample_method, "Resize method for upsample.")
        .def_readwrite("downsample_method", &ImgProc::downsample_method,
                       "Resize method for downsample.")
        .def_readwrite("denoise_method", &ImgProc::denoise_method, "Denoise method.")
        .def_readwrite("denoise_kernel", &ImgProc::denoise_kernel, "Median kernel size.")
        .def_readwrite("bilateral_diameter", &ImgProc::bilateral_diameter, "Bilateral diameter.")
        .def_readwrite("bilateral_sigma_color", &ImgProc::bilateral_sigma_color,
                       "Bilateral sigma color.")
        .def_readwrite("bilateral_sigma_space", &ImgProc::bilateral_sigma_space,
                       "Bilateral sigma space.")
        .def("run", &ImgProc::Run, py::arg("path"), "Run pipeline on an image path.");
}

} // namespace ChromaPrint3D::pybind
