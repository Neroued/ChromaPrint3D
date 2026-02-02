#include "bindings.h"

#include "common.h"
#include "vec3.h"

#include <string>

namespace py = pybind11;

namespace ChromaPrint3D::pybind {
namespace {

std::string Vec3iRepr(const Vec3i& v) {
    return "<Vec3i x=" + std::to_string(v.x) + " y=" + std::to_string(v.y) +
           " z=" + std::to_string(v.z) + ">";
}

std::string Vec3fRepr(const Vec3f& v) {
    return "<Vec3f x=" + std::to_string(v.x) + " y=" + std::to_string(v.y) +
           " z=" + std::to_string(v.z) + ">";
}

} // namespace

void BindCommon(py::module_& m) {
    py::enum_<ResizeMethod>(m, "ResizeMethod", "Resize sampling method.")
        .value("Nearest", ResizeMethod::Nearest)
        .value("Area", ResizeMethod::Area)
        .value("Linear", ResizeMethod::Linear)
        .value("Cubic", ResizeMethod::Cubic)
        .export_values();

    py::enum_<DenoiseMethod>(m, "DenoiseMethod", "Denoising algorithm.")
        .value("None", DenoiseMethod::None)
        .value("Bilateral", DenoiseMethod::Bilateral)
        .value("Median", DenoiseMethod::Median)
        .export_values();

    py::enum_<LayerOrder>(m, "LayerOrder", "Stacking order of layers.")
        .value("Top2Bottom", LayerOrder::Top2Bottom)
        .value("Bottom2Top", LayerOrder::Bottom2Top)
        .export_values();

    py::enum_<ColorSpace>(m, "ColorSpace", "Color space selection.")
        .value("Lab", ColorSpace::Lab)
        .value("Rgb", ColorSpace::Rgb);

    py::class_<Vec3i>(m, "Vec3i", "3D vector with integer components.")
        .def(py::init<>(), "Create a zero vector.")
        .def(py::init<int, int, int>(), py::arg("x"), py::arg("y"), py::arg("z"),
             "Create a vector from x, y, z.")
        .def_readwrite("x", &Vec3i::x, "X component.")
        .def_readwrite("y", &Vec3i::y, "Y component.")
        .def_readwrite("z", &Vec3i::z, "Z component.")
        .def("dot", &Vec3i::Dot, py::arg("other"), "Dot product with another Vec3i.")
        .def("length_squared", &Vec3i::LengthSquared, "Squared length.")
        .def("__repr__", &Vec3iRepr);

    py::class_<Vec3f>(m, "Vec3f", "3D vector with float components.")
        .def(py::init<>(), "Create a zero vector.")
        .def(py::init<float, float, float>(), py::arg("x"), py::arg("y"), py::arg("z"),
             "Create a vector from x, y, z.")
        .def_readwrite("x", &Vec3f::x, "X component.")
        .def_readwrite("y", &Vec3f::y, "Y component.")
        .def_readwrite("z", &Vec3f::z, "Z component.")
        .def("dot", &Vec3f::Dot, py::arg("other"), "Dot product with another Vec3f.")
        .def("length_squared", &Vec3f::LengthSquared, "Squared length.")
        .def("length", &Vec3f::Length, "Vector length.")
        .def("normalized", &Vec3f::Normalized, "Return a normalized copy.")
        .def("is_finite", &Vec3f::IsFinite, "Return True if all components are finite.")
        .def("nearly_equal", &Vec3f::NearlyEqual, py::arg("other"), py::arg("eps") = 1e-5f,
             "Return True if components are within eps.")
        .def("__repr__", &Vec3fRepr);

    py::class_<Rgb, Vec3f>(m, "Rgb", "RGB color in linear space.")
        .def(py::init<>(), "Create black RGB (0,0,0).")
        .def(py::init<float, float, float>(), py::arg("r"), py::arg("g"), py::arg("b"),
             "Create RGB from linear components.")
        .def_property(
            "r", [](const Rgb& v) { return v.r(); }, [](Rgb& v, float value) { v.r() = value; },
            "Red component in linear RGB.")
        .def_property(
            "g", [](const Rgb& v) { return v.g(); }, [](Rgb& v, float value) { v.g() = value; },
            "Green component in linear RGB.")
        .def_property(
            "b", [](const Rgb& v) { return v.b(); }, [](Rgb& v, float value) { v.b() = value; },
            "Blue component in linear RGB.")
        .def("to_lab", &Rgb::ToLab, "Convert to Lab color space.")
        .def(
            "to_rgb255",
            [](const Rgb& v) {
                uint8_t r = 0, g = 0, b = 0;
                v.ToRgb255(r, g, b);
                return py::make_tuple(r, g, b);
            },
            "Convert to 8-bit sRGB tuple (r,g,b).")
        .def_static("from_lab", &Rgb::FromLab, py::arg("lab"), "Create RGB from Lab color.")
        .def_static("from_rgb255", &Rgb::FromRgb255, py::arg("r"), py::arg("g"), py::arg("b"),
                    "Create RGB from 8-bit sRGB values.");

    py::class_<Lab, Vec3f>(m, "Lab", "CIELAB color.")
        .def(py::init<>(), "Create Lab (0,0,0).")
        .def(py::init<float, float, float>(), py::arg("l"), py::arg("a"), py::arg("b"),
             "Create Lab from components.")
        .def_property(
            "l", [](const Lab& v) { return v.l(); }, [](Lab& v, float value) { v.l() = value; },
            "L component.")
        .def_property(
            "a", [](const Lab& v) { return v.a(); }, [](Lab& v, float value) { v.a() = value; },
            "a component.")
        .def_property(
            "b", [](const Lab& v) { return v.b(); }, [](Lab& v, float value) { v.b() = value; },
            "b component.")
        .def("to_rgb", &Lab::ToRgb, "Convert to linear RGB.")
        .def_static("from_rgb", &Lab::FromRgb, py::arg("rgb"), "Create Lab from RGB.")
        .def_static("delta_e76", &Lab::DeltaE76, py::arg("lab1"), py::arg("lab2"),
                    "Compute Delta E (CIE76) distance.");
}

} // namespace ChromaPrint3D::pybind
