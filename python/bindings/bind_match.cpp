#include "bindings.h"

#include "match.h"
#include "pybind_utils.h"

#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace ChromaPrint3D::pybind {

void BindMatch(py::module_& m) {
    py::class_<MatchConfig>(m, "MatchConfig", "Matching configuration.")
        .def(py::init<>(), "Create default match config.")
        .def_readwrite("k_candidates", &MatchConfig::k_candidates,
                       "Number of candidates (k <= 1 for nearest).")
        .def_readwrite("color_space", &MatchConfig::color_space, "Color space for matching.");

    py::class_<RecipeMap>(m, "RecipeMap", "Matched recipe map.")
        .def(py::init<>(), "Create an empty recipe map.")
        .def_readwrite("name", &RecipeMap::name, "Recipe map name.")
        .def_readwrite("width", &RecipeMap::width, "Image width.")
        .def_readwrite("height", &RecipeMap::height, "Image height.")
        .def_readwrite("color_layers", &RecipeMap::color_layers, "Recipe layer count.")
        .def_readwrite("num_channels", &RecipeMap::num_channels, "Channel count.")
        .def_readwrite("layer_order", &RecipeMap::layer_order, "Layer order.")
        .def_readwrite("recipes", &RecipeMap::recipes, "Flattened recipe data (H*W*L).")
        .def_readwrite("mask", &RecipeMap::mask, "Flattened mask data (H*W).")
        .def_readwrite("mapped_color", &RecipeMap::mapped_color, "Mapped Lab colors (H*W).")
        .def(
            "recipe_at",
            [](const RecipeMap& map, int r, int c) {
                const uint8_t* recipe = map.RecipeAt(r, c);
                if (!recipe || map.color_layers <= 0) { return std::vector<uint8_t>{}; }
                return std::vector<uint8_t>(recipe, recipe + map.color_layers);
            },
            py::arg("row"), py::arg("col"), "Recipe at (row, col).")
        .def(
            "mask_at",
            [](const RecipeMap& map, int r, int c) {
                const uint8_t* value = map.MaskAt(r, c);
                return value ? *value : static_cast<uint8_t>(0);
            },
            py::arg("row"), py::arg("col"), "Mask value at (row, col).")
        .def("color_at", &RecipeMap::ColorAt, py::arg("row"), py::arg("col"),
             "Mapped Lab color at (row, col).")
        .def(
            "to_bgr_image",
            [](const RecipeMap& map, uint8_t b, uint8_t g, uint8_t r) {
                return pybind_utils::MatToNumpyAuto(map.ToBgrImage(b, g, r));
            },
            py::arg("background_b") = 0, py::arg("background_g") = 0,
            py::arg("background_r") = 0,
            "Render mapped colors to a BGR image.")
        .def_static("MatchFromImage", &RecipeMap::MatchFromImage, py::arg("img"), py::arg("db"),
                    py::arg("cfg") = MatchConfig{},
                    "Match an image to recipes using a ColorDB.");
}

} // namespace ChromaPrint3D::pybind
