#include "bindings.h"

#include "geo.h"

namespace py = pybind11;

namespace ChromaPrint3D::pybind {

void BindGeo(py::module_& m) {
    py::class_<VoxelGrid>(m, "VoxelGrid", "Voxel occupancy grid.")
        .def(py::init<>(), "Create an empty voxel grid.")
        .def_readwrite("width", &VoxelGrid::width, "Grid width.")
        .def_readwrite("height", &VoxelGrid::height, "Grid height.")
        .def_readwrite("num_layers", &VoxelGrid::num_layers, "Number of layers.")
        .def_readwrite("channel_idx", &VoxelGrid::channel_idx, "Channel index for this grid.")
        .def_readwrite("ooc", &VoxelGrid::ooc, "Flattened occupancy (H*W*L).")
        .def("get", &VoxelGrid::Get, py::arg("w"), py::arg("h"), py::arg("l"),
             "Get occupancy at (w, h, l).")
        .def("set", &VoxelGrid::Set, py::arg("w"), py::arg("h"), py::arg("l"), py::arg("value"),
             "Set occupancy at (w, h, l).");

    py::class_<BuildModelIRConfig>(m, "BuildModelIRConfig", "ModelIR build options.")
        .def(py::init<>(), "Create default build config.")
        .def_readwrite("flip_y", &BuildModelIRConfig::flip_y, "Flip Y axis in output.")
        .def_readwrite("base_layers", &BuildModelIRConfig::base_layers, "Override base layers.")
        .def_readwrite("double_sided", &BuildModelIRConfig::double_sided, "Build double sided.");

    py::class_<ModelIR>(m, "ModelIR", "Intermediate model representation.")
        .def(py::init<>(), "Create an empty ModelIR.")
        .def_readwrite("name", &ModelIR::name, "Model name.")
        .def_readwrite("width", &ModelIR::width, "Model width.")
        .def_readwrite("height", &ModelIR::height, "Model height.")
        .def_readwrite("color_layers", &ModelIR::color_layers, "Color layer count.")
        .def_readwrite("base_layers", &ModelIR::base_layers, "Base layer count.")
        .def_readwrite("base_channel_idx", &ModelIR::base_channel_idx, "Base channel index.")
        .def_readwrite("palette", &ModelIR::palette, "Channel palette.")
        .def_readwrite("voxel_grids", &ModelIR::voxel_grids, "Voxel grids per channel.")
        .def("num_channels", &ModelIR::NumChannels, "Number of channels in palette.")
        .def_static("Build", &ModelIR::Build, py::arg("recipe_map"), py::arg("db"),
                    py::arg("cfg") = BuildModelIRConfig{},
                    "Build a ModelIR from a RecipeMap and ColorDB.");

    py::class_<BuildMeshConfig>(m, "BuildMeshConfig", "Mesh build options.")
        .def(py::init<>(), "Create default mesh config.")
        .def_readwrite("layer_height_mm", &BuildMeshConfig::layer_height_mm, "Layer height in mm.")
        .def_readwrite("pixel_mm", &BuildMeshConfig::pixel_mm, "Pixel size in mm.");

    py::class_<Mesh>(m, "Mesh", "Triangle mesh.")
        .def(py::init<>(), "Create an empty mesh.")
        .def_readwrite("vertices", &Mesh::vertices, "Vertex list.")
        .def_readwrite("indices", &Mesh::indices, "Triangle indices.")
        .def_static("Build", &Mesh::Build, py::arg("voxel_grid"),
                    py::arg("cfg") = BuildMeshConfig{}, "Build a mesh from a voxel grid.");

    m.def("Export3mf", py::overload_cast<const std::string&, const ModelIR&>(&Export3mf),
          py::arg("path"), py::arg("model"), "Export ModelIR to 3MF.");
    m.def("Export3mf",
          py::overload_cast<const std::string&, const ModelIR&, const BuildMeshConfig&>(&Export3mf),
          py::arg("path"), py::arg("model"), py::arg("cfg"),
          "Export ModelIR to 3MF using mesh config.");
}

} // namespace ChromaPrint3D::pybind
