#include "bindings.h"

#include "calib.h"

namespace py = pybind11;

namespace ChromaPrint3D::pybind {

void BindCalib(py::module_& m) {
    py::class_<CalibrationRecipeSpec>(m, "CalibrationRecipeSpec", "Calibration recipe spec.")
        .def(py::init<>(), "Create default recipe spec.")
        .def_readwrite("num_channels", &CalibrationRecipeSpec::num_channels, "Number of channels.")
        .def_readwrite("color_layers", &CalibrationRecipeSpec::color_layers, "Number of layers.")
        .def_readwrite("layer_order", &CalibrationRecipeSpec::layer_order, "Layer order.")
        .def("num_recipes", &CalibrationRecipeSpec::NumRecipes, "Total recipe count.")
        .def("is_supported", &CalibrationRecipeSpec::IsSupported, "Check if spec is supported.")
        .def("recipe_at", &CalibrationRecipeSpec::RecipeAt, py::arg("index"),
             "Generate recipe at index.");

    py::class_<CalibrationFiducialSpec>(m, "CalibrationFiducialSpec", "Fiducial layout spec.")
        .def(py::init<>(), "Create default fiducial spec.")
        .def_readwrite("offset_factor", &CalibrationFiducialSpec::offset_factor, "Offset factor.")
        .def_readwrite("main_d_factor", &CalibrationFiducialSpec::main_d_factor,
                       "Main hole diameter factor.")
        .def_readwrite("tag_d_factor", &CalibrationFiducialSpec::tag_d_factor,
                       "Tag hole diameter factor.")
        .def_readwrite("tag_dx_factor", &CalibrationFiducialSpec::tag_dx_factor, "Tag dx factor.")
        .def_readwrite("tag_dy_factor", &CalibrationFiducialSpec::tag_dy_factor, "Tag dy factor.");

    py::class_<CalibrationBoardLayout>(m, "CalibrationBoardLayout", "Board layout settings.")
        .def(py::init<>(), "Create default board layout.")
        .def_readwrite("line_width_mm", &CalibrationBoardLayout::line_width_mm, "Line width in mm.")
        .def_readwrite("resolution_scale", &CalibrationBoardLayout::resolution_scale,
                       "Internal resolution scale.")
        .def_readwrite("tile_factor", &CalibrationBoardLayout::tile_factor, "Tile size factor.")
        .def_readwrite("gap_factor", &CalibrationBoardLayout::gap_factor, "Gap size factor.")
        .def_readwrite("margin_factor", &CalibrationBoardLayout::margin_factor, "Margin factor.")
        .def_readwrite("fiducial", &CalibrationBoardLayout::fiducial, "Fiducial spec.");

    py::class_<CalibrationBoardConfig>(m, "CalibrationBoardConfig", "Board configuration.")
        .def(py::init<>(), "Create default board config.")
        .def_readwrite("recipe", &CalibrationBoardConfig::recipe, "Recipe spec.")
        .def_readwrite("base_layers", &CalibrationBoardConfig::base_layers, "Base layer count.")
        .def_readwrite("base_channel_idx", &CalibrationBoardConfig::base_channel_idx,
                       "Base channel index.")
        .def_readwrite("layer_height_mm", &CalibrationBoardConfig::layer_height_mm,
                       "Layer height in mm.")
        .def_readwrite("palette", &CalibrationBoardConfig::palette, "Channel palette.")
        .def_readwrite("layout", &CalibrationBoardConfig::layout, "Board layout.")
        .def("num_recipes", &CalibrationBoardConfig::NumRecipes, "Total recipe count.")
        .def("is_supported", &CalibrationBoardConfig::IsSupported, "Check if supported.")
        .def("has_valid_palette", &CalibrationBoardConfig::HasValidPalette, "Check palette size.")
        .def_static("for_channels", &CalibrationBoardConfig::ForChannels, py::arg("num_channels"),
                    "Create config for a channel count.");

    py::class_<CalibrationBoardMeta>(m, "CalibrationBoardMeta", "Board metadata.")
        .def(py::init<>(), "Create empty board metadata.")
        .def_readwrite("name", &CalibrationBoardMeta::name, "Board name.")
        .def_readwrite("config", &CalibrationBoardMeta::config, "Board configuration.")
        .def_readwrite("grid_rows", &CalibrationBoardMeta::grid_rows, "Grid rows.")
        .def_readwrite("grid_cols", &CalibrationBoardMeta::grid_cols, "Grid columns.")
        .def_readwrite("patch_recipe_idx", &CalibrationBoardMeta::patch_recipe_idx,
                       "Patch recipe indices.")
        .def("num_patches", &CalibrationBoardMeta::NumPatches, "Total patch count.")
        .def("save_to_json", &CalibrationBoardMeta::SaveToJson, py::arg("path"),
             "Save metadata to JSON.")
        .def_static("load_from_json", &CalibrationBoardMeta::LoadFromJson, py::arg("path"),
                    "Load metadata from JSON.");

    m.def("BuildCalibrationBoardMeta", &BuildCalibrationBoardMeta, py::arg("cfg"),
          "Build board metadata from configuration.");
    m.def("GenCalibrationBoard", &GenCalibrationBoard, py::arg("cfg"), py::arg("board_path"),
          py::arg("meta_path"), "Generate calibration board and metadata files.");
    m.def("GenColorDBFromImage",
          py::overload_cast<const std::string&, const CalibrationBoardMeta&>(&GenColorDBFromImage),
          py::arg("image_path"), py::arg("meta"), "Generate ColorDB from image and metadata.");
    m.def("GenColorDBFromImage",
          py::overload_cast<const std::string&, const std::string&>(&GenColorDBFromImage),
          py::arg("image_path"), py::arg("json_path"),
          "Generate ColorDB from image and metadata JSON.");
}

} // namespace ChromaPrint3D::pybind
