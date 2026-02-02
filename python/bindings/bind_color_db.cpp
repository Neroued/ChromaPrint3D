#include "bindings.h"

#include "colorDB.h"

#include <cstddef>
#include <vector>

namespace py = pybind11;

namespace ChromaPrint3D::pybind {
namespace {

Entry CopyEntry(const Entry& entry) { return entry; }

} // namespace

void BindColorDB(py::module_& m) {
    py::class_<Channel>(m, "Channel", "Channel metadata.")
        .def(py::init<>(), "Create a channel with default metadata.")
        .def_readwrite("color", &Channel::color, "Color name, e.g. 'Cyan'.")
        .def_readwrite("material", &Channel::material, "Material name, e.g. 'PLA'.");

    py::class_<Entry>(m, "Entry", "A palette entry with Lab color and recipe.")
        .def(py::init<>(), "Create an empty entry.")
        .def_readwrite("lab", &Entry::lab, "Lab color of the entry.")
        .def_readwrite("recipe", &Entry::recipe, "Recipe as channel indices.")
        .def("color_layers", &Entry::ColorLayers, "Number of layers in the recipe.");

    py::class_<ColorDB>(m, "ColorDB", "Color database for mapping colors to recipes.")
        .def(py::init<>(), "Create an empty color database.")
        .def_readwrite("name", &ColorDB::name, "Database name.")
        .def_readwrite("max_color_layers", &ColorDB::max_color_layers, "Maximum recipe layers.")
        .def_readwrite("base_layers", &ColorDB::base_layers, "Base layer count.")
        .def_readwrite("base_channel_idx", &ColorDB::base_channel_idx, "Base channel index.")
        .def_readwrite("layer_height_mm", &ColorDB::layer_height_mm, "Layer height in mm.")
        .def_readwrite("line_width_mm", &ColorDB::line_width_mm, "Line width in mm.")
        .def_readwrite("layer_order", &ColorDB::layer_order, "Layer stacking order.")
        .def_readwrite("palette", &ColorDB::palette, "Channel palette.")
        .def_readwrite("entries", &ColorDB::entries, "Palette entries.")
        .def("num_channels", &ColorDB::NumChannels, "Number of channels in palette.")
        .def_static("load_from_json", &ColorDB::LoadFromJson, py::arg("path"),
                    "Load a ColorDB from a JSON file.")
        .def("save_to_json", &ColorDB::SaveToJson, py::arg("path"),
             "Save this ColorDB to a JSON file.")
        .def(
            "nearest_entry_lab",
            [](const ColorDB& db, const Lab& target) { return CopyEntry(db.NearestEntry(target)); },
            py::arg("target"), "Find nearest entry by Lab color.")
        .def(
            "nearest_entry_rgb",
            [](const ColorDB& db, const Rgb& target) { return CopyEntry(db.NearestEntry(target)); },
            py::arg("target"), "Find nearest entry by RGB color.")
        .def(
            "nearest_entries_lab",
            [](const ColorDB& db, const Lab& target, std::size_t k) {
                auto entries = db.NearestEntries(target, k);
                std::vector<Entry> out;
                out.reserve(entries.size());
                for (const Entry* entry : entries) {
                    if (entry) { out.push_back(*entry); }
                }
                return out;
            },
            py::arg("target"), py::arg("k"), "Find k nearest entries by Lab color.")
        .def(
            "nearest_entries_rgb",
            [](const ColorDB& db, const Rgb& target, std::size_t k) {
                auto entries = db.NearestEntries(target, k);
                std::vector<Entry> out;
                out.reserve(entries.size());
                for (const Entry* entry : entries) {
                    if (entry) { out.push_back(*entry); }
                }
                return out;
            },
            py::arg("target"), py::arg("k"), "Find k nearest entries by RGB color.");
}

} // namespace ChromaPrint3D::pybind
