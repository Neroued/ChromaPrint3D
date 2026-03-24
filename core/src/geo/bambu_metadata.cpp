#include "bambu_metadata.h"

#include "chromaprint3d/error.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>

namespace ChromaPrint3D {

namespace {

nlohmann::json LoadPresetJson(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) { throw IOError("Cannot open preset file: " + path); }
    try {
        return nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error& e) {
        throw FormatError("Failed to parse preset JSON: " + std::string(e.what()));
    }
}

std::tuple<int, int, int> ParseHexRGB(const std::string& hex) {
    if (hex.size() >= 7 && hex[0] == '#') {
        unsigned long val = std::strtoul(hex.c_str() + 1, nullptr, 16);
        return {static_cast<int>((val >> 16) & 0xFF), static_cast<int>((val >> 8) & 0xFF),
                static_cast<int>(val & 0xFF)};
    }
    return {255, 255, 255};
}

int RGBDistanceSq(const std::tuple<int, int, int>& a, const std::tuple<int, int, int>& b) {
    int dr = std::get<0>(a) - std::get<0>(b);
    int dg = std::get<1>(a) - std::get<1>(b);
    int db = std::get<2>(a) - std::get<2>(b);
    return dr * dr + dg * dg + db * db;
}

void PatchFlushMatrix(nlohmann::json& j, const std::vector<int>& matrix) {
    if (matrix.empty()) return;
    nlohmann::json arr = nlohmann::json::array();
    for (int v : matrix) { arr.push_back(std::to_string(v)); }
    j["flush_volumes_matrix"] = arr;
}

void PatchFilamentArrays(nlohmann::json& j, const std::vector<FilamentSlot>& filaments) {
    if (filaments.empty()) return;
    auto patch_array = [&](const char* key, auto extractor) {
        if (!j.contains(key) || !j[key].is_array()) return;
        auto& arr    = j[key];
        size_t limit = std::min(filaments.size(), arr.size());
        for (size_t i = 0; i < limit; ++i) { arr[i] = extractor(filaments[i]); }
    };
    patch_array("filament_colour", [](const FilamentSlot& s) { return s.colour; });
    patch_array("filament_multi_colour", [](const FilamentSlot& s) { return s.colour; });
    patch_array("filament_type", [](const FilamentSlot& s) { return s.type; });
    patch_array("filament_settings_id", [](const FilamentSlot& s) { return s.settings_id; });
    patch_array("filament_ids", [](const FilamentSlot& s) { return s.filament_id; });
    patch_array("filament_vendor", [](const FilamentSlot& s) { return s.vendor; });
    patch_array("nozzle_temperature",
                [](const FilamentSlot& s) { return std::to_string(s.nozzle_temp); });
    patch_array("nozzle_temperature_initial_layer",
                [](const FilamentSlot& s) { return std::to_string(s.nozzle_temp_initial); });
}

} // namespace

int MatchColorToSlot(const std::string& hex_color,
                     const std::vector<std::string>& filament_colours) {
    if (filament_colours.empty()) return 1;

    auto target  = ParseHexRGB(hex_color);
    int best_idx = 0;
    int best_d   = std::numeric_limits<int>::max();
    for (int i = 0; i < static_cast<int>(filament_colours.size()); ++i) {
        auto fc = ParseHexRGB(filament_colours[static_cast<size_t>(i)]);
        int d   = RGBDistanceSq(target, fc);
        if (d == 0) return i + 1;
        if (d < best_d) {
            best_d   = d;
            best_idx = i;
        }
    }
    return best_idx + 1;
}

std::vector<std::string> ReadFilamentColours(const std::string& preset_json_path) {
    std::vector<std::string> result;
    if (preset_json_path.empty()) return result;

    std::ifstream ifs(preset_json_path);
    if (!ifs.is_open()) {
        spdlog::warn("ReadFilamentColours: cannot open {}", preset_json_path);
        return result;
    }

    try {
        auto j = nlohmann::json::parse(ifs);
        if (j.contains("filament_colour") && j["filament_colour"].is_array()) {
            for (const auto& c : j["filament_colour"]) { result.push_back(c.get<std::string>()); }
        }
    } catch (const std::exception& e) {
        spdlog::warn("ReadFilamentColours: parse error for {}: {}", preset_json_path, e.what());
    }
    return result;
}

std::string FindPresetFile(const std::string& preset_dir, const std::string& /*printer_model*/,
                           float layer_height, NozzleSize nozzle, FaceOrientation face) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "bambu_p2s_%.2fmm_%s_%s.json",
                  static_cast<double>(layer_height), NozzleSizeTag(nozzle),
                  FaceOrientationTag(face));
    auto full_path = std::filesystem::path(preset_dir) / buf;
    if (std::filesystem::exists(full_path)) { return full_path.string(); }
    spdlog::warn("Preset file not found: {}", full_path.string());
    return {};
}

namespace {

std::string ToLowerStr(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::string DeduceFilamentType(const std::string& material) {
    if (material.empty() || material == "Default Material") return "PLA";
    std::string m = material;
    std::transform(m.begin(), m.end(), m.begin(), [](unsigned char c) { return std::toupper(c); });
    if (m.find("PETG") != std::string::npos) return "PETG";
    if (m.find("ABS") != std::string::npos) return "ABS";
    if (m.find("TPU") != std::string::npos) return "TPU";
    if (m.find("ASA") != std::string::npos) return "ASA";
    if (m.find("PA") != std::string::npos && m.find("PLA") == std::string::npos) return "PA";
    return "PLA";
}

std::string DeduceSettingsId(const std::string& filament_type) {
    if (filament_type == "PETG") return "Bambu PETG Basic @BBL P2S";
    if (filament_type == "ABS") return "Bambu ABS @BBL P2S";
    if (filament_type == "TPU") return "Bambu TPU 95A @BBL P2S";
    if (filament_type == "ASA") return "Bambu ASA @BBL P2S";
    if (filament_type == "PA") return "Bambu PA @BBL P2S";
    return "Bambu PLA Basic @BBL P2S";
}

std::string ResolveMaterialType(const std::string& material, const FilamentConfig* config) {
    if (!config) return DeduceFilamentType(material);

    std::string lower = ToLowerStr(material);
    auto it           = config->material_aliases.find(lower);
    if (it != config->material_aliases.end()) return it->second;

    auto pos = lower.rfind(' ');
    if (pos != std::string::npos) {
        it = config->material_aliases.find(lower.substr(pos + 1));
        if (it != config->material_aliases.end()) return it->second;
    }

    return DeduceFilamentType(material);
}

FilamentSlot BuildSlotFromConfig(const std::string& material_type, const FilamentConfig& config) {
    auto it = config.materials.find(material_type);
    if (it != config.materials.end()) return it->second;
    return {};
}

} // namespace

FilamentConfig FilamentConfig::BuiltinDefaults() {
    FilamentConfig cfg;

    cfg.colors = {
        {"bamboo green", "#00AE42"}, {"white", "#FFFFFF"},     {"black", "#000000"},
        {"red", "#C12E1F"},          {"green", "#00AE42"},     {"blue", "#0A2989"},
        {"cyan", "#0086D6"},         {"magenta", "#EC008C"},   {"yellow", "#F4EE2A"},
        {"orange", "#FF8C00"},       {"pink", "#FF69B4"},      {"purple", "#800080"},
        {"brown", "#8B4513"},        {"grey", "#808080"},      {"gray", "#808080"},
        {"gold", "#FFD700"},         {"silver", "#C0C0C0"},    {"navy", "#000080"},
        {"teal", "#008080"},         {"olive", "#808000"},     {"maroon", "#800000"},
        {"lime", "#00FF00"},         {"aqua", "#00FFFF"},      {"coral", "#FF7F50"},
        {"salmon", "#FA8072"},       {"turquoise", "#40E0D0"}, {"violet", "#EE82EE"},
        {"indigo", "#4B0082"},       {"crimson", "#DC143C"},   {"beige", "#F5F5DC"},
        {"ivory", "#FFFFF0"},        {"lavender", "#E6E6FA"},  {"chocolate", "#D2691E"},
        {"khaki", "#F0E68C"},
    };

    cfg.fallback_palette = {"#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231",
                            "#911EB4", "#42D4F4", "#F032E6", "#BFEF45", "#FABED4"};

    auto make_slot = [](const char* type, const char* sid, const char* fid, int temp) {
        FilamentSlot s;
        s.type                = type;
        s.settings_id         = sid;
        s.filament_id         = fid;
        s.nozzle_temp         = temp;
        s.nozzle_temp_initial = temp;
        return s;
    };
    cfg.materials["PLA"]  = make_slot("PLA", "Bambu PLA Basic @BBL P2S", "GFA00", 220);
    cfg.materials["PETG"] = make_slot("PETG", "Bambu PETG Basic @BBL P2S", "GFG00", 245);
    cfg.materials["ABS"]  = make_slot("ABS", "Bambu ABS @BBL P2S", "GFA01", 260);
    cfg.materials["TPU"]  = make_slot("TPU", "Bambu TPU 95A @BBL P2S", "GFU00", 230);
    cfg.materials["ASA"]  = make_slot("ASA", "Bambu ASA @BBL P2S", "GFA02", 260);
    cfg.materials["PA"]   = make_slot("PA", "Bambu PA @BBL P2S", "GFN00", 290);

    cfg.material_aliases = {
        {"pla", "PLA"},   {"pla basic", "PLA"},   {"pla matte", "PLA"},
        {"petg", "PETG"}, {"petg basic", "PETG"}, {"abs", "ABS"},
        {"tpu", "TPU"},   {"asa", "ASA"},         {"pa", "PA"},
    };

    return cfg;
}

FilamentConfig FilamentConfig::LoadFromJson(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) { throw IOError("Cannot open filament config: " + path); }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error& e) {
        throw FormatError("Failed to parse filament config: " + std::string(e.what()));
    }

    FilamentConfig cfg;

    if (j.contains("colors") && j["colors"].is_object()) {
        for (auto& [k, v] : j["colors"].items()) { cfg.colors[k] = v.get<std::string>(); }
    }

    if (j.contains("fallback_palette") && j["fallback_palette"].is_array()) {
        for (auto& v : j["fallback_palette"]) {
            cfg.fallback_palette.push_back(v.get<std::string>());
        }
    }

    if (j.contains("materials") && j["materials"].is_object()) {
        for (auto& [type_key, mat] : j["materials"].items()) {
            FilamentSlot slot;
            slot.type = type_key;
            if (mat.contains("settings_id"))
                slot.settings_id = mat["settings_id"].get<std::string>();
            if (mat.contains("vendor")) slot.vendor = mat["vendor"].get<std::string>();
            if (mat.contains("filament_id"))
                slot.filament_id = mat["filament_id"].get<std::string>();
            if (mat.contains("nozzle_temp")) slot.nozzle_temp = mat["nozzle_temp"].get<int>();
            if (mat.contains("nozzle_temp_initial"))
                slot.nozzle_temp_initial = mat["nozzle_temp_initial"].get<int>();
            else
                slot.nozzle_temp_initial = slot.nozzle_temp;
            cfg.materials[type_key] = std::move(slot);
        }
    }

    if (j.contains("material_aliases") && j["material_aliases"].is_object()) {
        for (auto& [alias, type_key] : j["material_aliases"].items()) {
            cfg.material_aliases[alias] = type_key.get<std::string>();
        }
    }

    spdlog::info("FilamentConfig loaded from {}: {} colors, {} materials, {} aliases", path,
                 cfg.colors.size(), cfg.materials.size(), cfg.material_aliases.size());
    return cfg;
}

FilamentConfig FilamentConfig::LoadFromDir(const std::string& preset_dir) {
    auto path = std::filesystem::path(preset_dir) / "filaments.json";
    if (std::filesystem::exists(path)) { return LoadFromJson(path.string()); }
    spdlog::debug("filaments.json not found in {}, using built-in defaults", preset_dir);
    return BuiltinDefaults();
}

std::string FilamentConfig::ResolveHexColor(const std::string& color_name, int fallback_idx) const {
    std::string lower = ToLowerStr(color_name);

    auto lookup = [](const std::string& key,
                     const std::unordered_map<std::string, std::string>& map) -> std::string {
        auto it = map.find(key);
        if (it != map.end()) return it->second;
        auto pos = key.rfind(' ');
        if (pos != std::string::npos) {
            it = map.find(key.substr(pos + 1));
            if (it != map.end()) return it->second;
        }
        return {};
    };

    if (!colors.empty()) {
        std::string result = lookup(lower, colors);
        if (!result.empty()) return result;
    }

    static const auto& builtin = BuiltinDefaults().colors;
    std::string result         = lookup(lower, builtin);
    if (!result.empty()) return result;

    if (!fallback_palette.empty()) {
        return fallback_palette[static_cast<size_t>(fallback_idx) % fallback_palette.size()];
    }
    return {};
}

SlicerPreset SlicerPreset::FromProfile(const std::string& preset_dir, const PrintProfile& profile,
                                       const FilamentConfig* config, bool double_sided) {
    const FaceOrientation preset_face =
        double_sided ? FaceOrientation::FaceDown : profile.face_orientation;

    SlicerPreset preset;
    preset.nozzle          = profile.nozzle_size;
    preset.face            = preset_face;
    preset.layer_height_mm = profile.layer_height_mm;
    preset.base_layers     = profile.base_layers;
    preset.color_layers    = profile.color_layers;
    preset.double_sided    = double_sided;
    preset.preset_json_path =
        FindPresetFile(preset_dir, preset_defaults::kPrinterModel, profile.layer_height_mm,
                       profile.nozzle_size, preset_face);

    for (const auto& ch : profile.palette) {
        FilamentSlot slot;
        slot.colour = ch.hex_color.empty() ? "#FFFFFF" : ch.hex_color;

        std::string mat_type = ResolveMaterialType(ch.material, config);
        if (config) {
            FilamentSlot tpl         = BuildSlotFromConfig(mat_type, *config);
            slot.type                = tpl.type;
            slot.settings_id         = tpl.settings_id;
            slot.vendor              = tpl.vendor;
            slot.filament_id         = tpl.filament_id;
            slot.nozzle_temp         = tpl.nozzle_temp;
            slot.nozzle_temp_initial = tpl.nozzle_temp_initial;
        } else {
            slot.type        = mat_type;
            slot.settings_id = DeduceSettingsId(mat_type);
        }

        preset.filaments.push_back(std::move(slot));
    }

    return preset;
}

namespace detail {

std::string BuildProjectSettings(const SlicerPreset& preset) {
    if (preset.preset_json_path.empty()) {
        throw IOError("SlicerPreset has no preset_json_path set");
    }
    nlohmann::json j = LoadPresetJson(preset.preset_json_path);

    PatchFlushMatrix(j, preset.flush_volumes_matrix);
    PatchFilamentArrays(j, preset.filaments);
    j["from"]    = "project";
    j["version"] = preset_defaults::kBambuStudioVersion;

    char id_buf[128];
    std::snprintf(id_buf, sizeof(id_buf), "ChromaPrint3D %.2fmm @P2S %smm nozzle %s",
                  static_cast<double>(preset.layer_height_mm),
                  preset.nozzle == NozzleSize::N02 ? "0.2" : "0.4",
                  FaceOrientationTag(preset.face));
    j["print_settings_id"] = id_buf;

    // "name" MUST remain "project_settings" — BambuStudio's load_from_json
    // hard-checks this value to enable different_settings_to_system handling.
    if (!j.contains("name")) { j["name"] = "project_settings"; }

    spdlog::debug("BuildProjectSettings: loaded preset {} ({} filament overrides)",
                  preset.preset_json_path, preset.filaments.size());
    return j.dump(4);
}

std::string BuildEmbeddedProcessPreset(const SlicerPreset& preset) {
    if (preset.preset_json_path.empty()) {
        throw IOError("SlicerPreset has no preset_json_path set");
    }
    nlohmann::json j = LoadPresetJson(preset.preset_json_path);

    PatchFlushMatrix(j, preset.flush_volumes_matrix);
    PatchFilamentArrays(j, preset.filaments);
    j["from"]    = "project";
    j["version"] = preset_defaults::kBambuStudioVersion;

    char id_buf[128];
    std::snprintf(id_buf, sizeof(id_buf), "ChromaPrint3D %.2fmm @P2S %smm nozzle %s",
                  static_cast<double>(preset.layer_height_mm),
                  preset.nozzle == NozzleSize::N02 ? "0.2" : "0.4",
                  FaceOrientationTag(preset.face));

    j["print_settings_id"] = id_buf;
    j["name"]              = id_buf;

    // BambuStudio's load_project_embedded_presets requires a valid "inherits"
    // pointing to an existing system preset; without it the embedded preset is skipped.
    const char* parent = (preset.nozzle == NozzleSize::N02)
                             ? "0.08mm High Quality @BBL P2S 0.2 nozzle"
                             : "0.08mm High Quality @BBL P2S";
    j["inherits"]      = parent;

    spdlog::debug("BuildEmbeddedProcessPreset: name={}, inherits={}", id_buf, parent);
    return j.dump(4);
}

namespace {

float NozzleDiameter(NozzleSize n) { return n == NozzleSize::N02 ? 0.2f : 0.4f; }

} // namespace

std::string BuildLayerConfigRanges(const SlicerPreset& preset) {
    if (preset.base_layers <= 0) return {};
    if (preset.custom_base_layers) return {};

    const float fine_lh   = preset.layer_height_mm;
    const float coarse_lh = NozzleDiameter(preset.nozzle) * 0.5f;
    if (coarse_lh <= fine_lh) return {};

    const float base_h  = static_cast<float>(preset.base_layers) * fine_lh;
    const float color_h = static_cast<float>(preset.color_layers) * fine_lh;
    const float total_h = preset.double_sided ? (base_h + color_h + color_h) : (base_h + color_h);

    const float t_offset = preset.transparent_layer_mm;

    float base_min_z, base_max_z;
    if (preset.double_sided) {
        base_min_z = color_h;
        base_max_z = color_h + base_h;
    } else if (preset.face == FaceOrientation::FaceUp) {
        base_min_z = 0.0f;
        base_max_z = base_h;
    } else {
        base_min_z = color_h + t_offset;
        base_max_z = total_h + t_offset;
    }

    const double dmin = static_cast<double>(base_min_z);
    const double dmax = static_cast<double>(base_max_z);
    const double dlh  = static_cast<double>(coarse_lh);

    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                  "<objects>\n"
                  " <object id=\"1\">\n"
                  "  <range min_z=\"%.17g\" max_z=\"%.17g\">\n"
                  "   <option opt_key=\"extruder\">0</option>\n"
                  "   <option opt_key=\"layer_height\">%.17g</option>\n"
                  "  </range>\n"
                  " </object>\n"
                  "</objects>\n",
                  dmin, dmax, dlh);

    spdlog::debug("BuildLayerConfigRanges: base=[{},{}]@{}mm, face={}, double_sided={}", dmin, dmax,
                  dlh, FaceOrientationTag(preset.face), preset.double_sided);
    return buf;
}

std::string BuildModelSettings(const ExportedGroup& group) {
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    xml << "<config>\n";

    if (group.assembly_object_id > 0) {
        xml << "  <object id=\"" << group.assembly_object_id << "\">\n";
        xml << "    <metadata key=\"name\" value=\"" << group.assembly_name << "\"/>\n";
        xml << "    <metadata key=\"extruder\" value=\"1\"/>\n";
        xml << "    <metadata face_count=\"" << group.total_face_count << "\"/>\n";
        for (const auto& obj : group.objects) {
            xml << "    <part id=\"" << obj.part_id << "\" subtype=\"normal_part\">\n";
            xml << "      <metadata key=\"name\" value=\"" << obj.name << "\"/>\n";
            xml << "      <metadata key=\"matrix\" "
                   "value=\"1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\"/>\n";
            xml << "      <metadata key=\"source_file\" value=\"" << group.assembly_name
                << ".3mf\"/>\n";
            xml << "      <metadata key=\"source_object_id\" value=\"0\"/>\n";
            xml << "      <metadata key=\"source_volume_id\" value=\"0\"/>\n";
            xml << "      <metadata key=\"source_offset_x\" value=\"0\"/>\n";
            xml << "      <metadata key=\"source_offset_y\" value=\"0\"/>\n";
            xml << "      <metadata key=\"source_offset_z\" value=\"0\"/>\n";
            xml << "      <metadata key=\"extruder\" value=\"" << obj.filament_slot << "\"/>\n";
            xml << "      <mesh_stat edges_fixed=\"0\" degenerate_facets=\"0\""
                << " facets_removed=\"0\" facets_reversed=\"0\" backwards_edges=\"0\""
                << " face_count=\"" << obj.face_count << "\"/>\n";
            xml << "    </part>\n";
        }
        xml << "  </object>\n";

        xml << "  <plate>\n";
        xml << "    <metadata key=\"plater_id\" value=\"1\"/>\n";
        xml << "    <metadata key=\"plater_name\" value=\"\"/>\n";
        xml << "    <metadata key=\"locked\" value=\"false\"/>\n";
        xml << "    <metadata key=\"filament_map_mode\" value=\"Auto For Flush\"/>\n";
        xml << "    <model_instance>\n";
        xml << "      <metadata key=\"object_id\" value=\"" << group.assembly_object_id << "\"/>\n";
        xml << "      <metadata key=\"instance_id\" value=\"0\"/>\n";
        xml << "      <metadata key=\"identify_id\" value=\"1\"/>\n";
        xml << "    </model_instance>\n";
        xml << "  </plate>\n";

        xml << "  <assemble>\n";
        xml << "    <assemble_item object_id=\"" << group.assembly_object_id
            << "\" instance_id=\"0\"" << " transform=\"1 0 0 0 1 0 0 0 1 " << group.offset_x << " "
            << group.offset_y << " " << group.offset_z << "\"" << " offset=\"" << group.offset_x
            << " " << group.offset_y << " " << group.offset_z << "\"/>\n";
        xml << "  </assemble>\n";
    } else {
        for (const auto& obj : group.objects) {
            xml << "  <object id=\"" << obj.part_id << "\">\n";
            xml << "    <metadata key=\"name\" value=\"" << obj.name << "\"/>\n";
            xml << "    <metadata key=\"extruder\" value=\"" << obj.filament_slot << "\"/>\n";
            xml << "    <metadata face_count=\"" << obj.face_count << "\"/>\n";
            xml << "  </object>\n";
        }
        xml << "  <plate>\n";
        xml << "    <metadata key=\"plater_id\" value=\"1\"/>\n";
        xml << "    <metadata key=\"plater_name\" value=\"\"/>\n";
        xml << "    <metadata key=\"locked\" value=\"false\"/>\n";
        xml << "    <metadata key=\"filament_map_mode\" value=\"Auto For Flush\"/>\n";
        int identify_id = 200;
        for (const auto& obj : group.objects) {
            xml << "    <model_instance>\n";
            xml << "      <metadata key=\"object_id\" value=\"" << obj.part_id << "\"/>\n";
            xml << "      <metadata key=\"instance_id\" value=\"0\"/>\n";
            xml << "      <metadata key=\"identify_id\" value=\"" << identify_id++ << "\"/>\n";
            xml << "    </model_instance>\n";
        }
        xml << "  </plate>\n";
        xml << "  <assemble>\n";
        for (const auto& obj : group.objects) {
            xml << "    <assemble_item object_id=\"" << obj.part_id
                << "\" instance_id=\"0\" transform=\"1 0 0 0 1 0 0 0 1 " << group.offset_x << " "
                << group.offset_y << " " << group.offset_z << "\"" << " offset=\"" << group.offset_x
                << " " << group.offset_y << " " << group.offset_z << "\"/>\n";
        }
        xml << "  </assemble>\n";
    }

    xml << "</config>\n";
    return xml.str();
}

std::string BuildSliceInfo() {
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    xml << "<config>\n";
    xml << "  <header>\n";
    xml << "    <header_item key=\"X-BBL-Client-Type\" value=\"slicer\"/>\n";
    xml << "    <header_item key=\"X-BBL-Client-Version\" value=\"02.05.00.66\"/>\n";
    xml << "  </header>\n";
    xml << "</config>\n";
    return xml.str();
}

std::string BuildCutInformation(const ExportedGroup& group) {
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
    xml << "<objects>\n";
    if (group.assembly_object_id > 0) {
        xml << " <object id=\"" << group.assembly_object_id << "\">\n";
        xml << "  <cut_id id=\"0\" check_sum=\"1\" connectors_cnt=\"0\"/>\n";
        xml << " </object>\n";
    } else {
        for (const auto& obj : group.objects) {
            xml << " <object id=\"" << obj.part_id << "\">\n";
            xml << "  <cut_id id=\"0\" check_sum=\"1\" connectors_cnt=\"0\"/>\n";
            xml << " </object>\n";
        }
    }
    xml << "</objects>\n";
    return xml.str();
}

std::string BuildFilamentSequence() { return R"({"plate_1":{"sequence":[]}})"; }

} // namespace detail
} // namespace ChromaPrint3D
