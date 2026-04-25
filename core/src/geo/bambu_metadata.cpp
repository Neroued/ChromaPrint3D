#include "bambu_metadata.h"

#include "chromaprint3d/error.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>

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

void PatchFilamentArrays(nlohmann::json& j, const std::vector<FilamentSlot>& filaments,
                         std::size_t K_per_extruder) {
    if (filaments.empty()) return;
    const std::size_t N = filaments.size();

    auto patch_no_variant = [&](const char* key, auto extractor) {
        if (!j.contains(key) || !j[key].is_array()) return;
        auto& arr = j[key];
        // Resize to N slots; default-fill with the first existing slot value.
        if (arr.size() != N) {
            nlohmann::json templ = arr.empty() ? nlohmann::json("") : arr.front();
            arr                  = nlohmann::json::array();
            for (std::size_t i = 0; i < N; ++i) arr.push_back(templ);
        }
        for (std::size_t i = 0; i < N; ++i) arr[i] = extractor(filaments[i]);
    };
    auto patch_with_variant = [&](const char* key, auto extractor) {
        if (!j.contains(key) || !j[key].is_array()) return;
        auto& arr                = j[key];
        const std::size_t target = N * K_per_extruder;
        if (arr.size() != target) {
            nlohmann::json templ = arr.empty() ? nlohmann::json("0") : arr.front();
            // Tile a single user-slot block (K_per_extruder entries) per filament.
            arr = nlohmann::json::array();
            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t k = 0; k < K_per_extruder; ++k) arr.push_back(templ);
            }
        }
        for (std::size_t i = 0; i < N; ++i) {
            const auto value = extractor(filaments[i]);
            for (std::size_t k = 0; k < K_per_extruder; ++k)
                arr[i * K_per_extruder + k] = value;
        }
    };

    patch_no_variant("filament_colour", [](const FilamentSlot& s) { return s.colour; });
    patch_no_variant("filament_multi_colour", [](const FilamentSlot& s) { return s.colour; });
    patch_no_variant("filament_type", [](const FilamentSlot& s) { return s.type; });
    patch_no_variant("filament_settings_id", [](const FilamentSlot& s) { return s.settings_id; });
    patch_no_variant("filament_ids", [](const FilamentSlot& s) { return s.filament_id; });
    patch_no_variant("filament_vendor", [](const FilamentSlot& s) { return s.vendor; });
    patch_with_variant("nozzle_temperature",
                       [](const FilamentSlot& s) { return std::to_string(s.nozzle_temp); });
    patch_with_variant("nozzle_temperature_initial_layer",
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

// Settings-id deduction is now machine-aware: the catalog provides the
// fallback PLA settings_id via MachineSpec.filament_template. For
// non-PLA materials we fall back to a P2S-style name; this is a temporary
// best-effort approximation that the user can override per-slot via the
// FilamentConfig material table.
std::string DeduceSettingsId(const std::string& filament_type, const std::string& fallback) {
    if (filament_type == "PETG") return "Bambu PETG Basic @BBL P2S";
    if (filament_type == "ABS") return "Bambu ABS @BBL P2S";
    if (filament_type == "TPU") return "Bambu TPU 95A @BBL P2S";
    if (filament_type == "ASA") return "Bambu ASA @BBL P2S";
    if (filament_type == "PA") return "Bambu PA @BBL P2S";
    if (!fallback.empty()) return fallback;
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
        {"bambu green", "#00AE42"}, {"white", "#FFFFFF"},     {"black", "#000000"},
        {"red", "#C12E1F"},         {"green", "#00AE42"},     {"blue", "#0A2989"},
        {"cyan", "#0086D6"},        {"magenta", "#EC008C"},   {"yellow", "#F4EE2A"},
        {"orange", "#FF8C00"},      {"pink", "#FF69B4"},      {"purple", "#800080"},
        {"brown", "#8B4513"},       {"grey", "#808080"},      {"gray", "#808080"},
        {"gold", "#FFD700"},        {"silver", "#C0C0C0"},    {"navy", "#000080"},
        {"teal", "#008080"},        {"olive", "#808000"},     {"maroon", "#800000"},
        {"lime", "#00FF00"},        {"aqua", "#00FFFF"},      {"coral", "#FF7F50"},
        {"salmon", "#FA8072"},      {"turquoise", "#40E0D0"}, {"violet", "#EE82EE"},
        {"indigo", "#4B0082"},      {"crimson", "#DC143C"},   {"beige", "#F5F5DC"},
        {"ivory", "#FFFFF0"},       {"lavender", "#E6E6FA"},  {"chocolate", "#D2691E"},
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

SlicerPreset SlicerPreset::FromProfile(const BambuPresetCatalog& catalog,
                                       const PrintProfile& profile, std::string_view target_machine,
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

    auto resolved = catalog.Resolve(target_machine, preset.nozzle, preset.layer_height_mm);
    if (resolved) {
        preset.machine = std::move(*resolved);
    } else {
        spdlog::warn("SlicerPreset::FromProfile: machine `{}` (n={}, lh={}) not resolved; "
                     "3MF export will fall back to standard mode.",
                     std::string(target_machine.empty() ? catalog.DefaultMachine()
                                                        : std::string(target_machine)),
                     NozzleSizeTag(preset.nozzle), static_cast<double>(preset.layer_height_mm));
    }

    const std::string fallback_settings_id =
        preset.machine_resolved() ? preset.machine.filament_template : std::string{};

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
            // For PLA Basic specifically, prefer the machine's filament_template
            // (which encodes any P1S/X1/X1E/X1C alias) over the FilamentConfig default.
            if (mat_type == "PLA" && !fallback_settings_id.empty()) {
                slot.settings_id = fallback_settings_id;
            }
        } else {
            slot.type        = mat_type;
            slot.settings_id = DeduceSettingsId(mat_type, fallback_settings_id);
        }

        preset.filaments.push_back(std::move(slot));
    }

    return preset;
}

namespace detail {

// ---------------------------------------------------------------------------
// Field-set constants (plan v13 §6.1).
// Hand-mirrored from BambuStudio src/libslic3r/PrintConfig.cpp print_options_with_variant
// (lines 6986-7028) and filament_options_with_variant (lines 7030-7074).
// scripts/build_preset_bases.py --list-{print,filament}-with-variant-keys reproduces
// these sets at runtime; they MUST match this constant.
// ---------------------------------------------------------------------------

const std::unordered_set<std::string>& PrintWithVariantKeys() {
    static const std::unordered_set<std::string> kSet = {
        "initial_layer_speed",
        "initial_layer_infill_speed",
        "outer_wall_speed",
        "inner_wall_speed",
        "small_perimeter_speed",
        "small_perimeter_threshold",
        "sparse_infill_speed",
        "internal_solid_infill_speed",
        "vertical_shell_speed",
        "top_surface_speed",
        "enable_overhang_speed",
        "overhang_1_4_speed",
        "overhang_2_4_speed",
        "overhang_3_4_speed",
        "overhang_4_4_speed",
        "overhang_totally_speed",
        "enable_height_slowdown",
        "slowdown_start_height",
        "slowdown_start_speed",
        "slowdown_start_acc",
        "slowdown_end_height",
        "slowdown_end_speed",
        "slowdown_end_acc",
        "bridge_speed",
        "gap_infill_speed",
        "support_speed",
        "support_interface_speed",
        "travel_speed",
        "travel_speed_z",
        "default_acceleration",
        "travel_acceleration",
        "travel_short_distance_acceleration",
        "initial_layer_travel_acceleration",
        "initial_layer_acceleration",
        "outer_wall_acceleration",
        "inner_wall_acceleration",
        "sparse_infill_acceleration",
        "top_surface_acceleration",
        "print_extruder_id",
        "print_extruder_variant",
        "top_solid_infill_flow_ratio",
    };
    return kSet;
}

const std::unordered_set<std::string>& FilamentWithVariantKeys() {
    static const std::unordered_set<std::string> kSet = {
        "filament_flow_ratio",
        "filament_max_volumetric_speed",
        "filament_ramming_volumetric_speed",
        "filament_pre_cooling_temperature",
        "filament_ramming_travel_time",
        "filament_ramming_volumetric_speed_nc",
        "filament_pre_cooling_temperature_nc",
        "filament_ramming_travel_time_nc",
        "filament_extruder_variant",
        "filament_retraction_length",
        "filament_retract_length_nc",
        "filament_z_hop",
        "filament_z_hop_types",
        "filament_retract_restart_extra",
        "filament_retraction_speed",
        "filament_deretraction_speed",
        "filament_retraction_minimum_travel",
        "filament_retract_when_changing_layer",
        "filament_wipe",
        "filament_wipe_distance",
        "filament_retract_before_wipe",
        "filament_long_retractions_when_cut",
        "filament_retraction_distances_when_cut",
        "long_retractions_when_ec",
        "retraction_distances_when_ec",
        "nozzle_temperature_initial_layer",
        "nozzle_temperature",
        "filament_flush_volumetric_speed",
        "filament_flush_temp",
        "filament_enable_overhang_speed",
        "filament_bridge_speed",
        "filament_overhang_1_4_speed",
        "filament_overhang_2_4_speed",
        "filament_overhang_3_4_speed",
        "filament_overhang_4_4_speed",
        "filament_overhang_totally_speed",
        "override_process_overhang_speed",
        "volumetric_speed_coefficients",
        "filament_adaptive_volumetric_speed",
        "filament_cooling_before_tower",
        "slow_down_min_speed",
    };
    return kSet;
}

// ---------------------------------------------------------------------------
// Variant-meta helpers
// ---------------------------------------------------------------------------

std::vector<std::string> BuildFilamentSelfIndex(std::size_t N, std::size_t K_per_extruder) {
    std::vector<std::string> out;
    out.reserve(N * K_per_extruder);
    for (std::size_t i = 1; i <= N; ++i) {
        const std::string id = std::to_string(i);
        for (std::size_t k = 0; k < K_per_extruder; ++k) out.push_back(id);
    }
    return out;
}

std::vector<std::string>
BuildFilamentExtruderVariant(const std::vector<std::string>& extruder0_variants,
                              std::size_t N) {
    std::vector<std::string> out;
    out.reserve(extruder0_variants.size() * N);
    for (std::size_t i = 0; i < N; ++i) {
        out.insert(out.end(), extruder0_variants.begin(), extruder0_variants.end());
    }
    return out;
}

namespace {

bool IsFilamentNoVariantKey(const std::string& key) {
    if (FilamentWithVariantKeys().count(key)) return false;
    if (PrintWithVariantKeys().count(key)) return false;
    static const std::unordered_set<std::string> kExtra = {
        "activate_air_filtration",
        "additional_cooling_fan_speed",
        "additional_fan_full_speed_layer",
        "chamber_temperatures",
        "circle_compensation_speed",
        "close_additional_fan_first_x_layers",
        "close_fan_the_first_x_layers",
        "complete_print_exhaust_fan_speed",
        "cool_plate_temp",
        "cool_plate_temp_initial_layer",
        "cooling_perimeter_transition_distance",
        "cooling_slowdown_logic",
        "counter_coef_1",
        "counter_coef_2",
        "counter_coef_3",
        "counter_limit_max",
        "counter_limit_min",
        "diameter_limit",
        "during_print_exhaust_fan_speed",
        "enable_overhang_bridge_fan",
        "enable_pressure_advance",
        "eng_plate_temp",
        "eng_plate_temp_initial_layer",
        "fan_cooling_layer_time",
        "fan_max_speed",
        "fan_min_speed",
        "first_x_layer_fan_speed",
        "full_fan_speed_layer",
        "hole_coef_1",
        "hole_coef_2",
        "hole_coef_3",
        "hole_limit_max",
        "hole_limit_min",
        "hot_plate_temp",
        "hot_plate_temp_initial_layer",
        "impact_strength_z",
        "no_slow_down_for_cooling_on_outwalls",
        "nozzle_temperature_range_high",
        "nozzle_temperature_range_low",
        "overhang_fan_speed",
        "overhang_fan_threshold",
        "overhang_threshold_participating_cooling",
        "pre_start_fan_time",
        "pressure_advance",
        "reduce_fan_stop_start_freq",
        "required_nozzle_HRC",
        "slow_down_for_layer_cooling",
        "slow_down_layer_time",
        "supertack_plate_temp",
        "supertack_plate_temp_initial_layer",
        "temperature_vitrification",
        "textured_plate_temp",
        "textured_plate_temp_initial_layer",
    };
    if (kExtra.count(key)) return true;
    return key.rfind("filament_", 0) == 0 || key.rfind("default_filament_", 0) == 0;
}

// Resolve a single $variant_indexed entry against the given process_variant.
void ApplyVariantIndexedPatch(nlohmann::json& final_dict, const std::string& key,
                              const nlohmann::json& dict_value,
                              const std::vector<std::string>& process_variant) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& vname : process_variant) {
        if (!dict_value.contains(vname)) {
            throw FormatError("chromaprint_patches.json: $variant_indexed for `" + key +
                              "` missing variant `" + vname + "`");
        }
        arr.push_back(dict_value.at(vname));
    }
    final_dict[key] = arr;
}

// Apply a JsonSection (key -> JSON-encoded value) to final_dict.
// Resolves $variant_indexed and ${layer_height} template substitutions.
void ApplyPatchSection(nlohmann::json& final_dict, const ChromaPrintPatches::JsonSection& section,
                       const std::vector<std::string>& process_variant, float layer_height_mm) {
    char lh_buf[16];
    std::snprintf(lh_buf, sizeof(lh_buf), "%.4g", static_cast<double>(layer_height_mm));
    const std::string lh_str             = lh_buf;
    constexpr const char* kLhPlaceholder = "${layer_height}";

    for (const auto& [key, raw_value] : section) {
        nlohmann::json v;
        try {
            v = nlohmann::json::parse(raw_value);
        } catch (const nlohmann::json::parse_error& e) {
            throw FormatError("chromaprint_patches.json: malformed value for `" + key +
                              "`: " + e.what());
        }
        if (v.is_object() && v.contains("$variant_indexed")) {
            ApplyVariantIndexedPatch(final_dict, key, v["$variant_indexed"], process_variant);
        } else if (v.is_string()) {
            std::string s  = v.get<std::string>();
            const auto pos = s.find(kLhPlaceholder);
            if (pos != std::string::npos) { s.replace(pos, std::strlen(kLhPlaceholder), lh_str); }
            final_dict[key] = s;
        } else {
            final_dict[key] = v;
        }
    }
}

std::string ComputeDynamicDiff(const nlohmann::json& final_dict, const nlohmann::json& base_dict) {
    std::vector<std::string> changed;
    for (auto it = final_dict.begin(); it != final_dict.end(); ++it) {
        const std::string& k = it.key();
        if (!k.empty() && k.front() == '_') continue;
        auto bit = base_dict.find(k);
        if (bit == base_dict.end()) {
            changed.push_back(k);
        } else if (*bit != it.value()) {
            changed.push_back(k);
        }
    }
    std::sort(changed.begin(), changed.end());
    std::ostringstream oss;
    for (size_t i = 0; i < changed.size(); ++i) {
        if (i > 0) oss << ';';
        oss << changed[i];
    }
    return oss.str();
}

void ExpandFilamentNoVariantToN(nlohmann::json& final_dict, std::size_t N) {
    for (auto it = final_dict.begin(); it != final_dict.end(); ++it) {
        if (!it.value().is_array()) continue;
        if (it.value().size() == N) continue;
        if (it.value().size() != 1) continue;
        if (!IsFilamentNoVariantKey(it.key())) continue;
        nlohmann::json templ = it.value().front();
        nlohmann::json arr   = nlohmann::json::array();
        for (std::size_t i = 0; i < N; ++i) arr.push_back(templ);
        it.value() = arr;
    }
}

/// Expand each `filament_options_with_variant` array from K_per_extruder to
/// N*K_per_extruder by repeating the per-extruder slice for every user filament slot.
///
/// **Pre-condition**: base_dict has been pre-aligned to K_per_extruder by
/// `scripts/build_preset_bases.py` step 3.5; any other length here indicates
/// data corruption or a base/code drift, so we throw rather than silently skip
/// (plan v13.1 / m5 + v13.2 / m-realfile; aligns with §10 risk table CCC).
void ExpandFilamentWithVariantToN(nlohmann::json& final_dict, std::size_t N,
                                  std::size_t K_per_extruder) {
    for (auto it = final_dict.begin(); it != final_dict.end(); ++it) {
        if (!FilamentWithVariantKeys().count(it.key())) continue;
        if (!it.value().is_array()) continue;
        const std::size_t target = N * K_per_extruder;
        if (it.value().size() == target) continue;
        if (it.value().size() != K_per_extruder) {
            throw FormatError(
                "ExpandFilamentWithVariantToN: field `" + it.key() + "` length " +
                std::to_string(it.value().size()) + " != K_per_extruder " +
                std::to_string(K_per_extruder) +
                " (expected base_dict to be pre-aligned by "
                "scripts/build_preset_bases.py step 3.5; check base file integrity)");
        }
        nlohmann::json arr = nlohmann::json::array();
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t k = 0; k < K_per_extruder; ++k) arr.push_back(it.value().at(k));
        }
        it.value() = arr;
    }
}

/// Parse extruder_variant_list[0] (CSV) -> extruder 0's variants.
/// Returns empty vector if the field is missing/malformed (caller throws).
std::vector<std::string>
ParseExtruder0Variants(const nlohmann::json& extruder_variant_list) {
    std::vector<std::string> out;
    if (!extruder_variant_list.is_array() || extruder_variant_list.empty()) return out;
    if (!extruder_variant_list.front().is_string()) return out;
    const std::string raw = extruder_variant_list.front().get<std::string>();
    std::string token;
    for (char c : raw) {
        if (c == ',') {
            // Trim whitespace.
            std::size_t l = 0, r = token.size();
            while (l < r && std::isspace(static_cast<unsigned char>(token[l]))) ++l;
            while (r > l && std::isspace(static_cast<unsigned char>(token[r - 1]))) --r;
            if (l < r) out.emplace_back(token.substr(l, r - l));
            token.clear();
        } else {
            token.push_back(c);
        }
    }
    if (!token.empty()) {
        std::size_t l = 0, r = token.size();
        while (l < r && std::isspace(static_cast<unsigned char>(token[l]))) ++l;
        while (r > l && std::isspace(static_cast<unsigned char>(token[r - 1]))) --r;
        if (l < r) out.emplace_back(token.substr(l, r - l));
    }
    return out;
}

void InjectMandatoryVariantMetaFields(nlohmann::json& final_dict, const nlohmann::json& base_dict,
                                      const std::vector<std::string>& extruder0_variants,
                                      std::size_t N) {
    if (base_dict.contains("extruder_variant_list")) {
        final_dict["extruder_variant_list"] = base_dict["extruder_variant_list"];
    } else {
        throw FormatError(
            "preset_base missing extruder_variant_list (BambuStudio load will throw)");
    }

    final_dict["filament_extruder_variant"] = BuildFilamentExtruderVariant(extruder0_variants, N);
    final_dict["filament_self_index"] = BuildFilamentSelfIndex(N, extruder0_variants.size());
}

void InjectInheritsGroup(nlohmann::json& final_dict, std::size_t N,
                          const std::string& process_template) {
    // BambuStudio expects N+2 entries: print, printer, then N filaments.
    // [0] = process inherits chain head (= machine.process_template);
    // [1..N+1] = empty (printer + N filaments are project-embedded).
    // Plan v13.2 / m-realfile: real BambuStudio 3MFs fill [0] with the
    // process_template name (e.g. "0.08mm High Quality @BBL P2S").
    nlohmann::json arr = nlohmann::json::array();
    arr.push_back(process_template);
    for (std::size_t i = 0; i < N + 1; ++i) arr.push_back("");
    final_dict["inherits_group"] = arr;
}

/// Compose the print_settings_id used as preset name in BambuStudio's
/// project_settings.config.
///
/// **Pre-condition**: `preset.machine_resolved() == true` (enforced at
/// `BuildProjectSettings` entry; see plan v13.1 / m4).
std::string MakePrintSettingsId(const SlicerPreset& preset) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "ChromaPrint3D %.2fmm @%s %smm nozzle %s",
                  static_cast<double>(preset.layer_height_mm),
                  preset.machine.machine_name.c_str(),
                  preset.nozzle == NozzleSize::N02 ? "0.2" : "0.4",
                  FaceOrientationTag(preset.face));
    return std::string(buf);
}

std::string FaceLabelForPatch(FaceOrientation f) {
    return f == FaceOrientation::FaceUp ? "FaceUp" : "FaceDown";
}

} // namespace

std::string BuildProjectSettings(const SlicerPreset& preset) {
    if (!preset.machine_resolved()) {
        throw IOError("SlicerPreset has no resolved MachineSpec; call "
                      "SlicerPreset::FromProfile with a valid catalog first");
    }
    nlohmann::json base_dict = LoadPresetJson(preset.machine.preset_base_path.string());
    if (base_dict.contains("_chromaprint3d_meta")) base_dict.erase("_chromaprint3d_meta");

    nlohmann::json final_dict = base_dict;

    std::vector<std::string> process_variant;
    if (base_dict.contains("print_extruder_variant") &&
        base_dict["print_extruder_variant"].is_array()) {
        for (const auto& v : base_dict["print_extruder_variant"]) {
            process_variant.push_back(v.get<std::string>());
        }
    }
    if (process_variant.empty()) {
        throw FormatError("preset_base missing print_extruder_variant");
    }
    const std::size_t K_process = process_variant.size();
    const std::size_t N         = preset.filaments.empty() ? 1 : preset.filaments.size();

    // Plan v13.2 / m-realfile: filament arrays use K_per_extruder = len(extruder_variant_list[0].csv),
    // NOT K_process. K_process is preserved separately for print_options_with_variant.
    if (!base_dict.contains("extruder_variant_list")) {
        throw FormatError("preset_base missing extruder_variant_list");
    }
    const std::vector<std::string> extruder0_variants =
        ParseExtruder0Variants(base_dict["extruder_variant_list"]);
    if (extruder0_variants.empty()) {
        throw FormatError(
            "preset_base extruder_variant_list[0] yielded no variants (malformed CSV)");
    }
    const std::size_t K_per_extruder = extruder0_variants.size();

    // Apply ChromaPrint3D patches (process_common -> per_nozzle -> per_face).
    if (preset.machine.patches) {
        const auto& patches = *preset.machine.patches;
        ApplyPatchSection(final_dict, patches.process_common, process_variant,
                          preset.layer_height_mm);

        const std::string nozzle_key = (preset.nozzle == NozzleSize::N02) ? "0.2" : "0.4";
        auto nit                     = patches.process_per_nozzle.find(nozzle_key);
        if (nit != patches.process_per_nozzle.end()) {
            ApplyPatchSection(final_dict, nit->second, process_variant, preset.layer_height_mm);
        }

        const std::string face_key = FaceLabelForPatch(preset.face);
        auto fit                   = patches.process_per_face.find(face_key);
        if (fit != patches.process_per_face.end()) {
            ApplyPatchSection(final_dict, fit->second, process_variant, preset.layer_height_mm);
        }
    }

    // Compute dynamic diff (single concatenated `;`-joined string).
    const std::string diff_str = ComputeDynamicDiff(final_dict, base_dict);

    // N-slot expansion (plan v13 §6.4 step 4 + v13.2 K_per_extruder semantics).
    ExpandFilamentNoVariantToN(final_dict, N);
    ExpandFilamentWithVariantToN(final_dict, N, K_per_extruder);

    // 3 mandatory variant meta-fields (plan v13 §6.4 step 5; BBB+WW + v13.2 m-realfile).
    InjectMandatoryVariantMetaFields(final_dict, base_dict, extruder0_variants, N);

    // Apply user-level overrides (palette colors, settings_id, etc.).
    PatchFlushMatrix(final_dict, preset.flush_volumes_matrix);
    PatchFilamentArrays(final_dict, preset.filaments, K_per_extruder);

    // Inject metadata (plan v13 §6.4 step 8 + v13.2 / m-realfile cleanup).
    final_dict["from"]                = "project";
    final_dict["name"]                = "project_settings";
    final_dict["version"]             = preset_defaults::kBambuStudioVersion;
    final_dict["print_settings_id"]   = MakePrintSettingsId(preset);
    final_dict["printer_settings_id"] = preset.machine.printer_template;
    final_dict["printer_model"]       = preset.machine.printer_model;
    if (!final_dict.contains("printer_variant") || !final_dict["printer_variant"].is_string()) {
        final_dict["printer_variant"] = preset.nozzle == NozzleSize::N02 ? "0.2" : "0.4";
    }
    // is_custom_defined / compatible_*_expression_group are NOT written here:
    // BambuStudio's `add_if_some_non_empty` (PresetBundle.cpp:264) only writes
    // expression_group fields when non-empty; user real-file 3MFs omit them.
    // is_custom_defined is also absent in user real-file 3MFs.

    // Project-level metadata. inherits_group[0] = process_template (plan v13.2).
    InjectInheritsGroup(final_dict, N, preset.machine.process_template);
    {
        nlohmann::json arr = nlohmann::json::array();
        arr.push_back(diff_str); // print
        for (std::size_t i = 0; i < N + 1; ++i) arr.push_back("");
        final_dict["different_settings_to_system"] = arr;
    }

    // Cross-machine retention list (plan v13 §6.4 step 8 / proposition 1).
    {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& pname : preset.machine.compatible_printers) arr.push_back(pname);
        final_dict["print_compatible_printers"] = arr;
    }

    spdlog::debug("BuildProjectSettings: machine={} N={} K_process={} K_per_extruder={} fields={}",
                  preset.machine.machine_name, N, K_process, K_per_extruder, final_dict.size());
    return final_dict.dump(4);
}

std::string BuildEmbeddedProcessPreset(const SlicerPreset& preset) {
    if (!preset.machine_resolved()) { throw IOError("SlicerPreset has no resolved MachineSpec"); }
    nlohmann::json base_dict = LoadPresetJson(preset.machine.preset_base_path.string());
    if (base_dict.contains("_chromaprint3d_meta")) base_dict.erase("_chromaprint3d_meta");

    nlohmann::json final_dict = base_dict;

    std::vector<std::string> process_variant;
    if (base_dict.contains("print_extruder_variant") &&
        base_dict["print_extruder_variant"].is_array()) {
        for (const auto& v : base_dict["print_extruder_variant"]) {
            process_variant.push_back(v.get<std::string>());
        }
    }
    if (process_variant.empty()) {
        throw FormatError("preset_base missing print_extruder_variant");
    }

    if (preset.machine.patches) {
        const auto& patches = *preset.machine.patches;
        ApplyPatchSection(final_dict, patches.process_common, process_variant,
                          preset.layer_height_mm);
        const std::string nozzle_key = (preset.nozzle == NozzleSize::N02) ? "0.2" : "0.4";
        auto nit                     = patches.process_per_nozzle.find(nozzle_key);
        if (nit != patches.process_per_nozzle.end()) {
            ApplyPatchSection(final_dict, nit->second, process_variant, preset.layer_height_mm);
        }
        const std::string face_key = FaceLabelForPatch(preset.face);
        auto fit                   = patches.process_per_face.find(face_key);
        if (fit != patches.process_per_face.end()) {
            ApplyPatchSection(final_dict, fit->second, process_variant, preset.layer_height_mm);
        }
    }

    const std::string diff_str = ComputeDynamicDiff(final_dict, base_dict);

    PatchFlushMatrix(final_dict, preset.flush_volumes_matrix);

    const std::string preset_id                = MakePrintSettingsId(preset);
    final_dict["from"]                         = "project";
    final_dict["is_custom_defined"]            = "0";
    final_dict["version"]                      = preset_defaults::kBambuStudioVersion;
    final_dict["name"]                         = preset_id;
    final_dict["print_settings_id"]            = preset_id;
    final_dict["inherits"]                     = preset.machine.process_template;
    final_dict["different_settings_to_system"] = nlohmann::json::array({diff_str});

    {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& pname : preset.machine.compatible_printers) arr.push_back(pname);
        final_dict["compatible_printers"] = arr;
    }

    spdlog::debug("BuildEmbeddedProcessPreset: name={} inherits={} compatible_printers={}",
                  preset_id, preset.machine.process_template,
                  preset.machine.compatible_printers.size());
    return final_dict.dump(4);
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

std::string BuildModelSettings(const ExportedGroup& group, const SlicerPreset& preset) {
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
        if (preset.machine_resolved() && !preset.machine.printer_model.empty()) {
            xml << "    <metadata key=\"printer_model_id\" value=\"" << preset.machine.printer_model
                << "\"/>\n";
        }
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
        if (preset.machine_resolved() && !preset.machine.printer_model.empty()) {
            xml << "    <metadata key=\"printer_model_id\" value=\"" << preset.machine.printer_model
                << "\"/>\n";
        }
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
