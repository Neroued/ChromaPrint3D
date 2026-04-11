#pragma once

/// \file slicer_preset.h
/// \brief Slicer preset configuration for embedding print parameters in 3MF exports.

#include "print_profile.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace ChromaPrint3D {

namespace preset_defaults {
constexpr const char* kBambuStudioVersion = "02.03.00.70";
constexpr const char* kPrinterModel       = "Bambu Lab P2S";
} // namespace preset_defaults

/// Per-slot filament description used when patching slicer presets.
struct FilamentSlot {
    std::string type        = "PLA";
    std::string colour      = "#FFFFFF";
    std::string settings_id = "Bambu PLA Basic @BBL P2S";
    std::string vendor      = "Bambu Lab";
    std::string filament_id = "GFA00";
    int nozzle_temp         = 220;
    int nozzle_temp_initial = 220;
};

/// External filament configuration loaded from JSON.
/// Replaces hardcoded color maps, material tables, and filament defaults.
struct FilamentConfig {
    /// Color name (lowercase) -> hex color string (e.g. "#00AE42").
    std::unordered_map<std::string, std::string> colors;

    /// Fallback hex colors for unknown color names (cycled by index).
    std::vector<std::string> fallback_palette;

    /// Material type key (e.g. "PLA") -> default FilamentSlot template.
    std::unordered_map<std::string, FilamentSlot> materials;

    /// Lowercase material alias -> canonical material type key.
    std::unordered_map<std::string, std::string> material_aliases;

    /// Load from a specific JSON file path.
    static FilamentConfig LoadFromJson(const std::string& path);

    /// Load from preset directory (looks for filaments.json); returns built-in defaults if missing.
    static FilamentConfig LoadFromDir(const std::string& preset_dir);

    /// Returns a FilamentConfig populated with the built-in hardcoded defaults.
    static FilamentConfig BuiltinDefaults();

    /// Resolve a color name (e.g. "Red", "Bambu Green") to a hex string (e.g. "#C12E1F").
    /// Tries this config's color table first, then built-in defaults, then fallback palette.
    std::string ResolveHexColor(const std::string& color_name, int fallback_idx = 0) const;
};

/// Slicer preset loaded from an external JSON template, with runtime filament overrides.
struct SlicerPreset {
    std::string preset_json_path;
    std::vector<FilamentSlot> filaments;
    std::vector<int> flush_volumes_matrix;

    NozzleSize nozzle          = NozzleSize::N04;
    FaceOrientation face       = FaceOrientation::FaceUp;
    float layer_height_mm      = 0.08f;
    int base_layers            = 0;
    int color_layers           = 5;
    bool double_sided          = false;
    bool custom_base_layers    = false; ///< User explicitly set base_layers; skip height modifier.
    float transparent_layer_mm = 0.0f;  ///< Transparent coating for FaceDown (0 = disabled).

    /// Build a SlicerPreset by selecting the best preset file from \p preset_dir
    /// and populating filament slots from the given \p profile.
    /// If \p config is non-null, material lookup uses its tables instead of hardcoded logic.
    static SlicerPreset FromProfile(const std::string& preset_dir, const PrintProfile& profile,
                                    const FilamentConfig* config = nullptr,
                                    bool double_sided            = false);
};

/// Select the best-matching preset JSON filename within \p preset_dir
/// for the given printer model, layer height, nozzle size, and face orientation.
/// Returns the full path to the preset file, or empty string if none found.
std::string FindPresetFile(const std::string& preset_dir, const std::string& printer_model,
                           float layer_height, NozzleSize nozzle = NozzleSize::N04,
                           FaceOrientation face = FaceOrientation::FaceUp);

/// Match a hex color string to the closest filament slot in the preset's filament_colour array.
/// Returns 1-based slot index. \p filament_colours entries are hex strings like "#FFFFFF".
int MatchColorToSlot(const std::string& hex_color,
                     const std::vector<std::string>& filament_colours);

/// Read the filament_colour array from a preset JSON file.
std::vector<std::string> ReadFilamentColours(const std::string& preset_json_path);

} // namespace ChromaPrint3D
