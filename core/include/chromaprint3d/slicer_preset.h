#pragma once

/// \file slicer_preset.h
/// \brief Slicer preset configuration for embedding print parameters in 3MF exports.

#include "bambu_preset_catalog.h"
#include "print_profile.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace ChromaPrint3D {

namespace preset_defaults {
constexpr const char* kBambuStudioVersion = "02.03.00.70";
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

/// Slicer preset loaded from a multi-machine catalog, with runtime filament overrides.
struct SlicerPreset {
    /// Resolved machine specification (machine name, topology, preset_base_path,
    /// compatible_printers list, etc.). Populated by `FromProfile`.
    MachineSpec machine;

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

    /// True iff `machine` was successfully resolved and `preset_base_path` exists.
    bool machine_resolved() const { return !machine.machine_name.empty(); }

    /// Build a SlicerPreset by resolving the requested machine in \p catalog and
    /// populating filament slots from \p profile.
    ///
    /// \p target_machine names the user-selected machine (e.g. "Bambu Lab P2S").
    ///   When empty, the catalog's `default_machine` is used.
    /// \p config supplies per-material defaults (FilamentConfig is loaded by the caller).
    /// \p double_sided forces FaceDown layout when true.
    ///
    /// Throws ChromaPrint3D::InputError when the machine is not registered or
    /// when no matching base file exists for (nozzle, layer height).
    static SlicerPreset FromProfile(const BambuPresetCatalog& catalog, const PrintProfile& profile,
                                    std::string_view target_machine = {},
                                    const FilamentConfig* config    = nullptr,
                                    bool double_sided               = false);
};

/// Match a hex color string to the closest filament slot in the preset's filament_colour array.
/// Returns 1-based slot index. \p filament_colours entries are hex strings like "#FFFFFF".
int MatchColorToSlot(const std::string& hex_color,
                     const std::vector<std::string>& filament_colours);

} // namespace ChromaPrint3D
