#pragma once

/// \file bambu_metadata.h
/// \brief Internal: generate Bambu Studio metadata files for 3MF embedding.

#include "chromaprint3d/slicer_preset.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ChromaPrint3D::detail {

/// Variant-meta helpers (plan v13 §6.1).
/// Returns N x K_process slot ids ["1","1",...,"1","2","2",...,"N","N",...,"N"].
std::vector<std::string> BuildFilamentSelfIndex(std::size_t N, std::size_t K_process);

/// Returns N x K_process variant labels by repeating \p process_variant N times.
/// BBB key fix (plan v13 §6.1): base must be `print_extruder_variant`, NOT the
/// 1-element `filament_extruder_variant` produced by filament inheritance.
std::vector<std::string>
BuildFilamentExtruderVariant(const std::vector<std::string>& process_variant, std::size_t N);

/// Describes a single exported mesh part and its filament slot assignment.
struct ExportedObject {
    std::string name;
    int filament_slot; ///< 1-based filament slot index for Bambu Studio extruder assignment.
    int part_id;       ///< 1-based part ID within the assembly (also used as object ID in
                       ///< external model).
    int face_count;    ///< Triangle count of this part's mesh.
};

/// Describes the complete exported model group (assembly-oriented).
struct ExportedGroup {
    std::vector<ExportedObject> objects;
    int assembly_object_id = 0; ///< Assembly wrapper object ID (0 = no assembly / flat mode).
    int total_face_count   = 0;
    std::string assembly_name;
    float offset_x = 0.0f; ///< Assembly placement offset (bed centering).
    float offset_y = 0.0f;
    float offset_z = 0.0f;
};

/// Load a preset JSON file preserving its filament configuration (colours, types, etc.).
/// Only flush_volumes_matrix is patched if provided.
std::string BuildProjectSettings(const SlicerPreset& preset);

/// Generate an embedded process preset JSON (Metadata/process_settings_1.config).
/// BambuStudio registers this as a project-embedded preset so that our custom
/// print_settings_id is displayed verbatim instead of "system preset (modified)".
std::string BuildEmbeddedProcessPreset(const SlicerPreset& preset);

/// Generate Metadata/layer_config_ranges.xml for per-height-range layer height overrides.
/// Returns an empty string when no height modifier is needed (no base layers, the
/// coarse base layer height equals the fine color layer height, or the user explicitly
/// set base_layers via custom_base_layers).
std::string BuildLayerConfigRanges(const SlicerPreset& preset);

/// Generate model_settings.config XML for independent mesh objects.
/// Embeds `<metadata key="printer_model_id" value="...">` (plate metadata,
/// plan v13 §6.6 / proposition 8). When `preset.machine_resolved()` is false,
/// the printer_model_id metadata is omitted.
std::string BuildModelSettings(const ExportedGroup& group, const SlicerPreset& preset);

/// Generate slice_info.config XML.
std::string BuildSliceInfo();

/// Generate cut_information.xml (minimal stub).
std::string BuildCutInformation(const ExportedGroup& group);

/// Generate filament_sequence.json.
std::string BuildFilamentSequence();

} // namespace ChromaPrint3D::detail
