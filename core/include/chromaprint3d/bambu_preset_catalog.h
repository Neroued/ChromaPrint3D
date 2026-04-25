#pragma once

/// \file bambu_preset_catalog.h
/// \brief Multi-machine preset catalog: maps user-selected machine to a
/// pre-generated base_dict and the ChromaPrint3D patch overlay.

#include "print_profile.h"

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ChromaPrint3D {

// Forward declaration
struct ChromaPrintPatches;

/// One entry in `data/presets/machines.json`, populated by
/// `BambuPresetCatalog::Resolve` with the concrete file paths and patch
/// overlay for a specific (machine, nozzle, layer height).
struct MachineSpec {
    std::string machine_name;         ///< e.g. "Bambu Lab P2S".
    std::string slug;                 ///< Filename stem from `machines.json`, e.g. "bambu_p2s".
    std::string printer_model;        ///< Read from base file cache, e.g. "Bambu Lab P2S".
    std::string extruder_topology;    ///< "single" | "dual".
    std::vector<std::string> nozzles; ///< e.g. ["0.2", "0.4"].
    std::string process_template;     ///< For embedded process preset `inherits`.
    std::string printer_template;     ///< e.g. "Bambu Lab P2S 0.4 nozzle".
    std::string filament_template;    ///< e.g. "Bambu PLA Basic @BBL P2S".

    /// `compatible_printers` list to embed in the generated process preset.
    /// Element format MUST be `<machine_name> <nozzle> nozzle` (per BambuStudio
    /// system process preset layout). Includes every same-topology + same-nozzle
    /// machine in the catalog so user retains parameters when switching machines.
    std::vector<std::string> compatible_printers;

    /// Path to `data/preset_bases/<slug>_<lh>_<nozzle>.json`.
    std::filesystem::path preset_base_path;

    /// Shared (cheap) snapshot of the catalog's patch overlay; populated by
    /// `BambuPresetCatalog::Resolve`. Held by `shared_ptr` so we avoid copying
    /// the (small but non-trivial) ChromaPrintPatches per MachineSpec while
    /// preserving value-semantics with no lifetime ties to the catalog.
    std::shared_ptr<const ChromaPrintPatches> patches;
};

/// Patch overlay loaded from `data/presets/chromaprint_patches.json`.
struct ChromaPrintPatches {
    using Section     = std::unordered_map<std::string, std::string>;
    using JsonSection = std::unordered_map<std::string, std::string>; // serialised JSON values

    /// Each entry: key -> JSON-encoded value (scalar literal or `{"$variant_indexed": {...}}`).
    /// We keep raw JSON strings to avoid pulling nlohmann::json into the public header.
    JsonSection process_common;
    std::map<std::string, JsonSection> process_per_nozzle; // nozzle "0.2" / "0.4"
    std::map<std::string, JsonSection> process_per_face;   // "FaceUp" / "FaceDown"
    JsonSection filament_common;

    static ChromaPrintPatches LoadFromFile(const std::filesystem::path& path);
};

/// Catalog of all registered machines, the ChromaPrint3D patch overlay, and
/// the FilamentConfig material table. Loaded once at process startup from
/// `<data_dir>/presets/`.
class BambuPresetCatalog {
public:
    /// Load from `<data_dir>/presets/machines.json`,
    /// `<data_dir>/presets/chromaprint_patches.json`, and the
    /// `<data_dir>/preset_bases/` directory of base files.
    /// Throws on missing/invalid files.
    static BambuPresetCatalog LoadFromDir(const std::filesystem::path& data_dir);

    /// Returns the MachineSpec for the given machine, populated with the
    /// concrete `preset_base_path` for the requested (nozzle, layer height).
    /// Returns std::nullopt if the machine is not registered or the
    /// (machine, nozzle, lh) base file is not present.
    std::optional<MachineSpec> Resolve(std::string_view machine_name, NozzleSize nozzle,
                                       float layer_height_mm) const;

    /// Names of all registered machines (insertion order).
    std::vector<std::string> ListMachines() const;

    /// `default_machine` from machines.json.
    const std::string& DefaultMachine() const { return default_machine_; }

    /// Read-only access to the patch overlay (always non-null after LoadFromDir).
    const ChromaPrintPatches& Patches() const { return *patches_; }

    /// Whether this catalog has been initialised (LoadFromDir was called).
    bool empty() const { return machines_.empty(); }

private:
    struct MachineRecord {
        std::string slug;                                     // from machines.json
        std::string extruder_topology;
        std::vector<std::string> nozzles;
        std::map<std::string, std::string> process_template;  // nozzle -> template
        std::map<std::string, std::string> filament_template; // nozzle -> template
        std::string printer_template;                         // with {nozzle} placeholder
    };

    std::filesystem::path data_dir_;
    std::vector<std::string> machine_names_; // registration order
    std::map<std::string, MachineRecord> machines_;
    std::shared_ptr<const ChromaPrintPatches> patches_;
    std::string default_machine_;

    /// Cache of `printer_model` keyed by base filename stem `<slug>_<lh>mm_<nozzle>`
    /// (e.g. `bambu_p2s_0.08mm_n04`). Populated once at LoadFromDir; consumed
    /// by Resolve to avoid re-parsing base JSON on every export call.
    /// Reads from `_chromaprint3d_meta.printer_model` first, falling back to
    /// the top-level `printer_model` field.
    std::unordered_map<std::string, std::string> base_printer_model_cache_;
};

} // namespace ChromaPrint3D
