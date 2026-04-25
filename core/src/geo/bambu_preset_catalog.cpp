#include "chromaprint3d/bambu_preset_catalog.h"

#include "chromaprint3d/error.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ChromaPrint3D {

namespace {

const char* NozzleSizeStr(NozzleSize n) { return n == NozzleSize::N02 ? "0.2" : "0.4"; }

std::string MachineSlug(const std::string& machine_name) {
    // Hand-curated mapping mirroring scripts/build_preset_bases.py:slugify_machine.
    static const std::map<std::string, std::string> kSlug = {
        {"Bambu Lab P2S", "bambu_p2s"},      {"Bambu Lab P1P", "bambu_p1p"},
        {"Bambu Lab P1S", "bambu_p1s"},      {"Bambu Lab X1 Carbon", "bambu_x1c"},
        {"Bambu Lab X1", "bambu_x1"},        {"Bambu Lab X1E", "bambu_x1e"},
        {"Bambu Lab A1", "bambu_a1"},        {"Bambu Lab A1 mini", "bambu_a1m"},
        {"Bambu Lab H2S", "bambu_h2s"},      {"Bambu Lab H2D", "bambu_h2d"},
        {"Bambu Lab H2D Pro", "bambu_h2dp"}, {"Bambu Lab H2C", "bambu_h2c"},
        {"Bambu Lab X2D", "bambu_x2d"},
    };
    auto it = kSlug.find(machine_name);
    if (it != kSlug.end()) return it->second;

    // Fallback: lowercase + non-alnum -> underscore.
    std::string out;
    out.reserve(machine_name.size());
    for (char c : machine_name) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        } else if (c == ' ' || c == '_') {
            out.push_back('_');
        }
    }
    return out;
}

std::string FormatPrinterTemplate(const std::string& tpl, const std::string& nozzle) {
    constexpr const char* kPlaceholder = "{nozzle}";
    auto pos                           = tpl.find(kPlaceholder);
    if (pos == std::string::npos) return tpl;
    std::string out = tpl;
    out.replace(pos, std::strlen(kPlaceholder), nozzle);
    return out;
}

/// Encode a JSON value back to a compact string. We use this to store patch
/// values without leaking nlohmann::json into the public header.
std::string EncodeJsonValue(const nlohmann::json& v) { return v.dump(); }

ChromaPrintPatches::JsonSection LoadSection(const nlohmann::json& obj) {
    ChromaPrintPatches::JsonSection out;
    if (!obj.is_object()) return out;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        if (it.key().rfind('_', 0) == 0) continue; // Skip _doc / _meta.
        out.emplace(it.key(), EncodeJsonValue(it.value()));
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// ChromaPrintPatches
// ---------------------------------------------------------------------------

ChromaPrintPatches ChromaPrintPatches::LoadFromFile(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) { throw IOError("Cannot open chromaprint_patches.json: " + path.string()); }
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error& e) {
        throw FormatError("Failed to parse chromaprint_patches.json: " + std::string(e.what()));
    }

    ChromaPrintPatches out;
    if (j.contains("process_common")) out.process_common = LoadSection(j["process_common"]);
    if (j.contains("process_per_nozzle") && j["process_per_nozzle"].is_object()) {
        for (auto it = j["process_per_nozzle"].begin(); it != j["process_per_nozzle"].end(); ++it) {
            out.process_per_nozzle.emplace(it.key(), LoadSection(it.value()));
        }
    }
    if (j.contains("process_per_face") && j["process_per_face"].is_object()) {
        for (auto it = j["process_per_face"].begin(); it != j["process_per_face"].end(); ++it) {
            out.process_per_face.emplace(it.key(), LoadSection(it.value()));
        }
    }
    if (j.contains("filament_common")) out.filament_common = LoadSection(j["filament_common"]);

    spdlog::debug("ChromaPrintPatches loaded: {} common, {} per-nozzle sections, {} per-face "
                  "sections, {} filament-common",
                  out.process_common.size(), out.process_per_nozzle.size(),
                  out.process_per_face.size(), out.filament_common.size());
    return out;
}

// ---------------------------------------------------------------------------
// BambuPresetCatalog
// ---------------------------------------------------------------------------

BambuPresetCatalog BambuPresetCatalog::LoadFromDir(const std::filesystem::path& data_dir) {
    BambuPresetCatalog cat;
    cat.data_dir_ = data_dir;

    const auto presets_dir   = data_dir / "presets";
    const auto bases_dir     = data_dir / "preset_bases";
    const auto machines_path = presets_dir / "machines.json";
    const auto patches_path  = presets_dir / "chromaprint_patches.json";

    if (!std::filesystem::exists(machines_path)) {
        throw IOError("machines.json not found at " + machines_path.string());
    }

    std::ifstream ifs(machines_path);
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error& e) {
        throw FormatError("Failed to parse machines.json: " + std::string(e.what()));
    }

    cat.default_machine_ = j.value("default_machine", std::string{});
    if (!j.contains("machines") || !j["machines"].is_object()) {
        throw FormatError("machines.json missing required `machines` object");
    }

    for (auto it = j["machines"].begin(); it != j["machines"].end(); ++it) {
        const std::string& name = it.key();
        const auto& spec_json   = it.value();

        MachineRecord rec;
        rec.extruder_topology = spec_json.value("extruder_topology", std::string{"single"});
        rec.printer_template  = spec_json.value("printer_template", std::string{});

        if (spec_json.contains("nozzles") && spec_json["nozzles"].is_array()) {
            for (const auto& n : spec_json["nozzles"]) {
                rec.nozzles.push_back(n.get<std::string>());
            }
        }
        if (spec_json.contains("process_template") && spec_json["process_template"].is_object()) {
            for (auto pit = spec_json["process_template"].begin();
                 pit != spec_json["process_template"].end(); ++pit) {
                rec.process_template.emplace(pit.key(), pit.value().get<std::string>());
            }
        }
        if (spec_json.contains("filament_template") && spec_json["filament_template"].is_object()) {
            for (auto fit = spec_json["filament_template"].begin();
                 fit != spec_json["filament_template"].end(); ++fit) {
                rec.filament_template.emplace(fit.key(), fit.value().get<std::string>());
            }
        }

        cat.machines_.emplace(name, std::move(rec));
        cat.machine_names_.push_back(name);
    }

    if (cat.machines_.empty()) { throw FormatError("machines.json contains no machine entries"); }
    if (cat.default_machine_.empty()) {
        cat.default_machine_ = cat.machine_names_.front();
    } else if (cat.machines_.find(cat.default_machine_) == cat.machines_.end()) {
        throw FormatError("default_machine `" + cat.default_machine_ + "` not in machines list");
    }

    if (std::filesystem::exists(patches_path)) {
        cat.patches_ =
            std::make_shared<ChromaPrintPatches>(ChromaPrintPatches::LoadFromFile(patches_path));
    } else {
        spdlog::warn("BambuPresetCatalog: chromaprint_patches.json not found at {}",
                     patches_path.string());
        cat.patches_ = std::make_shared<ChromaPrintPatches>();
    }

    if (!std::filesystem::exists(bases_dir)) {
        throw IOError("preset_bases directory not found at " + bases_dir.string());
    }

    spdlog::info("BambuPresetCatalog loaded: {} machines, default = {}", cat.machines_.size(),
                 cat.default_machine_);
    return cat;
}

std::optional<MachineSpec> BambuPresetCatalog::Resolve(std::string_view machine_name,
                                                       NozzleSize nozzle,
                                                       float layer_height_mm) const {
    std::string name(machine_name);
    if (name.empty()) name = default_machine_;

    auto it = machines_.find(name);
    if (it == machines_.end()) {
        spdlog::warn("BambuPresetCatalog::Resolve: machine `{}` not registered", name);
        return std::nullopt;
    }
    const MachineRecord& rec = it->second;

    const std::string nozzle_str = NozzleSizeStr(nozzle);
    auto pit                     = rec.process_template.find(nozzle_str);
    if (pit == rec.process_template.end()) {
        spdlog::warn("BambuPresetCatalog::Resolve: machine `{}` has no nozzle `{}`", name,
                     nozzle_str);
        return std::nullopt;
    }
    auto fit = rec.filament_template.find(nozzle_str);
    if (fit == rec.filament_template.end()) {
        spdlog::warn(
            "BambuPresetCatalog::Resolve: machine `{}` has no filament_template for nozzle `{}`",
            name, nozzle_str);
        return std::nullopt;
    }

    const std::string fname =
        MachineSlug(name) + "_" +
        [&] {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.2fmm", static_cast<double>(layer_height_mm));
            return std::string(buf);
        }() +
        "_" + (nozzle == NozzleSize::N02 ? "n02" : "n04") + ".json";
    auto base_path = data_dir_ / "preset_bases" / fname;
    if (!std::filesystem::exists(base_path)) {
        spdlog::warn(
            "BambuPresetCatalog::Resolve: preset base file missing for `{}` n{} lh{} -> {}", name,
            nozzle_str, layer_height_mm, base_path.string());
        return std::nullopt;
    }

    MachineSpec spec;
    spec.machine_name      = name;
    spec.extruder_topology = rec.extruder_topology;
    spec.nozzles           = rec.nozzles;
    spec.process_template  = pit->second;
    spec.printer_template  = FormatPrinterTemplate(rec.printer_template, nozzle_str);
    spec.filament_template = fit->second;
    spec.preset_base_path  = base_path;

    // compatible_printers: every registered machine of the same topology that
    // also exposes this nozzle. Format: "<machine_name> <nozzle> nozzle".
    spec.compatible_printers.reserve(machines_.size());
    for (const auto& other_name : machine_names_) {
        const auto& other = machines_.at(other_name);
        if (other.extruder_topology != rec.extruder_topology) continue;
        if (other.process_template.find(nozzle_str) == other.process_template.end()) continue;
        spec.compatible_printers.push_back(
            FormatPrinterTemplate(other.printer_template, nozzle_str));
    }

    // Share the catalog's patches with this MachineSpec (shared_ptr aliasing).
    spec.patches = patches_;

    // printer_model is read out of the base file.
    std::ifstream bif(base_path);
    if (bif.is_open()) {
        try {
            nlohmann::json bd = nlohmann::json::parse(bif);
            if (bd.contains("printer_model") && bd["printer_model"].is_string()) {
                spec.printer_model = bd["printer_model"].get<std::string>();
            }
        } catch (const std::exception& e) {
            spdlog::warn("BambuPresetCatalog::Resolve: failed to read printer_model from {}: {}",
                         base_path.string(), e.what());
        }
    }

    return spec;
}

std::vector<std::string> BambuPresetCatalog::ListMachines() const { return machine_names_; }

} // namespace ChromaPrint3D
