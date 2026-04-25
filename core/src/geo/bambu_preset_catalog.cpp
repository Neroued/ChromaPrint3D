#include "chromaprint3d/bambu_preset_catalog.h"

#include "chromaprint3d/error.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>

namespace ChromaPrint3D {

namespace {

const char* NozzleSizeStr(NozzleSize n) { return n == NozzleSize::N02 ? "0.2" : "0.4"; }

/// Validate a slug against `^[a-z0-9_]+$`. Mirrors `_SLUG_RE` in
/// `scripts/build_preset_bases.py` (single source of truth: `machines.json`).
bool IsValidSlug(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        const auto u = static_cast<unsigned char>(c);
        const bool is_lower = (u >= 'a' && u <= 'z');
        const bool is_digit = (u >= '0' && u <= '9');
        const bool is_underscore = (c == '_');
        if (!is_lower && !is_digit && !is_underscore) return false;
    }
    return true;
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

    std::unordered_map<std::string, int> seen_slugs;
    for (auto it = j["machines"].begin(); it != j["machines"].end(); ++it) {
        const std::string& name = it.key();
        const auto& spec_json   = it.value();

        MachineRecord rec;
        rec.extruder_topology = spec_json.value("extruder_topology", std::string{"single"});
        rec.printer_template  = spec_json.value("printer_template", std::string{});

        // Strict slug schema validation (plan v13.1 / m1: single source of truth).
        if (!spec_json.contains("slug") || !spec_json["slug"].is_string()) {
            throw FormatError("machines.json: machine `" + name + "` missing string `slug` field");
        }
        rec.slug = spec_json["slug"].get<std::string>();
        if (!IsValidSlug(rec.slug)) {
            throw FormatError("machines.json: machine `" + name + "` has invalid slug `" +
                              rec.slug + "` (must match ^[a-z0-9_]+$)");
        }
        if (!seen_slugs.emplace(rec.slug, 1).second) {
            throw FormatError("machines.json: duplicate slug `" + rec.slug + "`");
        }

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

    // One-shot scan of base files to populate printer_model cache (plan v13.1 / m2 + s2).
    // Reads `_chromaprint3d_meta.printer_model` first, falling back to top-level
    // `printer_model` field. Avoids re-parsing on every Resolve call.
    std::size_t cache_misses = 0;
    for (const auto& entry : std::filesystem::directory_iterator(bases_dir)) {
        if (!entry.is_regular_file()) continue;
        const auto& path = entry.path();
        if (path.extension() != ".json") continue;
        const std::string stem = path.stem().string();
        std::ifstream bif(path);
        if (!bif.is_open()) {
            ++cache_misses;
            continue;
        }
        try {
            nlohmann::json bd = nlohmann::json::parse(bif);
            std::string pm;
            if (bd.contains("_chromaprint3d_meta") && bd["_chromaprint3d_meta"].is_object()) {
                const auto& meta = bd["_chromaprint3d_meta"];
                if (meta.contains("printer_model") && meta["printer_model"].is_string()) {
                    pm = meta["printer_model"].get<std::string>();
                }
            }
            if (pm.empty() && bd.contains("printer_model") && bd["printer_model"].is_string()) {
                pm = bd["printer_model"].get<std::string>();
            }
            if (!pm.empty()) {
                cat.base_printer_model_cache_.emplace(stem, std::move(pm));
            } else {
                ++cache_misses;
                spdlog::warn("BambuPresetCatalog: base file {} has no printer_model",
                             path.string());
            }
        } catch (const std::exception& e) {
            ++cache_misses;
            spdlog::warn("BambuPresetCatalog: failed to parse base file {}: {}", path.string(),
                         e.what());
        }
    }

    spdlog::info("BambuPresetCatalog loaded: {} machines, default = {}, cached {} base "
                 "printer_model entries ({} misses)",
                 cat.machines_.size(), cat.default_machine_,
                 cat.base_printer_model_cache_.size(), cache_misses);
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

    char lh_buf[16];
    std::snprintf(lh_buf, sizeof(lh_buf), "%.2fmm", static_cast<double>(layer_height_mm));
    const std::string lh_str    = lh_buf;
    const std::string nozzle_tag = nozzle == NozzleSize::N02 ? "n02" : "n04";
    const std::string stem      = rec.slug + "_" + lh_str + "_" + nozzle_tag;
    const std::string fname     = stem + ".json";
    auto base_path              = data_dir_ / "preset_bases" / fname;
    if (!std::filesystem::exists(base_path)) {
        spdlog::warn(
            "BambuPresetCatalog::Resolve: preset base file missing for `{}` n{} lh{} -> {}", name,
            nozzle_str, layer_height_mm, base_path.string());
        return std::nullopt;
    }

    MachineSpec spec;
    spec.machine_name      = name;
    spec.slug              = rec.slug;
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

    // printer_model: pull from the cache populated at LoadFromDir time.
    auto cit = base_printer_model_cache_.find(stem);
    if (cit != base_printer_model_cache_.end()) {
        spec.printer_model = cit->second;
    }
    // (cache miss is non-fatal; spec.printer_model stays empty and downstream
    //  metadata writers omit the printer_model_id plate annotation.)

    return spec;
}

std::vector<std::string> BambuPresetCatalog::ListMachines() const { return machine_names_; }

} // namespace ChromaPrint3D
