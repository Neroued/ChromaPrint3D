#include "chromaprint3d/pipeline.h"
#include "chromaprint3d/vector_proc.h"
#include "chromaprint3d/vector_recipe_map.h"
#include "chromaprint3d/vector_mesh.h"
#include "chromaprint3d/vector_preview.h"
#include "chromaprint3d/export_3mf.h"
#include "chromaprint3d/slicer_preset.h"
#include "chromaprint3d/color_db.h"
#include "chromaprint3d/print_profile.h"
#include "chromaprint3d/error.h"
#include "geo/three_mf_writer.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <cmath>
#include <optional>

namespace ChromaPrint3D {

namespace {

void NotifyProgress(const ProgressCallback& cb, ConvertStage stage, float progress) {
    if (cb) { cb(stage, progress); }
}

std::vector<ColorDB> LoadColorDBsFromPaths(const std::vector<std::string>& resolved_paths) {
    std::vector<ColorDB> dbs;
    dbs.reserve(resolved_paths.size());
    for (const std::string& path : resolved_paths) { dbs.push_back(ColorDB::LoadFromJson(path)); }
    return dbs;
}

struct ResolvedGeometryOptions {
    int base_layers   = 0;
    bool double_sided = false;
};

ResolvedGeometryOptions ResolveGeometryOptions(const ConvertVectorRequest& request,
                                               const PrintProfile& profile) {
    if (request.base_layers < -1) { throw InputError("base_layers must be >= -1"); }

    ResolvedGeometryOptions out;
    out.base_layers = (request.base_layers >= 0) ? request.base_layers : profile.base_layers;
    if (out.base_layers < 0) { throw ConfigError("resolved base_layers must be >= 0"); }
    out.double_sided = request.double_sided;
    return out;
}

constexpr float kLayerPreviewPixelsPerMm = 5.0f;

} // namespace

ConvertResult ConvertVector(const ConvertVectorRequest& request, ProgressCallback progress) {
    spdlog::info("ConvertVector started: svg={}, dbs={}",
                 request.svg_path.empty() ? "(buffer)" : request.svg_path,
                 request.preloaded_dbs.empty() ? request.db_paths.size()
                                               : request.preloaded_dbs.size());

    // === 1. Load resources ===
    NotifyProgress(progress, ConvertStage::LoadingResources, 0.0f);

    std::vector<ColorDB> owned_dbs;
    std::vector<const ColorDB*> db_ptrs;

    if (!request.preloaded_dbs.empty()) {
        db_ptrs = request.preloaded_dbs;
    } else {
        const std::vector<std::string> resolved = ResolveDBPaths(request.db_paths);
        owned_dbs                               = LoadColorDBsFromPaths(resolved);
        db_ptrs.reserve(owned_dbs.size());
        for (const ColorDB& db : owned_dbs) { db_ptrs.push_back(&db); }
    }

    if (db_ptrs.empty()) { throw InputError("No ColorDB available"); }
    spdlog::info("Loaded {} ColorDB(s)", db_ptrs.size());

    std::vector<ColorDB> db_span_storage;
    if (!request.preloaded_dbs.empty()) {
        db_span_storage.reserve(db_ptrs.size());
        for (const ColorDB* p : db_ptrs) { db_span_storage.push_back(*p); }
    }
    const std::vector<ColorDB>& dbs_ref =
        request.preloaded_dbs.empty() ? owned_dbs : db_span_storage;

    std::optional<FilamentConfig> fil_config;
    if (!request.preset_dir.empty()) {
        fil_config.emplace(FilamentConfig::LoadFromDir(request.preset_dir));
    }

    PrintProfile profile     = PrintProfile::BuildFromColorDBs(dbs_ref, request.print_mode,
                                                           fil_config ? &*fil_config : nullptr);
    profile.nozzle_size      = request.nozzle_size;
    profile.face_orientation = request.face_orientation;

    if (!request.allowed_channel_keys.empty()) {
        profile.FilterChannels(request.allowed_channel_keys);
        spdlog::info("Filtered profile palette to {} channel(s)", profile.NumChannels());
    }

    std::optional<ModelPackage> owned_model_pack;
    const ModelPackage* model_pack_ptr = request.preloaded_model_pack;
    if (!model_pack_ptr && !request.model_pack_path.empty()) {
        owned_model_pack.emplace(ModelPackage::LoadFromJson(request.model_pack_path));
        model_pack_ptr = &owned_model_pack.value();
    }

    NotifyProgress(progress, ConvertStage::LoadingResources, 1.0f);

    // === 2. Process SVG ===
    NotifyProgress(progress, ConvertStage::Preprocessing, 0.0f);

    VectorProcConfig vproc_cfg;
    vproc_cfg.target_width_mm           = request.target_width_mm;
    vproc_cfg.target_height_mm          = request.target_height_mm;
    vproc_cfg.tessellation_tolerance_mm = request.tessellation_tolerance_mm;
    vproc_cfg.flip_y                    = request.flip_y;

    VectorProc vproc(vproc_cfg);
    VectorProcResult vimg;
    if (!request.svg_buffer.empty()) {
        vimg = vproc.RunFromBuffer(request.svg_buffer, request.svg_name);
    } else if (!request.svg_path.empty()) {
        vimg = vproc.Run(request.svg_path);
    } else {
        throw InputError("No SVG input provided (neither path nor buffer)");
    }

    NotifyProgress(progress, ConvertStage::Preprocessing, 1.0f);
    spdlog::info("Stage: processing_svg completed, {} shapes, {:.1f}x{:.1f} mm", vimg.shapes.size(),
                 vimg.width_mm, vimg.height_mm);

    // === 3. Match colors ===
    NotifyProgress(progress, ConvertStage::Matching, 0.0f);

    MatchConfig match_cfg;
    match_cfg.color_space  = request.color_space;
    match_cfg.k_candidates = request.k_candidates;

    ModelGateConfig model_gate;
    model_gate.enable     = false;
    model_gate.model_only = false;
    if (model_pack_ptr) {
        model_gate.model_only = request.model_only;
        model_gate.enable     = request.model_only ? true : request.model_enable;
        model_gate.threshold  = (request.model_threshold >= 0.0f)
                                    ? request.model_threshold
                                    : model_pack_ptr->default_threshold;
        model_gate.margin =
            (request.model_margin >= 0.0f) ? request.model_margin : model_pack_ptr->default_margin;
    }

    MatchStats match_stats;
    VectorRecipeMap recipe_map = VectorRecipeMap::Match(vimg, dbs_ref, profile, match_cfg,
                                                        model_pack_ptr, model_gate, &match_stats);

    NotifyProgress(progress, ConvertStage::Matching, 1.0f);
    spdlog::info("Stage: matching completed, {} entries, avg_de={:.2f}", recipe_map.entries.size(),
                 match_stats.avg_db_de);

    // === 4. Build 3D model ===
    NotifyProgress(progress, ConvertStage::BuildingModel, 0.0f);

    const float resolved_layer_height =
        (request.layer_height_mm > 0.0f)
            ? request.layer_height_mm
            : (profile.layer_height_mm > 0.0f ? profile.layer_height_mm : 0.08f);
    const ResolvedGeometryOptions geometry = ResolveGeometryOptions(request, profile);

    const float transparent_mm = request.transparent_layer_mm;
    const bool has_transparent = transparent_mm > 0.0f;
    if (has_transparent) {
        if (request.face_orientation != FaceOrientation::FaceDown) {
            throw InputError("transparent_layer_mm requires FaceDown orientation");
        }
        if (geometry.double_sided) {
            throw InputError("transparent_layer_mm not supported with double_sided");
        }
        if (transparent_mm != 0.04f && transparent_mm != 0.08f) {
            throw InputError("transparent_layer_mm must be 0.04 or 0.08");
        }
    }

    VectorMeshConfig mesh_cfg;
    mesh_cfg.layer_height_mm      = resolved_layer_height;
    mesh_cfg.base_layers          = geometry.base_layers;
    mesh_cfg.double_sided         = geometry.double_sided;
    mesh_cfg.transparent_layer_mm = transparent_mm;

    {
        constexpr float kDefaultGap = 0.005f;
        float gap                   = kDefaultGap;
        const float max_gap         = 0.25f * mesh_cfg.layer_height_mm;
        if (gap > max_gap) {
            spdlog::warn("base_color_gap_mm {:.4f} clamped to {:.4f} (0.25*layer_height)", gap,
                         max_gap);
            gap = max_gap;
        }
        if (geometry.double_sided && geometry.base_layers > 0 &&
            gap >= static_cast<float>(geometry.base_layers) * mesh_cfg.layer_height_mm) {
            const float limit = static_cast<float>(geometry.base_layers) * mesh_cfg.layer_height_mm;
            spdlog::warn("base_color_gap_mm {:.4f} clamped to {:.4f} (base thickness)", gap,
                         limit * 0.5f);
            gap = limit * 0.5f;
        }
        mesh_cfg.base_color_gap_mm = gap;
    }

    std::vector<Mesh> meshes = BuildVectorMeshes(vimg.shapes, recipe_map, mesh_cfg);

    NotifyProgress(progress, ConvertStage::BuildingModel, 1.0f);

    // === 5. Export ===
    NotifyProgress(progress, ConvertStage::Exporting, 0.0f);

    ConvertResult result;
    result.stats              = match_stats;
    result.physical_width_mm  = vimg.width_mm;
    result.physical_height_mm = vimg.height_mm;

    if (request.generate_preview) {
        result.preview_png = RenderVectorPreviewPng(vimg, recipe_map, profile.palette);
    }
    if (request.generate_source_mask) {
        result.source_mask_png = RenderVectorSourceMaskPng(vimg, recipe_map);
    }

    result.layer_previews.layers      = recipe_map.color_layers;
    result.layer_previews.layer_order = recipe_map.layer_order;
    result.layer_previews.width =
        std::max(1, static_cast<int>(std::ceil(vimg.width_mm * kLayerPreviewPixelsPerMm)));
    result.layer_previews.height =
        std::max(1, static_cast<int>(std::ceil(vimg.height_mm * kLayerPreviewPixelsPerMm)));
    result.layer_previews.palette.reserve(profile.palette.size());
    for (std::size_t i = 0; i < profile.palette.size(); ++i) {
        const auto& channel = profile.palette[i];
        result.layer_previews.palette.push_back(
            LayerPreviewChannel{static_cast<int>(i), channel.color, channel.material});
    }
    result.layer_previews.layer_pngs =
        RenderVectorLayerPreviewPngs(vimg, recipe_map, profile.palette, kLayerPreviewPixelsPerMm);

    PrintProfile export_profile = profile;
    export_profile.base_layers  = geometry.base_layers;

    int base_ch          = geometry.base_layers > 0 ? profile.base_channel_idx : -1;
    const bool file_only = !request.output_3mf_path.empty() && request.output_3mf_file_only;

    auto write_buffer_to_file = [&](const std::vector<uint8_t>& buf) {
        detail::WriteBufferToFileAtomically(request.output_3mf_path, buf);
        spdlog::info("Wrote 3MF to {}", request.output_3mf_path);
    };

    if (has_transparent) {
        auto ns = BuildMeshNamesAndSlots(meshes.size(), profile.palette, base_ch,
                                         geometry.base_layers, true);

        std::vector<uint8_t> buffer;
        if (!request.preset_dir.empty()) {
            auto preset = SlicerPreset::FromProfile(request.preset_dir, export_profile,
                                                    fil_config ? &*fil_config : nullptr,
                                                    geometry.double_sided);
            preset.custom_base_layers   = (request.base_layers >= 0);
            preset.transparent_layer_mm = transparent_mm;
            FilamentSlot t_slot;
            t_slot.type        = "PLA";
            t_slot.colour      = "#FEFEFE";
            t_slot.settings_id = "Bambu PLA Basic @BBL P2S";
            preset.filaments.push_back(std::move(t_slot));

            if (!preset.preset_json_path.empty()) {
                buffer = Export3mfFromMeshes(meshes, profile.palette, ns.names, ns.slots, preset,
                                             request.face_orientation, vimg.name);
                spdlog::info("Vector pipeline (transparent): preset from {}",
                             preset.preset_json_path);
            } else {
                spdlog::warn("Vector pipeline (transparent): preset not found in {}",
                             request.preset_dir);
                buffer = Export3mfFromMeshes(meshes, profile.palette, ns.names,
                                             request.face_orientation);
            }
        } else {
            buffer =
                Export3mfFromMeshes(meshes, profile.palette, ns.names, request.face_orientation);
        }

        if (!request.output_3mf_path.empty()) { write_buffer_to_file(buffer); }
        if (!file_only) { result.model_3mf = std::move(buffer); }
    } else if (file_only) {
        auto dir = std::filesystem::path(request.output_3mf_path).parent_path();
        if (!dir.empty()) { std::filesystem::create_directories(dir); }

        if (!request.preset_dir.empty()) {
            auto preset = SlicerPreset::FromProfile(request.preset_dir, export_profile,
                                                    fil_config ? &*fil_config : nullptr,
                                                    geometry.double_sided);
            preset.custom_base_layers = (request.base_layers >= 0);
            if (!preset.preset_json_path.empty()) {
                Export3mfFromMeshesToFile(request.output_3mf_path, meshes, profile.palette, base_ch,
                                          geometry.base_layers, preset, request.face_orientation,
                                          vimg.name);
                spdlog::info("Vector pipeline: injected slicer preset from {}",
                             preset.preset_json_path);
            } else {
                spdlog::warn("Vector pipeline: preset file not found in {}, exporting standard 3MF",
                             request.preset_dir);
                Export3mfFromMeshesToFile(request.output_3mf_path, meshes, profile.palette, base_ch,
                                          geometry.base_layers, request.face_orientation);
            }
        } else {
            Export3mfFromMeshesToFile(request.output_3mf_path, meshes, profile.palette, base_ch,
                                      geometry.base_layers, request.face_orientation);
        }
    } else {
        if (!request.preset_dir.empty()) {
            auto preset = SlicerPreset::FromProfile(request.preset_dir, export_profile,
                                                    fil_config ? &*fil_config : nullptr,
                                                    geometry.double_sided);
            preset.custom_base_layers = (request.base_layers >= 0);
            if (!preset.preset_json_path.empty()) {
                result.model_3mf =
                    Export3mfFromMeshes(meshes, profile.palette, base_ch, geometry.base_layers,
                                        preset, request.face_orientation, vimg.name);
                spdlog::info("Vector pipeline: injected slicer preset from {}",
                             preset.preset_json_path);
            } else {
                spdlog::warn("Vector pipeline: preset file not found in {}, exporting standard 3MF",
                             request.preset_dir);
                result.model_3mf =
                    Export3mfFromMeshes(meshes, profile.palette, base_ch, geometry.base_layers,
                                        request.face_orientation);
            }
        } else {
            result.model_3mf = Export3mfFromMeshes(meshes, profile.palette, base_ch,
                                                   geometry.base_layers, request.face_orientation);
        }

        if (!request.output_3mf_path.empty()) { write_buffer_to_file(result.model_3mf); }
    }

    NotifyProgress(progress, ConvertStage::Exporting, 1.0f);
    spdlog::info("ConvertVector completed: 3mf={} bytes, preview={} bytes, source_mask={} bytes, "
                 "physical={:.1f}x{:.1f} mm",
                 result.model_3mf.size(), result.preview_png.size(), result.source_mask_png.size(),
                 result.physical_width_mm, result.physical_height_mm);

    return result;
}

} // namespace ChromaPrint3D
