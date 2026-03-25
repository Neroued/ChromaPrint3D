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

std::size_t MatchVectorResult::EstimateBytes() const {
    std::size_t bytes = sizeof(*this);

    for (const auto& e : recipe_map.entries) {
        bytes += sizeof(VectorRecipeMap::ShapeEntry) + e.recipe.capacity();
    }

    for (const auto& s : proc_result.shapes) {
        for (const auto& c : s.contours) { bytes += c.capacity() * sizeof(Vec2f); }
    }

    for (const auto& ch : profile.palette) {
        bytes += ch.color.capacity() + ch.material.capacity() + ch.hex_color.capacity();
    }

    for (const auto& db : dbs) {
        for (const auto& entry : db.entries) { bytes += sizeof(Entry) + entry.recipe.capacity(); }
        for (const auto& ch : db.palette) {
            bytes += ch.color.capacity() + ch.material.capacity() + ch.hex_color.capacity();
        }
    }

    bytes += preset_dir.capacity();
    return bytes;
}

MatchVectorResult MatchVector(const ConvertVectorRequest& request, ProgressCallback progress) {
    spdlog::info("MatchVector started: svg={}, dbs={}",
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
    std::vector<ColorDB> all_dbs =
        request.preloaded_dbs.empty() ? std::move(owned_dbs) : std::move(db_span_storage);

    std::optional<FilamentConfig> fil_config;
    if (!request.preset_dir.empty()) {
        fil_config.emplace(FilamentConfig::LoadFromDir(request.preset_dir));
    }

    PrintProfile profile     = PrintProfile::BuildFromColorDBs(all_dbs, request.print_mode,
                                                           fil_config ? &*fil_config : nullptr);
    profile.nozzle_size      = request.nozzle_size;
    profile.face_orientation = request.face_orientation;

    if (!request.allowed_channel_keys.empty()) {
        profile.FilterChannels(request.allowed_channel_keys);
        spdlog::info("Filtered profile palette to {} channel(s)", profile.NumChannels());
    }

    const ModelPackage* model_pack_ptr = request.preloaded_model_pack;
    std::optional<ModelPackage> owned_model_pack;
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
    vproc_cfg.gradient_pixel_mm         = request.gradient_pixel_mm;

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
    VectorRecipeMap recipe_map = VectorRecipeMap::Match(vimg, all_dbs, profile, match_cfg,
                                                        model_pack_ptr, model_gate, &match_stats);

    NotifyProgress(progress, ConvertStage::Matching, 1.0f);
    spdlog::info("MatchVector completed: {} entries, avg_de={:.2f}", recipe_map.entries.size(),
                 match_stats.avg_db_de);

    MatchVectorResult mr;
    mr.recipe_map           = std::move(recipe_map);
    mr.proc_result          = std::move(vimg);
    mr.profile              = std::move(profile);
    mr.stats                = match_stats;
    mr.layer_height_mm      = request.layer_height_mm;
    mr.base_layers          = request.base_layers;
    mr.custom_base_layers   = (request.base_layers >= 0);
    mr.double_sided         = request.double_sided;
    mr.transparent_layer_mm = request.transparent_layer_mm;
    mr.nozzle_size          = request.nozzle_size;
    mr.face_orientation     = request.face_orientation;
    mr.preset_dir           = request.preset_dir;
    mr.dbs                  = std::move(all_dbs);
    mr.match_config         = match_cfg;
    mr.model_gate           = model_gate;
    mr.generate_preview     = request.generate_preview;
    mr.generate_source_mask = request.generate_source_mask;
    return mr;
}

ConvertResult GenerateVectorModel(MatchVectorResult& mr, ProgressCallback progress) {
    const auto& vimg       = mr.proc_result;
    const auto& recipe_map = mr.recipe_map;
    const auto& profile    = mr.profile;

    std::optional<FilamentConfig> fil_config;
    if (!mr.preset_dir.empty()) { fil_config.emplace(FilamentConfig::LoadFromDir(mr.preset_dir)); }

    // === 4. Build 3D model ===
    NotifyProgress(progress, ConvertStage::BuildingModel, 0.0f);

    const float resolved_layer_height =
        (mr.layer_height_mm > 0.0f)
            ? mr.layer_height_mm
            : (profile.layer_height_mm > 0.0f ? profile.layer_height_mm : 0.08f);

    const int resolved_base_layers = mr.custom_base_layers ? mr.base_layers : profile.base_layers;
    if (resolved_base_layers < 0) { throw ConfigError("resolved base_layers must be >= 0"); }

    const float transparent_mm = mr.transparent_layer_mm;
    const bool has_transparent = transparent_mm > 0.0f;
    if (has_transparent) {
        if (mr.face_orientation != FaceOrientation::FaceDown) {
            throw InputError("transparent_layer_mm requires FaceDown orientation");
        }
        if (mr.double_sided) {
            throw InputError("transparent_layer_mm not supported with double_sided");
        }
        if (transparent_mm != 0.04f && transparent_mm != 0.08f) {
            throw InputError("transparent_layer_mm must be 0.04 or 0.08");
        }
    }

    VectorMeshConfig mesh_cfg;
    mesh_cfg.layer_height_mm      = resolved_layer_height;
    mesh_cfg.base_layers          = resolved_base_layers;
    mesh_cfg.double_sided         = mr.double_sided;
    mesh_cfg.transparent_layer_mm = transparent_mm;

    {
        constexpr float kDefaultGap = 0.005f;
        float gap                   = kDefaultGap;
        const float max_gap         = 0.25f * mesh_cfg.layer_height_mm;
        if (gap > max_gap) { gap = max_gap; }
        if (mr.double_sided && resolved_base_layers > 0 &&
            gap >= static_cast<float>(resolved_base_layers) * mesh_cfg.layer_height_mm) {
            gap = static_cast<float>(resolved_base_layers) * mesh_cfg.layer_height_mm * 0.5f;
        }
        mesh_cfg.base_color_gap_mm = gap;
    }

    std::vector<Mesh> meshes = BuildVectorMeshes(vimg.shapes, recipe_map, mesh_cfg);

    NotifyProgress(progress, ConvertStage::BuildingModel, 1.0f);

    // === 5. Export ===
    NotifyProgress(progress, ConvertStage::Exporting, 0.0f);

    ConvertResult result;
    result.stats              = mr.stats;
    result.physical_width_mm  = vimg.width_mm;
    result.physical_height_mm = vimg.height_mm;

    if (mr.generate_preview) {
        result.preview_png = RenderVectorPreviewPng(vimg, recipe_map, profile.palette);
    }
    if (mr.generate_source_mask) {
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
    export_profile.base_layers  = resolved_base_layers;

    int base_ch = resolved_base_layers > 0 ? profile.base_channel_idx : -1;

    if (has_transparent) {
        auto ns = BuildMeshNamesAndSlots(meshes.size(), profile.palette, base_ch,
                                         resolved_base_layers, true);

        if (!mr.preset_dir.empty()) {
            auto preset =
                SlicerPreset::FromProfile(mr.preset_dir, export_profile,
                                          fil_config ? &*fil_config : nullptr, mr.double_sided);
            preset.custom_base_layers   = mr.custom_base_layers;
            preset.transparent_layer_mm = transparent_mm;
            FilamentSlot t_slot;
            t_slot.type        = "PLA";
            t_slot.colour      = "#FEFEFE";
            t_slot.settings_id = "Bambu PLA Basic @BBL P2S";
            preset.filaments.push_back(std::move(t_slot));

            if (!preset.preset_json_path.empty()) {
                result.model_3mf = Export3mfFromMeshes(meshes, profile.palette, ns.names, ns.slots,
                                                       preset, mr.face_orientation, vimg.name);
            } else {
                result.model_3mf =
                    Export3mfFromMeshes(meshes, profile.palette, ns.names, mr.face_orientation);
            }
        } else {
            result.model_3mf =
                Export3mfFromMeshes(meshes, profile.palette, ns.names, mr.face_orientation);
        }
    } else {
        if (!mr.preset_dir.empty()) {
            auto preset =
                SlicerPreset::FromProfile(mr.preset_dir, export_profile,
                                          fil_config ? &*fil_config : nullptr, mr.double_sided);
            preset.custom_base_layers = mr.custom_base_layers;
            if (!preset.preset_json_path.empty()) {
                result.model_3mf =
                    Export3mfFromMeshes(meshes, profile.palette, base_ch, resolved_base_layers,
                                        preset, mr.face_orientation, vimg.name);
            } else {
                result.model_3mf = Export3mfFromMeshes(meshes, profile.palette, base_ch,
                                                       resolved_base_layers, mr.face_orientation);
            }
        } else {
            result.model_3mf = Export3mfFromMeshes(meshes, profile.palette, base_ch,
                                                   resolved_base_layers, mr.face_orientation);
        }
    }

    NotifyProgress(progress, ConvertStage::Exporting, 1.0f);
    spdlog::info("GenerateVectorModel completed: 3mf={} bytes, preview={} bytes, "
                 "source_mask={} bytes, physical={:.1f}x{:.1f} mm",
                 result.model_3mf.size(), result.preview_png.size(), result.source_mask_png.size(),
                 result.physical_width_mm, result.physical_height_mm);

    return result;
}

ConvertResult ConvertVector(const ConvertVectorRequest& request, ProgressCallback progress) {
    auto match_result    = MatchVector(request, progress);
    ConvertResult result = GenerateVectorModel(match_result, progress);

    if (!request.output_3mf_path.empty() && !result.model_3mf.empty()) {
        detail::WriteBufferToFileAtomically(request.output_3mf_path, result.model_3mf);
        spdlog::info("Wrote 3MF to {}", request.output_3mf_path);
        if (request.output_3mf_file_only) {
            result.model_3mf.clear();
            result.model_3mf.shrink_to_fit();
        }
    }

    return result;
}

} // namespace ChromaPrint3D
