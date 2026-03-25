#include "chromaprint3d/error.h"
#include "chromaprint3d/export_3mf.h"
#include "chromaprint3d/mesh.h"
#include "chromaprint3d/voxel.h"
#include "bambu_metadata.h"

#include <neroued/3mf/neroued_3mf.h>

#include <spdlog/spdlog.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace ChromaPrint3D {
namespace {

// ── Mesh type conversion ────────────────────────────────────────────────────

bool IsDegenerateTriangle(const Vec3f& a, const Vec3f& b, const Vec3f& c) {
    Vec3f ab = b - a;
    Vec3f ac = c - a;
    float cx = ab.y * ac.z - ab.z * ac.y;
    float cy = ab.z * ac.x - ab.x * ac.z;
    float cz = ab.x * ac.y - ab.y * ac.x;
    return (cx * cx + cy * cy + cz * cz) <= 1e-12f;
}

neroued_3mf::Mesh ConvertMesh(const Mesh& src) {
    neroued_3mf::Mesh dst;
    const std::size_t vc = src.vertices.size();

    dst.vertices.reserve(vc);
    for (const Vec3f& v : src.vertices) { dst.vertices.push_back({v.x, v.y, v.z}); }

    dst.triangles.reserve(src.indices.size());
    std::size_t dropped = 0;
    for (const Vec3i& tri : src.indices) {
        if (tri.x < 0 || tri.y < 0 || tri.z < 0 || static_cast<std::size_t>(tri.x) >= vc ||
            static_cast<std::size_t>(tri.y) >= vc || static_cast<std::size_t>(tri.z) >= vc) {
            throw InputError("Mesh index out of range");
        }
        if (tri.x == tri.y || tri.y == tri.z || tri.x == tri.z) {
            ++dropped;
            continue;
        }
        const Vec3f& v0 = src.vertices[static_cast<std::size_t>(tri.x)];
        const Vec3f& v1 = src.vertices[static_cast<std::size_t>(tri.y)];
        const Vec3f& v2 = src.vertices[static_cast<std::size_t>(tri.z)];
        if (IsDegenerateTriangle(v0, v1, v2)) {
            ++dropped;
            continue;
        }
        dst.triangles.push_back({static_cast<uint32_t>(tri.x), static_cast<uint32_t>(tri.y),
                                 static_cast<uint32_t>(tri.z)});
    }
    if (dropped > 0) { spdlog::debug("ConvertMesh: filtered {} degenerate triangle(s)", dropped); }
    return dst;
}

// ── Hex color normalization ─────────────────────────────────────────────────

neroued_3mf::Color ParseHexColor(const std::string& hex) {
    if ((hex.size() == 7 || hex.size() == 9) && hex[0] == '#') {
        unsigned long val = std::strtoul(hex.c_str() + 1, nullptr, 16);
        if (hex.size() == 9) {
            return {static_cast<uint8_t>((val >> 24) & 0xFF),
                    static_cast<uint8_t>((val >> 16) & 0xFF),
                    static_cast<uint8_t>((val >> 8) & 0xFF), static_cast<uint8_t>(val & 0xFF)};
        }
        return {static_cast<uint8_t>((val >> 16) & 0xFF), static_cast<uint8_t>((val >> 8) & 0xFF),
                static_cast<uint8_t>(val & 0xFF), 255};
    }
    return {255, 255, 255, 255};
}

// ── Object naming helpers ───────────────────────────────────────────────────

std::string BuildObjectNameFromPalette(std::size_t idx, const std::vector<Channel>& palette,
                                       int base_channel_idx, int base_layers) {
    if (idx == palette.size() && base_layers > 0) {
        std::string name = "Base";
        if (base_channel_idx >= 0 && base_channel_idx < static_cast<int>(palette.size())) {
            const Channel& base_ch = palette[static_cast<size_t>(base_channel_idx)];
            if (!base_ch.material.empty() && base_ch.material != "Default Material") {
                name += " - " + base_ch.material;
            }
        }
        return name;
    }
    if (idx < palette.size()) {
        const Channel& ch = palette[idx];
        std::string name  = ch.color.empty() ? ("Channel " + std::to_string(idx)) : ch.color;
        if (!ch.material.empty() && ch.material != "Default Material") {
            name += " - " + ch.material;
        }
        return name;
    }
    return "Channel " + std::to_string(idx);
}

std::string BuildObjectName(const ModelIR& model_ir, std::size_t idx) {
    return BuildObjectNameFromPalette(idx, model_ir.palette, model_ir.base_channel_idx,
                                      model_ir.base_layers);
}

// ── Face orientation ────────────────────────────────────────────────────────

void RotateMeshesFaceDownInPlace(std::vector<Mesh>& meshes) {
    float min_x     = std::numeric_limits<float>::infinity();
    float max_x     = -std::numeric_limits<float>::infinity();
    float min_z     = std::numeric_limits<float>::infinity();
    float max_z     = -std::numeric_limits<float>::infinity();
    bool has_vertex = false;

    for (const Mesh& mesh : meshes) {
        for (const Vec3f& v : mesh.vertices) {
            min_x      = std::min(min_x, v.x);
            max_x      = std::max(max_x, v.x);
            min_z      = std::min(min_z, v.z);
            max_z      = std::max(max_z, v.z);
            has_vertex = true;
        }
    }

    if (!has_vertex) { return; }

    const float sum_x = min_x + max_x;
    const float sum_z = min_z + max_z;
    for (Mesh& mesh : meshes) {
        for (Vec3f& v : mesh.vertices) {
            v.x = sum_x - v.x;
            v.z = sum_z - v.z;
        }
    }
}

void OrientMeshesInPlace(std::vector<Mesh>& meshes, FaceOrientation face_orientation) {
    if (face_orientation != FaceOrientation::FaceDown) { return; }
    RotateMeshesFaceDownInPlace(meshes);
}

// ── Utility: string to bytes ────────────────────────────────────────────────

std::vector<uint8_t> ToBytes(const std::string& s) { return {s.begin(), s.end()}; }

// ── Builder helpers ─────────────────────────────────────────────────────────

struct InputObject {
    std::string name;
    std::string display_color_hex;
    const Mesh* mesh = nullptr;
    int slot         = 0;
};

std::vector<InputObject> BuildInputObjects(const std::vector<Mesh>& meshes,
                                           const std::function<std::string(std::size_t)>& name_fn,
                                           const std::function<std::string(std::size_t)>& color_fn,
                                           const std::function<int(std::size_t)>& slot_fn = {}) {
    std::vector<InputObject> objects;
    objects.reserve(meshes.size());
    for (std::size_t i = 0; i < meshes.size(); ++i) {
        if (meshes[i].vertices.empty() || meshes[i].indices.empty()) {
            spdlog::warn("BuildInputObjects: mesh {} ('{}') skipped (empty)", i, name_fn(i));
            continue;
        }
        objects.push_back({
            .name              = name_fn(i),
            .display_color_hex = color_fn(i),
            .mesh              = &meshes[i],
            .slot              = slot_fn ? slot_fn(i) : static_cast<int>(i) + 1,
        });
    }
    return objects;
}

void EnsureNonEmptyGeometry(const std::vector<InputObject>& objects, std::size_t total_input = 0) {
    if (objects.empty()) {
        std::string msg = "No geometry to export";
        if (total_input > 0) {
            msg += " (all " + std::to_string(total_input) + " input mesh(es) were empty)";
        }
        throw InputError(msg);
    }
}

neroued_3mf::WriteOptions DefaultWriteOptions() { return {}; }

neroued_3mf::Document BuildStandardDocument(const std::vector<InputObject>& objects) {
    neroued_3mf::DocumentBuilder builder;
    builder.SetUnit(neroued_3mf::Unit::Millimeter);
    builder.AddMetadata("Application", "ChromaPrint3D");

    std::vector<neroued_3mf::BaseMaterial> materials;
    materials.reserve(objects.size());
    for (const auto& obj : objects) {
        materials.push_back({obj.name, ParseHexColor(obj.display_color_hex)});
    }
    uint32_t bmg_id = builder.AddBaseMaterialGroup(std::move(materials));

    for (std::size_t i = 0; i < objects.size(); ++i) {
        neroued_3mf::Mesh converted = ConvertMesh(*objects[i].mesh);
        if (converted.triangles.empty()) {
            spdlog::warn("BuildStandardDocument: object '{}' dropped (all triangles degenerate)",
                         objects[i].name);
            continue;
        }
        uint32_t oid = builder.AddMeshObject(objects[i].name, std::move(converted), bmg_id,
                                             static_cast<uint32_t>(i));
        builder.AddBuildItem(oid);
    }

    return builder.Build();
}

neroued_3mf::Document BuildBambuDocument(const std::vector<InputObject>& objects,
                                         const SlicerPreset& preset,
                                         const std::string& model_name) {
    neroued_3mf::DocumentBuilder builder;
    builder.SetUnit(neroued_3mf::Unit::Millimeter);
    builder.AddMetadata("Application",
                        std::string("BambuStudio-") + preset_defaults::kBambuStudioVersion);
    builder.AddMetadata("BambuStudio:3mfVersion", "1");
    builder.AddNamespace("BambuStudio", "http://schemas.bambulab.com/package/2021");

    // Compute bed centering transform before enabling production
    float min_x = std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float min_y = std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    for (const auto& obj : objects) {
        if (!obj.mesh) continue;
        for (const Vec3f& v : obj.mesh->vertices) {
            if (v.x < min_x) min_x = v.x;
            if (v.x > max_x) max_x = v.x;
            if (v.y < min_y) min_y = v.y;
            if (v.y > max_y) max_y = v.y;
        }
    }
    constexpr float kBedCenterX = 128.0f;
    constexpr float kBedCenterY = 128.0f;
    float center_tx             = 0.0f;
    float center_ty             = 0.0f;
    if (std::isfinite(min_x) && std::isfinite(max_x) && std::isfinite(min_y) &&
        std::isfinite(max_y)) {
        center_tx = kBedCenterX - (min_x + max_x) * 0.5f;
        center_ty = kBedCenterY - (min_y + max_y) * 0.5f;
    }

    builder.EnableProduction(neroued_3mf::Transform::Translation(center_tx, center_ty, 0));
    builder.SetProductionMergeObjects(true);
    builder.AddExternalModelMetadata("BambuStudio:3mfVersion", "1");

    detail::ExportedGroup group;
    group.assembly_name = model_name.empty() ? "ChromaPrint3D-Model" : model_name;

    std::vector<uint32_t> object_ids;
    int part_id = 1;
    for (std::size_t i = 0; i < objects.size(); ++i) {
        neroued_3mf::Mesh converted = ConvertMesh(*objects[i].mesh);
        int face_count              = static_cast<int>(converted.triangles.size());
        if (converted.triangles.empty()) {
            spdlog::warn("BuildBambuDocument: object '{}' dropped (all triangles degenerate)",
                         objects[i].name);
            continue;
        }

        uint32_t oid = builder.AddMeshObject(objects[i].name, std::move(converted));
        builder.AddBuildItem(oid);
        object_ids.push_back(oid);

        group.total_face_count += face_count;
        group.objects.push_back(detail::ExportedObject{
            .name          = objects[i].name,
            .filament_slot = objects[i].slot,
            .part_id       = part_id,
            .face_count    = face_count,
        });
        ++part_id;
    }

    group.assembly_object_id = static_cast<int>(object_ids.empty() ? 1 : object_ids.back() + 1);
    group.offset_x           = center_tx;
    group.offset_y           = center_ty;

    // Bambu metadata files
    auto addMeta = [&](const std::string& path, const std::string& ct, const std::string& data) {
        builder.AddCustomPart({path, ct, ToBytes(data)});
    };
    addMeta("Metadata/project_settings.config", "application/octet-stream",
            detail::BuildProjectSettings(preset));
    addMeta("Metadata/process_settings_1.config", "application/octet-stream",
            detail::BuildEmbeddedProcessPreset(preset));
    addMeta("Metadata/model_settings.config", "application/octet-stream",
            detail::BuildModelSettings(group));
    addMeta("Metadata/slice_info.config", "application/octet-stream", detail::BuildSliceInfo());
    addMeta("Metadata/cut_information.xml", "application/xml", detail::BuildCutInformation(group));
    addMeta("Metadata/filament_sequence.json", "application/json", detail::BuildFilamentSequence());

    std::string layer_ranges = detail::BuildLayerConfigRanges(preset);
    if (!layer_ranges.empty()) {
        addMeta("Metadata/layer_config_ranges.xml", "application/xml", std::move(layer_ranges));
    }

    // OPC relationships for Bambu metadata
    constexpr const char* kBambuSettingsRelType =
        "http://schemas.bambulab.com/package/2021/settings";
    const std::array<const char*, 6> metadata_targets = {
        "/Metadata/project_settings.config", "/Metadata/process_settings_1.config",
        "/Metadata/model_settings.config",   "/Metadata/slice_info.config",
        "/Metadata/cut_information.xml",     "/Metadata/filament_sequence.json",
    };
    for (std::size_t i = 0; i < metadata_targets.size(); ++i) {
        builder.AddCustomRelationship({
            .source_part = "3D/3dmodel.model",
            .id          = "bambuRel" + std::to_string(i),
            .type        = kBambuSettingsRelType,
            .target      = metadata_targets[i],
        });
    }

    builder.AddCustomContentType({"config", "application/octet-stream"});
    builder.AddCustomContentType({"xml", "application/xml"});
    builder.AddCustomContentType({"json", "application/json"});

    return builder.Build();
}

std::vector<uint8_t> WriteDocument(const neroued_3mf::Document& doc) {
    auto buffer = neroued_3mf::WriteToBuffer(doc, DefaultWriteOptions());
    spdlog::info("WriteDocument: {} bytes", buffer.size());
    return buffer;
}

void WriteDocumentToFile(const std::string& path, const neroued_3mf::Document& doc) {
    neroued_3mf::WriteToFile(path, doc, DefaultWriteOptions());
    spdlog::info("WriteDocumentToFile: written to {}", path);
}

// ── Color lookup helpers ────────────────────────────────────────────────────

std::string PaletteColor(std::size_t idx, const std::vector<Channel>& palette, int base_channel_idx,
                         int base_layers) {
    if (idx < palette.size()) return palette[idx].hex_color;
    if (idx >= palette.size() && base_layers > 0 && base_channel_idx >= 0 &&
        static_cast<std::size_t>(base_channel_idx) < palette.size()) {
        return palette[static_cast<std::size_t>(base_channel_idx)].hex_color;
    }
    return {};
}

int PaletteSlot(std::size_t idx, const std::vector<Channel>& palette, int base_channel_idx,
                int base_layers) {
    if (idx < palette.size()) return static_cast<int>(idx) + 1;
    if (base_layers > 0 && base_channel_idx >= 0) return base_channel_idx + 1;
    return static_cast<int>(idx) + 1;
}

} // namespace

// ── BuildMeshes (unchanged) ─────────────────────────────────────────────────

std::vector<Mesh> BuildMeshes(const ModelIR& model_ir, const BuildMeshConfig& cfg) {
    if (model_ir.voxel_grids.empty()) { throw InputError("ModelIR voxel_grids is empty"); }

    const auto n           = static_cast<int>(model_ir.voxel_grids.size());
    const int num_channels = static_cast<int>(model_ir.palette.size());
    const bool has_base    = model_ir.base_layers > 0;
    const float half_gap   = cfg.base_color_gap_mm * 0.5f;
    const bool apply_gap   = has_base && half_gap > 0.0f;

    std::vector<Mesh> meshes(static_cast<std::size_t>(n));

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        const VoxelGrid& grid = model_ir.voxel_grids[static_cast<std::size_t>(i)];
        if (grid.width <= 0 || grid.height <= 0 || grid.num_layers <= 0) {
            spdlog::warn("BuildMeshes: grid {} skipped (dimensions {}x{}x{})", i, grid.width,
                         grid.height, grid.num_layers);
            continue;
        }
        if (grid.ooc.empty()) {
            spdlog::warn("BuildMeshes: grid {} skipped (empty voxel data)", i);
            continue;
        }

        BuildMeshConfig per_grid = cfg;
        if (apply_gap) {
            const bool is_base = (i == num_channels);
            if (cfg.double_sided) {
                const int z_bot = model_ir.color_layers;
                const int z_top = model_ir.color_layers + model_ir.base_layers;
                if (is_base) {
                    per_grid.interface_offsets[0] = {z_bot, +half_gap};
                    per_grid.interface_offsets[1] = {z_top, -half_gap};
                } else {
                    per_grid.interface_offsets[0] = {z_bot, -half_gap};
                    per_grid.interface_offsets[1] = {z_top, +half_gap};
                }
            } else {
                const int z_iface = model_ir.base_layers;
                if (is_base) {
                    per_grid.interface_offsets[0] = {z_iface, -half_gap};
                } else {
                    per_grid.interface_offsets[0] = {z_iface, +half_gap};
                }
            }
        }
        meshes[static_cast<std::size_t>(i)] = Mesh::Build(grid, per_grid);
    }

    std::size_t total_verts = 0;
    std::size_t total_tris  = 0;
    for (const auto& mesh : meshes) {
        total_verts += mesh.vertices.size();
        total_tris += mesh.indices.size();
    }
    spdlog::debug("Mesh::Build: {} grids, total vertices={}, triangles={}", n, total_verts,
                  total_tris);
    return meshes;
}

Mesh BuildTransparentLayerFromModelIR(const ModelIR& model_ir, float pixel_mm,
                                      float layer_height_mm, float thickness_mm) {
    if (model_ir.voxel_grids.empty()) { return {}; }

    const int w = model_ir.width;
    const int h = model_ir.height;

    VoxelGrid tg;
    tg.width      = w;
    tg.height     = h;
    tg.num_layers = 1;
    tg.ooc.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0);

    for (const auto& grid : model_ir.voxel_grids) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (tg.Get(x, y, 0)) continue;
                for (int z = 0; z < grid.num_layers; ++z) {
                    if (grid.Get(x, y, z)) {
                        tg.Set(x, y, 0, true);
                        break;
                    }
                }
            }
        }
    }

    BuildMeshConfig mcfg;
    mcfg.pixel_mm        = pixel_mm;
    mcfg.layer_height_mm = thickness_mm;
    Mesh mesh            = Mesh::Build(tg, mcfg);

    float total_z =
        static_cast<float>(model_ir.base_layers + model_ir.color_layers) * layer_height_mm;
    if (model_ir.voxel_grids.front().num_layers > 0) {
        const auto& first_grid = model_ir.voxel_grids.front();
        int total_voxel_layers = first_grid.num_layers;
        total_z                = static_cast<float>(total_voxel_layers) * layer_height_mm;
    }
    for (auto& v : mesh.vertices) { v.z += total_z; }

    spdlog::debug("BuildTransparentLayerFromModelIR: verts={}, tris={}, z=[{}, {}]",
                  mesh.vertices.size(), mesh.indices.size(), total_z, total_z + thickness_mm);
    return mesh;
}

MeshNamesAndSlots BuildMeshNamesAndSlots(size_t mesh_count, const std::vector<Channel>& palette,
                                         int base_channel_idx, int base_layers,
                                         bool has_transparent_layer) {
    MeshNamesAndSlots result;
    result.names.reserve(mesh_count);
    result.slots.reserve(mesh_count);

    const size_t num_channels = palette.size();
    const bool has_base       = base_layers > 0;

    for (size_t i = 0; i < mesh_count; ++i) {
        if (i < num_channels) {
            const auto& ch   = palette[i];
            std::string name = ch.color + (ch.material.empty() ? "" : " " + ch.material);
            result.names.push_back(std::move(name));
            result.slots.push_back(static_cast<int>(i) + 1);
        } else if (i == num_channels && has_base) {
            std::string name = "Base";
            if (base_channel_idx >= 0 && base_channel_idx < static_cast<int>(num_channels)) {
                name += " (" + palette[static_cast<size_t>(base_channel_idx)].color + ")";
            }
            result.names.push_back(std::move(name));
            result.slots.push_back(base_channel_idx >= 0 ? base_channel_idx + 1 : 1);
        } else {
            result.names.emplace_back("Transparent Layer");
            result.slots.push_back(static_cast<int>(num_channels) + (has_base ? 1 : 0) + 1);
        }
    }
    return result;
}

// ── Public export API ───────────────────────────────────────────────────────

void Export3mf(const std::string& path, const ModelIR& model_ir) {
    Export3mf(path, model_ir, BuildMeshConfig{});
}

void Export3mf(const std::string& path, const ModelIR& model_ir, const BuildMeshConfig& cfg) {
    if (path.empty()) { throw InputError("Export3mf path is empty"); }
    std::vector<Mesh> meshes = BuildMeshes(model_ir, cfg);
    auto objects             = BuildInputObjects(
        meshes, [&](std::size_t i) { return BuildObjectName(model_ir, i); },
        [&](std::size_t i) {
            return PaletteColor(i, model_ir.palette, model_ir.base_channel_idx,
                                            model_ir.base_layers);
        });
    EnsureNonEmptyGeometry(objects, meshes.size());
    auto doc = BuildStandardDocument(objects);
    WriteDocumentToFile(path, doc);
}

std::vector<uint8_t> Export3mfToBuffer(const ModelIR& model_ir, const BuildMeshConfig& cfg,
                                       FaceOrientation face_orientation) {
    std::vector<Mesh> meshes = BuildMeshes(model_ir, cfg);
    OrientMeshesInPlace(meshes, face_orientation);

    auto objects = BuildInputObjects(
        meshes, [&](std::size_t i) { return BuildObjectName(model_ir, i); },
        [&](std::size_t i) {
            return PaletteColor(i, model_ir.palette, model_ir.base_channel_idx,
                                model_ir.base_layers);
        });
    EnsureNonEmptyGeometry(objects, meshes.size());
    return WriteDocument(BuildStandardDocument(objects));
}

std::vector<uint8_t> Export3mfToBuffer(const ModelIR& model_ir, const BuildMeshConfig& cfg,
                                       const SlicerPreset& preset,
                                       FaceOrientation face_orientation) {
    std::vector<Mesh> meshes = BuildMeshes(model_ir, cfg);
    OrientMeshesInPlace(meshes, face_orientation);

    const std::size_t num_channels = model_ir.palette.size();
    const bool has_base            = model_ir.base_layers > 0;
    const int base_ch              = model_ir.base_channel_idx;

    auto objects = BuildInputObjects(
        meshes, [&](std::size_t i) { return BuildObjectName(model_ir, i); },
        [&](std::size_t i) {
            return PaletteColor(i, model_ir.palette, base_ch, model_ir.base_layers);
        },
        [&](std::size_t i) -> int {
            return PaletteSlot(i, model_ir.palette, base_ch, model_ir.base_layers);
        });
    EnsureNonEmptyGeometry(objects, meshes.size());

    if (!preset.preset_json_path.empty()) {
        return WriteDocument(BuildBambuDocument(objects, preset, model_ir.name));
    }
    return WriteDocument(BuildStandardDocument(objects));
}

std::vector<uint8_t> Export3mfFromMeshes(const std::vector<Mesh>& meshes,
                                         const std::vector<Channel>& palette, int base_channel_idx,
                                         int base_layers, FaceOrientation face_orientation) {
    if (meshes.empty()) { throw InputError("meshes vector is empty"); }
    std::vector<Mesh> mutable_meshes(meshes.begin(), meshes.end());
    OrientMeshesInPlace(mutable_meshes, face_orientation);

    auto objects = BuildInputObjects(
        mutable_meshes,
        [&](std::size_t i) {
            return BuildObjectNameFromPalette(i, palette, base_channel_idx, base_layers);
        },
        [&](std::size_t i) { return PaletteColor(i, palette, base_channel_idx, base_layers); });
    EnsureNonEmptyGeometry(objects, mutable_meshes.size());
    return WriteDocument(BuildStandardDocument(objects));
}

std::vector<uint8_t> Export3mfFromMeshes(const std::vector<Mesh>& meshes,
                                         const std::vector<Channel>& palette, int base_channel_idx,
                                         int base_layers, const SlicerPreset& preset,
                                         FaceOrientation face_orientation,
                                         const std::string& model_name) {
    if (meshes.empty()) { throw InputError("meshes vector is empty"); }
    std::vector<Mesh> mutable_meshes(meshes.begin(), meshes.end());
    OrientMeshesInPlace(mutable_meshes, face_orientation);

    auto objects = BuildInputObjects(
        mutable_meshes,
        [&](std::size_t i) {
            return BuildObjectNameFromPalette(i, palette, base_channel_idx, base_layers);
        },
        [&](std::size_t i) { return PaletteColor(i, palette, base_channel_idx, base_layers); },
        [&](std::size_t i) -> int {
            return PaletteSlot(i, palette, base_channel_idx, base_layers);
        });
    EnsureNonEmptyGeometry(objects, mutable_meshes.size());

    if (!preset.preset_json_path.empty()) {
        return WriteDocument(BuildBambuDocument(objects, preset, model_name));
    }
    return WriteDocument(BuildStandardDocument(objects));
}

std::vector<uint8_t> Export3mfFromMeshes(const std::vector<Mesh>& meshes,
                                         const std::vector<Channel>& palette,
                                         const std::vector<std::string>& names,
                                         FaceOrientation face_orientation) {
    if (meshes.empty()) { throw InputError("meshes vector is empty"); }
    if (names.size() != meshes.size()) { throw InputError("names size must match meshes size"); }
    std::vector<Mesh> mutable_meshes(meshes.begin(), meshes.end());
    OrientMeshesInPlace(mutable_meshes, face_orientation);

    auto objects = BuildInputObjects(
        mutable_meshes, [&](std::size_t i) { return names[i]; },
        [&](std::size_t i) -> std::string {
            if (i < palette.size()) return palette[i].hex_color;
            return {};
        });
    EnsureNonEmptyGeometry(objects, mutable_meshes.size());
    return WriteDocument(BuildStandardDocument(objects));
}

std::vector<uint8_t> Export3mfFromMeshes(const std::vector<Mesh>& meshes,
                                         const std::vector<Channel>& palette,
                                         const std::vector<std::string>& names,
                                         const std::vector<int>& slots, const SlicerPreset& preset,
                                         FaceOrientation face_orientation,
                                         const std::string& model_name) {
    if (meshes.empty()) { throw InputError("meshes vector is empty"); }
    if (names.size() != meshes.size() || slots.size() != meshes.size()) {
        throw InputError("names/slots size must match meshes size");
    }
    std::vector<Mesh> mutable_meshes(meshes.begin(), meshes.end());
    OrientMeshesInPlace(mutable_meshes, face_orientation);

    auto objects = BuildInputObjects(
        mutable_meshes, [&](std::size_t i) { return names[i]; },
        [&](std::size_t i) -> std::string {
            int slot = slots[i];
            if (slot >= 1 && static_cast<std::size_t>(slot - 1) < palette.size()) {
                return palette[static_cast<std::size_t>(slot - 1)].hex_color;
            }
            return {};
        },
        [&](std::size_t i) -> int { return slots[i]; });
    EnsureNonEmptyGeometry(objects, mutable_meshes.size());

    if (!preset.preset_json_path.empty()) {
        return WriteDocument(BuildBambuDocument(objects, preset, model_name));
    }
    return WriteDocument(BuildStandardDocument(objects));
}

// ── Direct-to-file variants ─────────────────────────────────────────────────

void Export3mfToFile(const std::string& path, const ModelIR& model_ir, const BuildMeshConfig& cfg,
                     FaceOrientation face_orientation) {
    if (path.empty()) { throw InputError("Export3mfToFile path is empty"); }
    std::vector<Mesh> meshes = BuildMeshes(model_ir, cfg);
    OrientMeshesInPlace(meshes, face_orientation);

    auto objects = BuildInputObjects(
        meshes, [&](std::size_t i) { return BuildObjectName(model_ir, i); },
        [&](std::size_t i) {
            return PaletteColor(i, model_ir.palette, model_ir.base_channel_idx,
                                model_ir.base_layers);
        });
    EnsureNonEmptyGeometry(objects, meshes.size());
    WriteDocumentToFile(path, BuildStandardDocument(objects));
}

void Export3mfToFile(const std::string& path, const ModelIR& model_ir, const BuildMeshConfig& cfg,
                     const SlicerPreset& preset, FaceOrientation face_orientation) {
    if (path.empty()) { throw InputError("Export3mfToFile path is empty"); }
    std::vector<Mesh> meshes = BuildMeshes(model_ir, cfg);
    OrientMeshesInPlace(meshes, face_orientation);

    const int base_ch = model_ir.base_channel_idx;

    auto objects = BuildInputObjects(
        meshes, [&](std::size_t i) { return BuildObjectName(model_ir, i); },
        [&](std::size_t i) {
            return PaletteColor(i, model_ir.palette, base_ch, model_ir.base_layers);
        },
        [&](std::size_t i) -> int {
            return PaletteSlot(i, model_ir.palette, base_ch, model_ir.base_layers);
        });
    EnsureNonEmptyGeometry(objects, meshes.size());

    if (!preset.preset_json_path.empty()) {
        WriteDocumentToFile(path, BuildBambuDocument(objects, preset, model_ir.name));
    } else {
        WriteDocumentToFile(path, BuildStandardDocument(objects));
    }
}

void Export3mfFromMeshesToFile(const std::string& path, const std::vector<Mesh>& meshes,
                               const std::vector<Channel>& palette, int base_channel_idx,
                               int base_layers, FaceOrientation face_orientation) {
    if (path.empty()) { throw InputError("Export3mfFromMeshesToFile path is empty"); }
    if (meshes.empty()) { throw InputError("meshes vector is empty"); }
    std::vector<Mesh> mutable_meshes(meshes.begin(), meshes.end());
    OrientMeshesInPlace(mutable_meshes, face_orientation);

    auto objects = BuildInputObjects(
        mutable_meshes,
        [&](std::size_t i) {
            return BuildObjectNameFromPalette(i, palette, base_channel_idx, base_layers);
        },
        [&](std::size_t i) { return PaletteColor(i, palette, base_channel_idx, base_layers); });
    EnsureNonEmptyGeometry(objects, mutable_meshes.size());
    WriteDocumentToFile(path, BuildStandardDocument(objects));
}

void Export3mfFromMeshesToFile(const std::string& path, const std::vector<Mesh>& meshes,
                               const std::vector<Channel>& palette, int base_channel_idx,
                               int base_layers, const SlicerPreset& preset,
                               FaceOrientation face_orientation, const std::string& model_name) {
    if (path.empty()) { throw InputError("Export3mfFromMeshesToFile path is empty"); }
    if (meshes.empty()) { throw InputError("meshes vector is empty"); }
    std::vector<Mesh> mutable_meshes(meshes.begin(), meshes.end());
    OrientMeshesInPlace(mutable_meshes, face_orientation);

    auto objects = BuildInputObjects(
        mutable_meshes,
        [&](std::size_t i) {
            return BuildObjectNameFromPalette(i, palette, base_channel_idx, base_layers);
        },
        [&](std::size_t i) { return PaletteColor(i, palette, base_channel_idx, base_layers); },
        [&](std::size_t i) -> int {
            return PaletteSlot(i, palette, base_channel_idx, base_layers);
        });
    EnsureNonEmptyGeometry(objects, mutable_meshes.size());

    if (!preset.preset_json_path.empty()) {
        WriteDocumentToFile(path, BuildBambuDocument(objects, preset, model_name));
    } else {
        WriteDocumentToFile(path, BuildStandardDocument(objects));
    }
}

// ── Atomic file writing utility (for pipeline use) ──────────────────────────

namespace detail {

void WriteBufferToFileAtomically(const std::filesystem::path& final_path,
                                 const std::vector<uint8_t>& bytes) {
    if (final_path.empty()) { throw InputError("output path is empty"); }

    const auto dir = final_path.parent_path();
    if (!dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            throw IOError("Failed to create output directory " + dir.string() + ": " +
                          ec.message());
        }
    }

    auto make_temp = [&]() -> std::filesystem::path {
        static constexpr char kHex[] = "0123456789abcdef";
        std::random_device rd;
        for (int attempt = 0; attempt < 16; ++attempt) {
            std::string suffix;
            suffix.reserve(16);
            for (int i = 0; i < 8; ++i) {
                auto v = static_cast<unsigned char>(rd());
                suffix.push_back(kHex[(v >> 4U) & 0x0F]);
                suffix.push_back(kHex[v & 0x0F]);
            }
            auto candidate = dir / (final_path.filename().string() + ".tmp-" + suffix);
            std::error_code ec2;
            if (!std::filesystem::exists(candidate, ec2) && !ec2) { return candidate; }
        }
        throw IOError("Failed to allocate temporary output path for " + final_path.string());
    };

    auto temp_path = make_temp();

    {
        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            throw IOError("Cannot open file for writing: " + temp_path.string());
        }
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        if (!out.good()) { throw IOError("Cannot write to " + temp_path.string()); }
    }

    std::error_code ec;
#if defined(_WIN32)
    if (std::filesystem::exists(final_path, ec) && !ec) {
        std::filesystem::remove(final_path, ec);
        if (ec) {
            std::filesystem::remove(temp_path, ec);
            throw IOError("Failed to replace " + final_path.string() + ": " + ec.message());
        }
    }
#endif
    std::filesystem::rename(temp_path, final_path, ec);
    if (ec) {
        std::filesystem::remove(temp_path, ec);
        throw IOError("Failed to move temp file for " + final_path.string() + ": " + ec.message());
    }
    spdlog::debug("WriteBufferToFileAtomically: {} bytes to {}", bytes.size(), final_path.string());
}

} // namespace detail

} // namespace ChromaPrint3D
