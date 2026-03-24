#include "chromaprint3d/error.h"
#include "chromaprint3d/export_3mf.h"
#include "chromaprint3d/mesh.h"
#include "chromaprint3d/voxel.h"
#include "three_mf_writer.h"

#include <spdlog/spdlog.h>

#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace ChromaPrint3D {
namespace {

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

} // namespace

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

namespace { // reopen anonymous namespace for internal helpers

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

detail::ThreeMfExportOptions DefaultWriterOptions() {
    detail::ThreeMfExportOptions options;
    options.unit = detail::ThreeMfUnit::Millimeter;
    options.metadata.push_back({"Application", "ChromaPrint3D"});
    return options;
}

std::vector<detail::ThreeMfInputObject>
BuildWriterObjects(const std::vector<Mesh>& meshes,
                   const std::function<std::string(std::size_t)>& name_fn,
                   const std::function<std::string(std::size_t)>& color_fn) {
    std::vector<detail::ThreeMfInputObject> objects;
    objects.reserve(meshes.size());
    std::size_t skipped = 0;
    for (std::size_t i = 0; i < meshes.size(); ++i) {
        if (meshes[i].vertices.empty() || meshes[i].indices.empty()) {
            spdlog::warn("BuildWriterObjects: mesh {} ('{}') skipped (vertices={}, indices={})", i,
                         name_fn(i), meshes[i].vertices.size(), meshes[i].indices.size());
            ++skipped;
            continue;
        }
        detail::ThreeMfInputObject object;
        object.name              = name_fn(i);
        object.display_color_hex = color_fn(i);
        object.mesh              = &meshes[i];
        object.transform         = detail::ThreeMfTransform::Identity();
        objects.push_back(std::move(object));
    }
    spdlog::debug("BuildWriterObjects: {} mesh(es) in, {} object(s) out, {} skipped", meshes.size(),
                  objects.size(), skipped);
    return objects;
}

struct WriterObjectsAndSlots {
    std::vector<detail::ThreeMfInputObject> objects;
    std::vector<int> slots;
};

WriterObjectsAndSlots
BuildWriterObjectsAndSlots(const std::vector<Mesh>& meshes,
                           const std::function<std::string(std::size_t)>& name_fn,
                           const std::function<std::string(std::size_t)>& color_fn,
                           const std::function<int(std::size_t)>& slot_fn) {
    WriterObjectsAndSlots result;
    result.objects.reserve(meshes.size());
    result.slots.reserve(meshes.size());
    std::size_t skipped = 0;
    for (std::size_t i = 0; i < meshes.size(); ++i) {
        if (meshes[i].vertices.empty() || meshes[i].indices.empty()) {
            spdlog::warn("BuildWriterObjectsAndSlots: mesh {} ('{}') skipped (vertices={}, "
                         "indices={})",
                         i, name_fn(i), meshes[i].vertices.size(), meshes[i].indices.size());
            ++skipped;
            continue;
        }
        detail::ThreeMfInputObject object;
        object.name              = name_fn(i);
        object.display_color_hex = color_fn(i);
        object.mesh              = &meshes[i];
        object.transform         = detail::ThreeMfTransform::Identity();
        result.objects.push_back(std::move(object));
        result.slots.push_back(slot_fn(i));
    }
    spdlog::debug("BuildWriterObjectsAndSlots: {} mesh(es) in, {} object(s) out, {} skipped",
                  meshes.size(), result.objects.size(), skipped);
    return result;
}

void EnsureNonEmptyGeometry(const std::vector<detail::ThreeMfInputObject>& objects,
                            std::size_t total_input_meshes = 0) {
    if (objects.empty()) {
        std::string msg = "No geometry to export";
        if (total_input_meshes > 0) {
            msg += " (all " + std::to_string(total_input_meshes) +
                   " input mesh(es) were empty or filtered)";
        }
        spdlog::error("EnsureNonEmptyGeometry: {}", msg);
        throw InputError(msg);
    }
}

detail::ThreeMfWriter BuildWriter(const SlicerPreset* preset, std::vector<int> slots,
                                  std::string model_name) {
    detail::ThreeMfWriter writer(DefaultWriterOptions());
    if (preset && !preset->preset_json_path.empty()) {
        writer.RegisterExtension(
            detail::MakeBambuMetadataExtension(*preset, std::move(slots), std::move(model_name)));
    } else if (preset) {
        spdlog::warn("3MF preset export requested but preset_json_path is empty; exporting without "
                     "private metadata");
    }
    return writer;
}

std::vector<uint8_t> WriteObjectsToBuffer(const std::vector<detail::ThreeMfInputObject>& objects,
                                          const SlicerPreset* preset = nullptr,
                                          std::vector<int> slots     = {},
                                          std::string model_name     = {}) {
    auto writer = BuildWriter(preset, std::move(slots), std::move(model_name));
    return writer.WriteToBuffer(objects);
}

void WriteObjectsToFile(const std::string& path,
                        const std::vector<detail::ThreeMfInputObject>& objects,
                        const SlicerPreset* preset = nullptr, std::vector<int> slots = {},
                        std::string model_name = {}) {
    auto writer = BuildWriter(preset, std::move(slots), std::move(model_name));
    writer.WriteToFile(path, objects);
}

} // namespace

void Export3mf(const std::string& path, const ModelIR& model_ir) {
    Export3mf(path, model_ir, BuildMeshConfig{});
}

void Export3mf(const std::string& path, const ModelIR& model_ir, const BuildMeshConfig& cfg) {
    if (path.empty()) { throw InputError("Export3mf path is empty"); }
    spdlog::debug("Export3mf: exporting to file {}, {} grid(s)", path, model_ir.voxel_grids.size());

    std::vector<Mesh> meshes = BuildMeshes(model_ir, cfg);
    auto objects             = BuildWriterObjects(
        meshes, [&](std::size_t i) { return BuildObjectName(model_ir, i); },
        [&](std::size_t i) -> std::string {
            if (i < model_ir.palette.size()) return model_ir.palette[i].hex_color;
            if (i >= model_ir.palette.size() && model_ir.base_layers > 0 &&
                model_ir.base_channel_idx >= 0 &&
                static_cast<std::size_t>(model_ir.base_channel_idx) < model_ir.palette.size()) {
                return model_ir.palette[static_cast<std::size_t>(model_ir.base_channel_idx)]
                    .hex_color;
            }
            return {};
        });
    EnsureNonEmptyGeometry(objects, meshes.size());

    detail::ThreeMfWriter writer(DefaultWriterOptions());
    writer.WriteToFile(path, objects);
    spdlog::info("Export3mf: written {} object(s) to {}", objects.size(), path);
}

std::vector<uint8_t> Export3mfToBuffer(const ModelIR& model_ir, const BuildMeshConfig& cfg,
                                       FaceOrientation face_orientation) {
    spdlog::debug("Export3mfToBuffer: exporting {} grid(s) to memory", model_ir.voxel_grids.size());
    std::vector<Mesh> meshes = BuildMeshes(model_ir, cfg);
    OrientMeshesInPlace(meshes, face_orientation);

    auto objects = BuildWriterObjects(
        meshes, [&](std::size_t i) { return BuildObjectName(model_ir, i); },
        [&](std::size_t i) -> std::string {
            if (i < model_ir.palette.size()) return model_ir.palette[i].hex_color;
            if (i >= model_ir.palette.size() && model_ir.base_layers > 0 &&
                model_ir.base_channel_idx >= 0 &&
                static_cast<std::size_t>(model_ir.base_channel_idx) < model_ir.palette.size()) {
                return model_ir.palette[static_cast<std::size_t>(model_ir.base_channel_idx)]
                    .hex_color;
            }
            return {};
        });
    EnsureNonEmptyGeometry(objects, meshes.size());

    detail::ThreeMfWriter writer(DefaultWriterOptions());
    std::vector<uint8_t> buffer = writer.WriteToBuffer(objects);
    spdlog::info("Export3mfToBuffer: written {} object(s), {} bytes", objects.size(),
                 buffer.size());
    return buffer;
}

std::vector<uint8_t> Export3mfFromMeshes(const std::vector<Mesh>& meshes,
                                         const std::vector<Channel>& palette, int base_channel_idx,
                                         int base_layers, FaceOrientation face_orientation) {
    if (meshes.empty()) { throw InputError("meshes vector is empty"); }
    spdlog::debug("Export3mfFromMeshes: exporting {} mesh(es) to memory", meshes.size());
    std::vector<Mesh> mutable_meshes(meshes.begin(), meshes.end());
    OrientMeshesInPlace(mutable_meshes, face_orientation);

    auto objects = BuildWriterObjects(
        mutable_meshes,
        [&](std::size_t i) {
            return BuildObjectNameFromPalette(i, palette, base_channel_idx, base_layers);
        },
        [&](std::size_t i) -> std::string {
            if (i < palette.size()) return palette[i].hex_color;
            if (i >= palette.size() && base_layers > 0 && base_channel_idx >= 0 &&
                static_cast<std::size_t>(base_channel_idx) < palette.size()) {
                return palette[static_cast<std::size_t>(base_channel_idx)].hex_color;
            }
            return {};
        });
    EnsureNonEmptyGeometry(objects, mutable_meshes.size());

    detail::ThreeMfWriter writer(DefaultWriterOptions());
    std::vector<uint8_t> buffer = writer.WriteToBuffer(objects);
    spdlog::info("Export3mfFromMeshes: written {} object(s), {} bytes", objects.size(),
                 buffer.size());
    return buffer;
}

std::vector<uint8_t> Export3mfFromMeshes(const std::vector<Mesh>& meshes,
                                         const std::vector<Channel>& palette,
                                         const std::vector<std::string>& names,
                                         FaceOrientation face_orientation) {
    if (meshes.empty()) { throw InputError("meshes vector is empty"); }
    if (names.size() != meshes.size()) { throw InputError("names size must match meshes size"); }
    spdlog::debug("Export3mfFromMeshes (explicit names): exporting {} mesh(es)", meshes.size());
    std::vector<Mesh> mutable_meshes(meshes.begin(), meshes.end());
    OrientMeshesInPlace(mutable_meshes, face_orientation);

    auto objects = BuildWriterObjects(
        mutable_meshes, [&](std::size_t i) { return names[i]; },
        [&](std::size_t i) -> std::string {
            if (i < palette.size()) return palette[i].hex_color;
            return {};
        });
    EnsureNonEmptyGeometry(objects, mutable_meshes.size());

    detail::ThreeMfWriter writer(DefaultWriterOptions());
    std::vector<uint8_t> buffer = writer.WriteToBuffer(objects);
    spdlog::info("Export3mfFromMeshes (explicit names): written {} object(s), {} bytes",
                 objects.size(), buffer.size());
    return buffer;
}

std::vector<uint8_t> Export3mfFromMeshes(const std::vector<Mesh>& meshes,
                                         const std::vector<Channel>& palette, int base_channel_idx,
                                         int base_layers, const SlicerPreset& preset,
                                         FaceOrientation face_orientation,
                                         const std::string& model_name) {
    if (meshes.empty()) { throw InputError("meshes vector is empty"); }
    spdlog::debug("Export3mfFromMeshes (with preset): exporting {} mesh(es)", meshes.size());
    std::vector<Mesh> mutable_meshes(meshes.begin(), meshes.end());
    OrientMeshesInPlace(mutable_meshes, face_orientation);

    auto result = BuildWriterObjectsAndSlots(
        mutable_meshes,
        [&](std::size_t i) {
            return BuildObjectNameFromPalette(i, palette, base_channel_idx, base_layers);
        },
        [&](std::size_t i) -> std::string {
            if (i < palette.size()) return palette[i].hex_color;
            if (i >= palette.size() && base_layers > 0 && base_channel_idx >= 0 &&
                static_cast<std::size_t>(base_channel_idx) < palette.size()) {
                return palette[static_cast<std::size_t>(base_channel_idx)].hex_color;
            }
            return {};
        },
        [&](std::size_t i) -> int {
            if (i < palette.size()) return static_cast<int>(i) + 1;
            if (base_layers > 0 && base_channel_idx >= 0) return base_channel_idx + 1;
            return static_cast<int>(i) + 1;
        });
    EnsureNonEmptyGeometry(result.objects, mutable_meshes.size());

    std::vector<uint8_t> buffer =
        WriteObjectsToBuffer(result.objects, &preset, std::move(result.slots), model_name);
    spdlog::info("Export3mfFromMeshes (with preset): written {} object(s), {} bytes",
                 result.objects.size(), buffer.size());
    return buffer;
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
    spdlog::debug("Export3mfFromMeshes (explicit slots): exporting {} mesh(es)", meshes.size());
    std::vector<Mesh> mutable_meshes(meshes.begin(), meshes.end());
    OrientMeshesInPlace(mutable_meshes, face_orientation);

    auto result = BuildWriterObjectsAndSlots(
        mutable_meshes, [&](std::size_t i) { return names[i]; },
        [&](std::size_t i) -> std::string {
            int slot = slots[i];
            if (slot >= 1 && static_cast<std::size_t>(slot - 1) < palette.size()) {
                return palette[static_cast<std::size_t>(slot - 1)].hex_color;
            }
            return {};
        },
        [&](std::size_t i) -> int { return slots[i]; });
    EnsureNonEmptyGeometry(result.objects, mutable_meshes.size());

    std::vector<uint8_t> buffer =
        WriteObjectsToBuffer(result.objects, &preset, std::move(result.slots), model_name);
    spdlog::info("Export3mfFromMeshes (explicit slots): written {} object(s), {} bytes",
                 result.objects.size(), buffer.size());
    return buffer;
}

std::vector<uint8_t> Export3mfToBuffer(const ModelIR& model_ir, const BuildMeshConfig& cfg,
                                       const SlicerPreset& preset,
                                       FaceOrientation face_orientation) {
    spdlog::debug("Export3mfToBuffer (with preset): exporting {} grid(s)",
                  model_ir.voxel_grids.size());
    std::vector<Mesh> meshes = BuildMeshes(model_ir, cfg);
    OrientMeshesInPlace(meshes, face_orientation);

    const std::size_t num_channels = model_ir.palette.size();
    const bool has_base            = model_ir.base_layers > 0;
    const int base_channel_idx     = model_ir.base_channel_idx;

    auto result = BuildWriterObjectsAndSlots(
        meshes, [&](std::size_t i) { return BuildObjectName(model_ir, i); },
        [&](std::size_t i) -> std::string {
            if (i < num_channels) return model_ir.palette[i].hex_color;
            if (i >= num_channels && has_base && base_channel_idx >= 0 &&
                static_cast<std::size_t>(base_channel_idx) < num_channels) {
                return model_ir.palette[static_cast<std::size_t>(base_channel_idx)].hex_color;
            }
            return {};
        },
        [&](std::size_t i) -> int {
            if (i < num_channels) return static_cast<int>(i) + 1;
            if (has_base && base_channel_idx >= 0) return base_channel_idx + 1;
            return static_cast<int>(i) + 1;
        });
    EnsureNonEmptyGeometry(result.objects, meshes.size());

    std::vector<uint8_t> buffer =
        WriteObjectsToBuffer(result.objects, &preset, std::move(result.slots), model_ir.name);
    spdlog::info("Export3mfToBuffer (with preset): written {} object(s), {} bytes",
                 result.objects.size(), buffer.size());
    return buffer;
}

// ── Direct-to-file variants ──────────────────────────────────────────────────

void Export3mfToFile(const std::string& path, const ModelIR& model_ir, const BuildMeshConfig& cfg,
                     FaceOrientation face_orientation) {
    if (path.empty()) { throw InputError("Export3mfToFile path is empty"); }
    spdlog::debug("Export3mfToFile: exporting {} grid(s) to {}", model_ir.voxel_grids.size(), path);
    std::vector<Mesh> meshes = BuildMeshes(model_ir, cfg);
    OrientMeshesInPlace(meshes, face_orientation);

    auto objects = BuildWriterObjects(
        meshes, [&](std::size_t i) { return BuildObjectName(model_ir, i); },
        [&](std::size_t i) -> std::string {
            if (i < model_ir.palette.size()) return model_ir.palette[i].hex_color;
            if (i >= model_ir.palette.size() && model_ir.base_layers > 0 &&
                model_ir.base_channel_idx >= 0 &&
                static_cast<std::size_t>(model_ir.base_channel_idx) < model_ir.palette.size()) {
                return model_ir.palette[static_cast<std::size_t>(model_ir.base_channel_idx)]
                    .hex_color;
            }
            return {};
        });
    EnsureNonEmptyGeometry(objects, meshes.size());
    WriteObjectsToFile(path, objects);
    spdlog::info("Export3mfToFile: written {} object(s) to {}", objects.size(), path);
}

void Export3mfToFile(const std::string& path, const ModelIR& model_ir, const BuildMeshConfig& cfg,
                     const SlicerPreset& preset, FaceOrientation face_orientation) {
    if (path.empty()) { throw InputError("Export3mfToFile path is empty"); }
    spdlog::debug("Export3mfToFile (with preset): exporting {} grid(s) to {}",
                  model_ir.voxel_grids.size(), path);
    std::vector<Mesh> meshes = BuildMeshes(model_ir, cfg);
    OrientMeshesInPlace(meshes, face_orientation);

    const std::size_t num_channels = model_ir.palette.size();
    const bool has_base            = model_ir.base_layers > 0;
    const int base_channel_idx     = model_ir.base_channel_idx;

    auto result = BuildWriterObjectsAndSlots(
        meshes, [&](std::size_t i) { return BuildObjectName(model_ir, i); },
        [&](std::size_t i) -> std::string {
            if (i < num_channels) return model_ir.palette[i].hex_color;
            if (i >= num_channels && has_base && base_channel_idx >= 0 &&
                static_cast<std::size_t>(base_channel_idx) < num_channels) {
                return model_ir.palette[static_cast<std::size_t>(base_channel_idx)].hex_color;
            }
            return {};
        },
        [&](std::size_t i) -> int {
            if (i < num_channels) return static_cast<int>(i) + 1;
            if (has_base && base_channel_idx >= 0) return base_channel_idx + 1;
            return static_cast<int>(i) + 1;
        });
    EnsureNonEmptyGeometry(result.objects, meshes.size());
    WriteObjectsToFile(path, result.objects, &preset, std::move(result.slots), model_ir.name);
    spdlog::info("Export3mfToFile (with preset): written {} object(s) to {}", result.objects.size(),
                 path);
}

void Export3mfFromMeshesToFile(const std::string& path, const std::vector<Mesh>& meshes,
                               const std::vector<Channel>& palette, int base_channel_idx,
                               int base_layers, FaceOrientation face_orientation) {
    if (path.empty()) { throw InputError("Export3mfFromMeshesToFile path is empty"); }
    if (meshes.empty()) { throw InputError("meshes vector is empty"); }
    spdlog::debug("Export3mfFromMeshesToFile: exporting {} mesh(es) to {}", meshes.size(), path);
    std::vector<Mesh> mutable_meshes(meshes.begin(), meshes.end());
    OrientMeshesInPlace(mutable_meshes, face_orientation);

    auto objects = BuildWriterObjects(
        mutable_meshes,
        [&](std::size_t i) {
            return BuildObjectNameFromPalette(i, palette, base_channel_idx, base_layers);
        },
        [&](std::size_t i) -> std::string {
            if (i < palette.size()) return palette[i].hex_color;
            if (i >= palette.size() && base_layers > 0 && base_channel_idx >= 0 &&
                static_cast<std::size_t>(base_channel_idx) < palette.size()) {
                return palette[static_cast<std::size_t>(base_channel_idx)].hex_color;
            }
            return {};
        });
    EnsureNonEmptyGeometry(objects, mutable_meshes.size());
    WriteObjectsToFile(path, objects);
    spdlog::info("Export3mfFromMeshesToFile: written {} object(s) to {}", objects.size(), path);
}

void Export3mfFromMeshesToFile(const std::string& path, const std::vector<Mesh>& meshes,
                               const std::vector<Channel>& palette, int base_channel_idx,
                               int base_layers, const SlicerPreset& preset,
                               FaceOrientation face_orientation, const std::string& model_name) {
    if (path.empty()) { throw InputError("Export3mfFromMeshesToFile path is empty"); }
    if (meshes.empty()) { throw InputError("meshes vector is empty"); }
    spdlog::debug("Export3mfFromMeshesToFile (with preset): exporting {} mesh(es) to {}",
                  meshes.size(), path);
    std::vector<Mesh> mutable_meshes(meshes.begin(), meshes.end());
    OrientMeshesInPlace(mutable_meshes, face_orientation);

    auto result = BuildWriterObjectsAndSlots(
        mutable_meshes,
        [&](std::size_t i) {
            return BuildObjectNameFromPalette(i, palette, base_channel_idx, base_layers);
        },
        [&](std::size_t i) -> std::string {
            if (i < palette.size()) return palette[i].hex_color;
            if (i >= palette.size() && base_layers > 0 && base_channel_idx >= 0 &&
                static_cast<std::size_t>(base_channel_idx) < palette.size()) {
                return palette[static_cast<std::size_t>(base_channel_idx)].hex_color;
            }
            return {};
        },
        [&](std::size_t i) -> int {
            if (i < palette.size()) return static_cast<int>(i) + 1;
            if (base_layers > 0 && base_channel_idx >= 0) return base_channel_idx + 1;
            return static_cast<int>(i) + 1;
        });
    EnsureNonEmptyGeometry(result.objects, mutable_meshes.size());
    WriteObjectsToFile(path, result.objects, &preset, std::move(result.slots), model_name);
    spdlog::info("Export3mfFromMeshesToFile (with preset): written {} object(s) to {}",
                 result.objects.size(), path);
}

} // namespace ChromaPrint3D
