#include "chromaprint3d/voxel.h"
#include "chromaprint3d/mesh.h"
#include "chromaprint3d/export_3mf.h"
#include "chromaprint3d/error.h"

#include "lib3mf_implicit.hpp"

#include <spdlog/spdlog.h>

#include <limits>
#include <string>
#include <vector>

namespace ChromaPrint3D {
namespace {

static sLib3MFPosition ToPosition(const Vec3f& v) {
    sLib3MFPosition pos{};
    pos.m_Coordinates[0] = v.x;
    pos.m_Coordinates[1] = v.y;
    pos.m_Coordinates[2] = v.z;
    return pos;
}

static Lib3MF_uint32 ToIndex(int idx, std::size_t vertex_count) {
    if (idx < 0) { throw InputError("Mesh index is negative"); }
    const std::size_t uidx = static_cast<std::size_t>(idx);
    if (uidx >= vertex_count) { throw InputError("Mesh index out of range"); }
    if (uidx > static_cast<std::size_t>(std::numeric_limits<Lib3MF_uint32>::max())) {
        throw InputError("Mesh index exceeds lib3mf limit");
    }
    return static_cast<Lib3MF_uint32>(uidx);
}

static sLib3MFTriangle ToTriangle(const Vec3i& tri, std::size_t vertex_count) {
    sLib3MFTriangle t{};
    t.m_Indices[0] = ToIndex(tri.x, vertex_count);
    t.m_Indices[1] = ToIndex(tri.y, vertex_count);
    t.m_Indices[2] = ToIndex(tri.z, vertex_count);
    return t;
}

static std::string BuildObjectNameFromPalette(std::size_t idx,
                                               const std::vector<Channel>& palette,
                                               int base_channel_idx, int base_layers) {
    if (idx == palette.size() && base_layers > 0) {
        std::string name = "Base";
        if (base_channel_idx >= 0 &&
            base_channel_idx < static_cast<int>(palette.size())) {
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

static std::string BuildObjectName(const ModelIR& model_ir, std::size_t idx) {
    return BuildObjectNameFromPalette(idx, model_ir.palette, model_ir.base_channel_idx,
                                      model_ir.base_layers);
}

static void AddMeshToModel(Lib3MF::PModel& model, Lib3MF::PWrapper& wrapper,
                           const Mesh& mesh, const std::string& name) {
    const std::size_t vertex_count = mesh.vertices.size();
    if (vertex_count >
        static_cast<std::size_t>(std::numeric_limits<Lib3MF_uint32>::max())) {
        throw InputError("Mesh vertex count exceeds lib3mf limit");
    }

    std::vector<sLib3MFPosition> vertices;
    vertices.reserve(vertex_count);
    for (const Vec3f& v : mesh.vertices) {
        if (!v.IsFinite()) { throw InputError("Mesh vertex is not finite"); }
        vertices.push_back(ToPosition(v));
    }

    std::vector<sLib3MFTriangle> triangles;
    triangles.reserve(mesh.indices.size());
    for (const Vec3i& tri : mesh.indices) {
        triangles.push_back(ToTriangle(tri, vertex_count));
    }

    Lib3MF::PMeshObject mesh_object = model->AddMeshObject();
    mesh_object->SetName(name);
    mesh_object->SetGeometry(vertices, triangles);
    model->AddBuildItem(mesh_object.get(), wrapper->GetIdentityTransform());
}

static void Export3mfInternal(const std::string& path, const ModelIR& model_ir,
                              const BuildMeshConfig& cfg) {
    if (path.empty()) { throw InputError("Export3mf path is empty"); }
    if (model_ir.voxel_grids.empty()) { throw InputError("ModelIR voxel_grids is empty"); }
    spdlog::info("Export3mf: exporting to file {}, {} grid(s)", path, model_ir.voxel_grids.size());

    const auto n = static_cast<int>(model_ir.voxel_grids.size());
    std::vector<Mesh> meshes(static_cast<std::size_t>(n));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        const VoxelGrid& grid = model_ir.voxel_grids[static_cast<std::size_t>(i)];
        if (grid.width <= 0 || grid.height <= 0 || grid.num_layers <= 0) { continue; }
        if (grid.ooc.empty()) { continue; }
        meshes[static_cast<std::size_t>(i)] = Mesh::Build(grid, cfg);
    }

    try {
        Lib3MF::PWrapper wrapper = Lib3MF::CWrapper::loadLibrary();
        Lib3MF::PModel model     = wrapper->CreateModel();

        std::size_t exported = 0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
            if (meshes[i].vertices.empty() || meshes[i].indices.empty()) { continue; }
            AddMeshToModel(model, wrapper, meshes[i], BuildObjectName(model_ir, i));
            ++exported;
        }

        if (exported == 0) { throw InputError("No geometry to export"); }

        Lib3MF::PWriter writer = model->QueryWriter("3mf");
        writer->WriteToFile(path);
        spdlog::info("Export3mf: written {} object(s) to {}", exported, path);
    } catch (const Lib3MF::ELib3MFException& e) { throw IOError(e.what()); }
}

} // namespace

void Export3mf(const std::string& path, const ModelIR& model_ir) {
    Export3mfInternal(path, model_ir, BuildMeshConfig{});
}

void Export3mf(const std::string& path, const ModelIR& model_ir, const BuildMeshConfig& cfg) {
    Export3mfInternal(path, model_ir, cfg);
}

std::vector<uint8_t> Export3mfToBuffer(const ModelIR& model_ir, const BuildMeshConfig& cfg) {
    if (model_ir.voxel_grids.empty()) { throw InputError("ModelIR voxel_grids is empty"); }
    spdlog::info("Export3mfToBuffer: exporting {} grid(s) to memory", model_ir.voxel_grids.size());

    const auto n = static_cast<int>(model_ir.voxel_grids.size());
    std::vector<Mesh> meshes(static_cast<std::size_t>(n));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        const VoxelGrid& grid = model_ir.voxel_grids[static_cast<std::size_t>(i)];
        if (grid.width <= 0 || grid.height <= 0 || grid.num_layers <= 0) { continue; }
        if (grid.ooc.empty()) { continue; }
        meshes[static_cast<std::size_t>(i)] = Mesh::Build(grid, cfg);
    }

    try {
        Lib3MF::PWrapper wrapper = Lib3MF::CWrapper::loadLibrary();
        Lib3MF::PModel model     = wrapper->CreateModel();

        std::size_t exported = 0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
            if (meshes[i].vertices.empty() || meshes[i].indices.empty()) { continue; }
            AddMeshToModel(model, wrapper, meshes[i], BuildObjectName(model_ir, i));
            ++exported;
        }

        if (exported == 0) { throw InputError("No geometry to export"); }

        Lib3MF::PWriter writer = model->QueryWriter("3mf");
        std::vector<uint8_t> buffer;
        writer->WriteToBuffer(buffer);
        spdlog::info("Export3mfToBuffer: written {} object(s), {} bytes", exported, buffer.size());
        return buffer;
    } catch (const Lib3MF::ELib3MFException& e) { throw IOError(e.what()); }
}

std::vector<uint8_t> Export3mfFromMeshes(const std::vector<Mesh>& meshes,
                                          const std::vector<Channel>& palette,
                                          int base_channel_idx, int base_layers) {
    if (meshes.empty()) { throw InputError("meshes vector is empty"); }
    spdlog::info("Export3mfFromMeshes: exporting {} mesh(es) to memory", meshes.size());

    try {
        Lib3MF::PWrapper wrapper = Lib3MF::CWrapper::loadLibrary();
        Lib3MF::PModel model     = wrapper->CreateModel();

        std::size_t exported = 0;
        for (std::size_t i = 0; i < meshes.size(); ++i) {
            if (meshes[i].vertices.empty() || meshes[i].indices.empty()) { continue; }
            std::string name = BuildObjectNameFromPalette(i, palette, base_channel_idx, base_layers);
            AddMeshToModel(model, wrapper, meshes[i], name);
            ++exported;
        }

        if (exported == 0) { throw InputError("No geometry to export"); }

        Lib3MF::PWriter writer = model->QueryWriter("3mf");
        std::vector<uint8_t> buffer;
        writer->WriteToBuffer(buffer);
        spdlog::info("Export3mfFromMeshes: written {} object(s), {} bytes", exported,
                     buffer.size());
        return buffer;
    } catch (const Lib3MF::ELib3MFException& e) { throw IOError(e.what()); }
}

} // namespace ChromaPrint3D
