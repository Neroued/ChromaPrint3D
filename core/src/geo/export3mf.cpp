#include "geo.h"

#include "lib3mf_implicit.hpp"

#include <limits>
#include <stdexcept>
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
    if (idx < 0) { throw std::runtime_error("Mesh index is negative"); }
    const std::size_t uidx = static_cast<std::size_t>(idx);
    if (uidx >= vertex_count) { throw std::runtime_error("Mesh index out of range"); }
    if (uidx > static_cast<std::size_t>(std::numeric_limits<Lib3MF_uint32>::max())) {
        throw std::runtime_error("Mesh index exceeds lib3mf limit");
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

static std::string BuildObjectName(const ModelIR& model_ir, std::size_t idx) {
    if (idx < model_ir.palette.size()) {
        const Channel& ch = model_ir.palette[idx];
        std::string name  = ch.color.empty() ? ("Channel " + std::to_string(idx)) : ch.color;
        if (!ch.material.empty() && ch.material != "Default Material") {
            name += " - " + ch.material;
        }
        return name;
    }
    return "Channel " + std::to_string(idx);
}

static void Export3mfInternal(const std::string& path, const ModelIR& model_ir,
                              const BuildMeshConfig& cfg) {
    if (path.empty()) { throw std::runtime_error("Export3mf path is empty"); }
    if (model_ir.voxel_grids.empty()) { throw std::runtime_error("ModelIR voxel_grids is empty"); }

    try {
        Lib3MF::PWrapper wrapper = Lib3MF::CWrapper::loadLibrary();
        Lib3MF::PModel model     = wrapper->CreateModel();

        std::size_t exported = 0;
        for (std::size_t i = 0; i < model_ir.voxel_grids.size(); ++i) {
            const VoxelGrid& grid = model_ir.voxel_grids[i];
            if (grid.width <= 0 || grid.height <= 0 || grid.num_layers <= 0) { continue; }
            if (grid.ooc.empty()) { continue; }

            Mesh mesh = Mesh::Build(grid, cfg);
            if (mesh.vertices.empty() || mesh.indices.empty()) { continue; }

            const std::size_t vertex_count = mesh.vertices.size();
            if (vertex_count >
                static_cast<std::size_t>(std::numeric_limits<Lib3MF_uint32>::max())) {
                throw std::runtime_error("Mesh vertex count exceeds lib3mf limit");
            }

            std::vector<sLib3MFPosition> vertices;
            vertices.reserve(vertex_count);
            for (const Vec3f& v : mesh.vertices) {
                if (!v.IsFinite()) { throw std::runtime_error("Mesh vertex is not finite"); }
                vertices.push_back(ToPosition(v));
            }

            std::vector<sLib3MFTriangle> triangles;
            triangles.reserve(mesh.indices.size());
            for (const Vec3i& tri : mesh.indices) {
                triangles.push_back(ToTriangle(tri, vertex_count));
            }

            Lib3MF::PMeshObject mesh_object = model->AddMeshObject();
            mesh_object->SetName(BuildObjectName(model_ir, i));
            mesh_object->SetGeometry(vertices, triangles);
            model->AddBuildItem(mesh_object.get(), wrapper->GetIdentityTransform());
            ++exported;
        }

        if (exported == 0) { throw std::runtime_error("No geometry to export"); }

        Lib3MF::PWriter writer = model->QueryWriter("3mf");
        writer->WriteToFile(path);
    } catch (const Lib3MF::ELib3MFException& e) { throw std::runtime_error(e.what()); }
}

} // namespace

void Export3mf(const std::string& path, const ModelIR& model_ir) {
    Export3mfInternal(path, model_ir, BuildMeshConfig{});
}

void Export3mf(const std::string& path, const ModelIR& model_ir, const BuildMeshConfig& cfg) {
    Export3mfInternal(path, model_ir, cfg);
}

} // namespace ChromaPrint3D
