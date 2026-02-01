#include "geo.h"

#include <stdexcept>
#include <unordered_map>

namespace ChromaPrint3D {
namespace {
inline size_t GridIndex(int w, int h, int l, int width, int height, int layers) {
    return (static_cast<size_t>(h) * static_cast<size_t>(width) + static_cast<size_t>(w)) *
               static_cast<size_t>(layers) +
           static_cast<size_t>(l);
}
} // namespace

bool VoxelGrid::Get(int w, int h, int l) const {
    if (w < 0 || h < 0 || l < 0 || w >= width || h >= height || l >= num_layers) { return false; }
    const size_t idx = GridIndex(w, h, l, width, height, num_layers);
    if (idx >= ooc.size()) { return false; }
    return ooc[idx] != 0;
}

bool VoxelGrid::Set(int w, int h, int l, bool v) {
    if (w < 0 || h < 0 || l < 0 || w >= width || h >= height || l >= num_layers) { return false; }
    const size_t idx = GridIndex(w, h, l, width, height, num_layers);
    if (idx >= ooc.size()) { return false; }
    ooc[idx] = v ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    return true;
}

ModelIR ModelIR::Build(const RecipeMap& recipe_map, const ColorDB& db,
                       const BuildModelIRConfig& cfg) {
    if (recipe_map.width <= 0 || recipe_map.height <= 0) {
        throw std::runtime_error("RecipeMap size is invalid");
    }
    if (recipe_map.color_layers < 0 || recipe_map.num_channels < 0) {
        throw std::runtime_error("RecipeMap layers or channels are invalid");
    }

    const int width        = recipe_map.width;
    const int height       = recipe_map.height;
    const int color_layers = recipe_map.color_layers;

    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (color_layers > 0) {
        const size_t expected = pixel_count * static_cast<size_t>(color_layers);
        if (recipe_map.recipes.size() < expected) {
            throw std::runtime_error("RecipeMap recipes size mismatch");
        }
    }
    if (!recipe_map.mask.empty() && recipe_map.mask.size() < pixel_count) {
        throw std::runtime_error("RecipeMap mask size mismatch");
    }

    int num_channels = recipe_map.num_channels;
    if (num_channels <= 0) { num_channels = static_cast<int>(db.NumChannels()); }
    if (num_channels <= 0) { throw std::runtime_error("num_channels is invalid"); }
    if (db.NumChannels() > 0 && static_cast<int>(db.NumChannels()) != num_channels) {
        throw std::runtime_error("RecipeMap num_channels does not match ColorDB");
    }

    const int base_layers = (cfg.base_layers != 0) ? cfg.base_layers : db.base_layers;
    if (base_layers < 0) { throw std::runtime_error("base_layers is invalid"); }

    const int base_channel_idx = db.base_channel_idx;
    if (base_layers > 0 && (base_channel_idx < 0 || base_channel_idx >= num_channels)) {
        throw std::runtime_error("base_channel_idx is out of range");
    }

    const bool double_sided = cfg.double_sided;
    const int base_start    = double_sided ? color_layers : 0;
    const int total_layers  = base_start + base_layers + color_layers;
    if (total_layers < 0) { throw std::runtime_error("total_layers is invalid"); }

    ModelIR result;
    result.name             = recipe_map.name;
    result.width            = width;
    result.height           = height;
    result.color_layers     = color_layers;
    result.base_layers      = base_layers;
    result.base_channel_idx = base_channel_idx;
    result.palette          = db.palette;
    if (result.palette.empty()) {
        result.palette.resize(static_cast<size_t>(num_channels));
    } else if (static_cast<int>(result.palette.size()) != num_channels) {
        throw std::runtime_error("palette size does not match num_channels");
    }
    result.voxel_grids.resize(static_cast<size_t>(num_channels));

    for (int ch = 0; ch < num_channels; ++ch) {
        VoxelGrid& grid  = result.voxel_grids[static_cast<size_t>(ch)];
        grid.width       = width;
        grid.height      = height;
        grid.num_layers  = total_layers;
        grid.channel_idx = ch;
        grid.ooc.assign(pixel_count * static_cast<size_t>(total_layers), 0);
    }

    const bool has_mask = !recipe_map.mask.empty();

    for (int r = 0; r < height; ++r) {
        const int vh = cfg.flip_y ? (height - 1 - r) : r;
        for (int c = 0; c < width; ++c) {
            const size_t idx =
                static_cast<size_t>(r) * static_cast<size_t>(width) + static_cast<size_t>(c);
            if (has_mask && recipe_map.mask[idx] == 0) { continue; }

            if (base_layers > 0) {
                VoxelGrid& base_grid = result.voxel_grids[static_cast<size_t>(base_channel_idx)];
                for (int l = 0; l < base_layers; ++l) {
                    const int base_layer = base_start + l;
                    const size_t offset = GridIndex(c, vh, base_layer, width, height, total_layers);
                    if (offset < base_grid.ooc.size()) { base_grid.ooc[offset] = 1; }
                }
            }

            if (color_layers == 0) { continue; }
            const uint8_t* recipe = recipe_map.RecipeAt(r, c);
            if (!recipe) { continue; }

            for (int layer = 0; layer < color_layers; ++layer) {
                const int mapped_layer = (recipe_map.layer_order == LayerOrder::Top2Bottom)
                                             ? (color_layers - 1 - layer)
                                             : layer;
                const int stored_layer = base_start + base_layers + mapped_layer;
                // VoxelGrid 层序为自底向上，layer 0 是最底层（含 base）。
                const int channel_idx = static_cast<int>(recipe[layer]);
                if (channel_idx < 0 || channel_idx >= num_channels) { continue; }

                VoxelGrid& grid     = result.voxel_grids[static_cast<size_t>(channel_idx)];
                const size_t offset = GridIndex(c, vh, stored_layer, width, height, total_layers);
                if (offset < grid.ooc.size()) { grid.ooc[offset] = 1; }
            }

            if (!double_sided) { continue; }
            for (int layer = 0; layer < color_layers; ++layer) {
                const int mapped_layer = (recipe_map.layer_order == LayerOrder::Top2Bottom)
                                             ? (color_layers - 1 - layer)
                                             : layer;
                const int stored_layer = (base_start - 1) - mapped_layer;
                const int channel_idx  = static_cast<int>(recipe[layer]);
                if (channel_idx < 0 || channel_idx >= num_channels) { continue; }

                VoxelGrid& grid     = result.voxel_grids[static_cast<size_t>(channel_idx)];
                const size_t offset = GridIndex(c, vh, stored_layer, width, height, total_layers);
                if (offset < grid.ooc.size()) { grid.ooc[offset] = 1; }
            }
        }
    }

    return result;
}

Mesh Mesh::Build(const VoxelGrid& voxel_grid, const BuildMeshConfig& cfg) {
    Mesh mesh;
    const int width  = voxel_grid.width;
    const int height = voxel_grid.height;
    const int layers = voxel_grid.num_layers;
    if (width <= 0 || height <= 0 || layers <= 0) { return mesh; }
    if (cfg.pixel_mm <= 0.0f || cfg.layer_height_mm <= 0.0f) {
        throw std::runtime_error("BuildMeshConfig values must be positive");
    }

    const size_t expected =
        static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(layers);
    if (voxel_grid.ooc.size() < expected) {
        throw std::runtime_error("VoxelGrid ooc size mismatch");
    }

    struct Vec3iHash {
        size_t operator()(const Vec3i& v) const {
            size_t h = std::hash<int>{}(v.x);
            h ^= std::hash<int>{}(v.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(v.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct Vec3iEq {
        bool operator()(const Vec3i& a, const Vec3i& b) const {
            return a.x == b.x && a.y == b.y && a.z == b.z;
        }
    };

    std::unordered_map<Vec3i, int, Vec3iHash, Vec3iEq> vertex_map;

    const float px = cfg.pixel_mm;
    const float pz = cfg.layer_height_mm;

    auto add_vertex = [&](const Vec3i& v) {
        auto it = vertex_map.find(v);
        if (it != vertex_map.end()) { return it->second; }
        const int idx = static_cast<int>(mesh.vertices.size());
        mesh.vertices.emplace_back(static_cast<float>(v.x) * px, static_cast<float>(v.y) * px,
                                   static_cast<float>(v.z) * pz);
        vertex_map.emplace(v, idx);
        return idx;
    };

    auto add_quad = [&](const Vec3i& v0, const Vec3i& v1, const Vec3i& v2, const Vec3i& v3) {
        const int i0 = add_vertex(v0);
        const int i1 = add_vertex(v1);
        const int i2 = add_vertex(v2);
        const int i3 = add_vertex(v3);
        mesh.indices.emplace_back(i0, i1, i2);
        mesh.indices.emplace_back(i0, i2, i3);
    };

    auto is_filled = [&](int x, int y, int z) -> bool {
        if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= layers) { return false; }
        const size_t idx =
            (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) *
                static_cast<size_t>(layers) +
            static_cast<size_t>(z);
        return voxel_grid.ooc[idx] != 0;
    };

    const int dims[3] = {width, height, layers};

    for (int d = 0; d < 3; ++d) {
        const int u = (d + 1) % 3;
        const int v = (d + 2) % 3;

        std::vector<int> mask(static_cast<size_t>(dims[u]) * static_cast<size_t>(dims[v]), 0);

        int x[3] = {0, 0, 0};
        for (int slice = 0; slice <= dims[d]; ++slice) {
            int n = 0;
            for (x[v] = 0; x[v] < dims[v]; ++x[v]) {
                for (x[u] = 0; x[u] < dims[u]; ++x[u]) {
                    bool a = false;
                    bool b = false;
                    if (slice > 0) {
                        x[d] = slice - 1;
                        a    = is_filled(x[0], x[1], x[2]);
                    }
                    if (slice < dims[d]) {
                        x[d] = slice;
                        b    = is_filled(x[0], x[1], x[2]);
                    }
                    mask[n++] = (a != b) ? (a ? 1 : -1) : 0;
                }
            }

            n = 0;
            for (int j = 0; j < dims[v]; ++j) {
                for (int i = 0; i < dims[u];) {
                    const int c = mask[n];
                    if (c == 0) {
                        ++i;
                        ++n;
                        continue;
                    }

                    int w = 1;
                    while (i + w < dims[u] && mask[n + w] == c) { ++w; }

                    int h = 1;
                    for (; j + h < dims[v]; ++h) {
                        bool ok = true;
                        for (int k = 0; k < w; ++k) {
                            if (mask[n + k + h * dims[u]] != c) {
                                ok = false;
                                break;
                            }
                        }
                        if (!ok) { break; }
                    }

                    int x0[3] = {0, 0, 0};
                    x0[u]     = i;
                    x0[v]     = j;
                    x0[d]     = slice;

                    int x1[3] = {x0[0], x0[1], x0[2]};
                    int x2[3] = {x0[0], x0[1], x0[2]};
                    int x3[3] = {x0[0], x0[1], x0[2]};
                    x1[u] += w;
                    x2[v] += h;
                    x3[u] += w;
                    x3[v] += h;

                    const Vec3i v0{x0[0], x0[1], x0[2]};
                    const Vec3i v1{x1[0], x1[1], x1[2]};
                    const Vec3i v2{x2[0], x2[1], x2[2]};
                    const Vec3i v3{x3[0], x3[1], x3[2]};
                    if (c > 0) {
                        add_quad(v0, v1, v3, v2);
                    } else {
                        add_quad(v0, v2, v3, v1);
                    }

                    for (int dy = 0; dy < h; ++dy) {
                        for (int dx = 0; dx < w; ++dx) { mask[n + dx + dy * dims[u]] = 0; }
                    }

                    i += w;
                    n += w;
                }
            }
        }
    }

    return mesh;
}

} // namespace ChromaPrint3D
