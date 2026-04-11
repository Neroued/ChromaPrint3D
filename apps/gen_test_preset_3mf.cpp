#include "chromaprint3d/common.h"
#include "chromaprint3d/export_3mf.h"
#include "chromaprint3d/slicer_preset.h"
#include "chromaprint3d/voxel.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace ChromaPrint3D;

static Mesh MakeBoxMesh(int w, int h, int layers) {
    VoxelGrid grid;
    grid.width      = w;
    grid.height     = h;
    grid.num_layers = layers;
    grid.ooc.assign(static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(layers),
                    0);
    for (int z = 0; z < layers; ++z)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) grid.Set(x, y, z, true);

    BuildMeshConfig cfg;
    cfg.layer_height_mm = 0.08f;
    cfg.pixel_mm        = 0.42f;
    return Mesh::Build(grid, cfg);
}

int main(int argc, char* argv[]) {
    std::string preset_dir = "data/presets";
    std::string output     = "test_8color_preset.3mf";
    std::string nozzle_str = "n04";
    std::string face_str   = "faceup";
    if (argc > 1) output = argv[1];
    if (argc > 2) preset_dir = argv[2];
    if (argc > 3) nozzle_str = argv[3];
    if (argc > 4) face_str = argv[4];

    // Project channels: the logical color channels used in the image
    struct ChannelInfo {
        Channel channel;
        int ams_slot; // 1-based AMS slot this color maps to
    };

    // AMS layout (from user's actual AMS, read from reference 3MF):
    //   slot 1: White, slot 2: Black, slot 3: Yellow, slot 4: Red
    //   slot 5: Blue,  slot 6: Bambu Green, slot 7: Magenta, slot 8: Cyan
    std::vector<ChannelInfo> project_channels = {
        {{"Bambu Green", "PLA Basic", "#00AE42"}, 6}, {{"Black", "PLA Basic", "#000000"}, 2},
        {{"Blue", "PLA Basic", "#0A2989"}, 5},        {{"Cyan", "PLA Basic", "#0086D6"}, 8},
        {{"Magenta", "PLA Basic", "#EC008C"}, 7},     {{"Red", "PLA Basic", "#C12E1F"}, 4},
        {{"White", "PLA Basic", "#FFFFFF"}, 1},       {{"Yellow", "PLA Basic", "#F4EE2A"}, 3},
    };

    // Build AMS-ordered palette for filament slots (slot 1..8 in physical order)
    constexpr int kSlots = 8;
    std::vector<Channel> ams_palette(kSlots);
    for (const auto& ci : project_channels) {
        ams_palette[static_cast<size_t>(ci.ams_slot - 1)] = ci.channel;
    }

    PrintProfile profile;
    profile.layer_height_mm  = 0.08f;
    profile.color_layers     = 5;
    profile.line_width_mm    = 0.42f;
    profile.base_layers      = 7;
    profile.base_channel_idx = 0;
    profile.palette          = ams_palette;
    profile.nozzle_size      = ParseNozzleSize(nozzle_str);
    profile.face_orientation = ParseFaceOrientation(face_str);

    std::cout << "Nozzle: " << nozzle_str << ", Face: " << face_str << "\n";

    auto preset = SlicerPreset::FromProfile(preset_dir, profile);
    if (preset.preset_json_path.empty()) {
        std::cerr << "ERROR: preset file not found in " << preset_dir << "\n";
        return 1;
    }

    // Build meshes: one per project channel, plus base.
    // Each mesh's extruder must point to its AMS slot (not sequential index).
    std::vector<Mesh> all_meshes;
    std::vector<std::string> mesh_names;
    std::vector<int> mesh_slots;

    for (size_t i = 0; i < project_channels.size(); ++i) {
        all_meshes.push_back(MakeBoxMesh(10 + static_cast<int>(i) * 2, 10, 5));
        mesh_names.push_back(project_channels[i].channel.color + " - " +
                             project_channels[i].channel.material);
        mesh_slots.push_back(project_channels[i].ams_slot);
    }
    // Base mesh uses White = AMS slot 1
    all_meshes.push_back(MakeBoxMesh(80, 80, 7));
    mesh_names.push_back("Base - PLA Basic");
    mesh_slots.push_back(1);

    // Direct export with explicit slot assignments
    auto buffer = Export3mfFromMeshes(all_meshes, ams_palette, mesh_names, mesh_slots, preset);

    auto dir = std::filesystem::path(output).parent_path();
    if (!dir.empty()) std::filesystem::create_directories(dir);
    std::ofstream ofs(output, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(buffer.data()),
              static_cast<std::streamsize>(buffer.size()));

    std::cout << "Wrote " << buffer.size() << " bytes to " << output << "\n";
    std::cout << "AMS-ordered filament slots:\n";
    for (int i = 0; i < kSlots; ++i) {
        std::cout << "  slot " << (i + 1) << ": " << preset.filaments[static_cast<size_t>(i)].colour
                  << " (" << ams_palette[static_cast<size_t>(i)].color << ")\n";
    }
    std::cout << "Part -> AMS slot mapping:\n";
    for (size_t i = 0; i < mesh_names.size(); ++i) {
        std::cout << "  " << mesh_names[i] << " -> slot " << mesh_slots[i] << "\n";
    }
    return 0;
}
