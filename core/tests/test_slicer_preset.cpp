#include <gtest/gtest.h>

#include "chromaprint3d/export_3mf.h"
#include "chromaprint3d/slicer_preset.h"
#include "chromaprint3d/voxel.h"
#include "geo/bambu_metadata.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(CHROMAPRINT3D_TEST_HAS_ZLIB)
#    include <zlib.h>
#endif

using namespace ChromaPrint3D;

namespace {

constexpr uint32_t kZipLocalFileHeaderSignature = 0x04034B50u;

uint16_t ReadU16(const std::vector<uint8_t>& bytes, std::size_t offset) {
    if (offset + 1 >= bytes.size()) { throw std::runtime_error("ReadU16 out of range"); }
    return static_cast<uint16_t>(bytes[offset] | (static_cast<uint16_t>(bytes[offset + 1]) << 8));
}

uint32_t ReadU32(const std::vector<uint8_t>& bytes, std::size_t offset) {
    if (offset + 3 >= bytes.size()) { throw std::runtime_error("ReadU32 out of range"); }
    return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

struct ZipEntry {
    std::string name;
    uint16_t compression_method = 0;
    std::vector<uint8_t> raw_data;
    std::vector<uint8_t> data;
};

std::vector<uint8_t> InflateRawDeflate(const std::vector<uint8_t>& compressed,
                                       std::size_t expected_size) {
#if defined(CHROMAPRINT3D_TEST_HAS_ZLIB)
    std::vector<uint8_t> output(expected_size);
    z_stream stream{};
    stream.next_in   = const_cast<Bytef*>(compressed.data());
    stream.avail_in  = static_cast<uInt>(compressed.size());
    stream.next_out  = output.data();
    stream.avail_out = static_cast<uInt>(output.size());

    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) {
        throw std::runtime_error("inflateInit2 failed");
    }
    const int rc = inflate(&stream, Z_FINISH);
    inflateEnd(&stream);
    if (rc != Z_STREAM_END) { throw std::runtime_error("inflate failed"); }
    output.resize(static_cast<std::size_t>(stream.total_out));
    return output;
#else
    (void)compressed;
    (void)expected_size;
    throw std::runtime_error("Deflate entry encountered but zlib is unavailable in test build");
#endif
}

std::vector<ZipEntry> ParseZipEntries(const std::vector<uint8_t>& bytes) {
    std::vector<ZipEntry> entries;
    std::size_t pos = 0;
    while (pos + 4 <= bytes.size()) {
        uint32_t signature = ReadU32(bytes, pos);
        if (signature != kZipLocalFileHeaderSignature) { break; }
        if (pos + 30 > bytes.size()) { throw std::runtime_error("Corrupted ZIP local header"); }

        uint16_t general_purpose   = ReadU16(bytes, pos + 6);
        uint16_t compression       = ReadU16(bytes, pos + 8);
        uint32_t compressed_size   = ReadU32(bytes, pos + 18);
        uint32_t uncompressed_size = ReadU32(bytes, pos + 22);
        uint16_t name_len          = ReadU16(bytes, pos + 26);
        uint16_t extra_len         = ReadU16(bytes, pos + 28);

        std::size_t name_off = pos + 30;
        std::size_t data_off =
            name_off + static_cast<std::size_t>(name_len) + static_cast<std::size_t>(extra_len);

        bool has_data_descriptor = (general_purpose & 0x0008) != 0;

        if (has_data_descriptor && compressed_size == 0) {
            std::size_t scan = data_off;
            while (scan + 4 <= bytes.size()) {
                uint32_t sig = ReadU32(bytes, scan);
                if (sig == kZipLocalFileHeaderSignature || sig == 0x02014B50u) { break; }
                if (sig == 0x08074B50u) {
                    compressed_size   = ReadU32(bytes, scan + 8);
                    uncompressed_size = ReadU32(bytes, scan + 12);
                    break;
                }
                ++scan;
            }
            std::size_t data_end = (compressed_size > 0) ? data_off + compressed_size : scan;

            ZipEntry entry;
            entry.compression_method = compression;
            entry.name.assign(reinterpret_cast<const char*>(bytes.data() + name_off), name_len);
            entry.raw_data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(data_off),
                                  bytes.begin() + static_cast<std::ptrdiff_t>(data_end));
            if (compression == 0) {
                entry.data = entry.raw_data;
            } else {
                entry.data =
                    InflateRawDeflate(entry.raw_data, static_cast<std::size_t>(uncompressed_size));
            }
            entries.push_back(std::move(entry));
            // Advance past data descriptor
            if (scan + 4 <= bytes.size() && ReadU32(bytes, scan) == 0x08074B50u) {
                pos = scan + 16;
            } else {
                pos = scan;
            }
        } else {
            std::size_t data_end = data_off + static_cast<std::size_t>(compressed_size);
            if (data_end > bytes.size()) {
                throw std::runtime_error("Corrupted ZIP entry payload");
            }
            if (compression != 0 && compression != 8) {
                throw std::runtime_error("Unsupported ZIP compression method");
            }

            ZipEntry entry;
            entry.compression_method = compression;
            entry.name.assign(reinterpret_cast<const char*>(bytes.data() + name_off), name_len);
            entry.raw_data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(data_off),
                                  bytes.begin() + static_cast<std::ptrdiff_t>(data_end));
            if (compression == 0) {
                entry.data = entry.raw_data;
            } else {
                entry.data =
                    InflateRawDeflate(entry.raw_data, static_cast<std::size_t>(uncompressed_size));
            }
            entries.push_back(std::move(entry));
            pos = data_end;

            if (has_data_descriptor) {
                if (pos + 4 <= bytes.size() && ReadU32(bytes, pos) == 0x08074B50u) {
                    pos += 16;
                } else if (pos + 12 <= bytes.size()) {
                    pos += 12;
                }
            }
        }
    }
    return entries;
}

const ZipEntry* FindEntry(const std::vector<ZipEntry>& entries, const std::string& name) {
    for (const auto& entry : entries) {
        if (entry.name == name) { return &entry; }
    }
    return nullptr;
}

std::string EntryAsString(const ZipEntry* entry) {
    if (!entry) { return {}; }
    return std::string(entry->data.begin(), entry->data.end());
}

std::pair<double, double> ParseFirstRangeBounds(const std::string& xml) {
    const std::size_t range_pos = xml.find("<range ");
    if (range_pos == std::string::npos) { throw std::runtime_error("range tag not found"); }

    auto parse_attr = [&](const char* key) -> double {
        const std::string token = std::string(key) + "=\"";
        const std::size_t pos   = xml.find(token, range_pos);
        if (pos == std::string::npos) {
            throw std::runtime_error(std::string("attribute missing: ") + key);
        }
        const std::size_t begin = pos + token.size();
        const std::size_t end   = xml.find('"', begin);
        if (end == std::string::npos) {
            throw std::runtime_error(std::string("attribute malformed: ") + key);
        }
        return std::stod(xml.substr(begin, end - begin));
    };

    return {parse_attr("min_z"), parse_attr("max_z")};
}

std::vector<Vec3f> ParseVerticesFromModelXml(const std::string& model_xml) {
    std::vector<Vec3f> vertices;
    std::size_t pos = 0;
    while (true) {
        pos = model_xml.find("<vertex ", pos);
        if (pos == std::string::npos) { break; }

        const std::size_t x_pos = model_xml.find("x=\"", pos);
        const std::size_t y_pos = model_xml.find("y=\"", pos);
        const std::size_t z_pos = model_xml.find("z=\"", pos);
        if (x_pos == std::string::npos || y_pos == std::string::npos ||
            z_pos == std::string::npos) {
            throw std::runtime_error("Malformed vertex tag in model XML");
        }

        const std::size_t x_end = model_xml.find('"', x_pos + 3);
        const std::size_t y_end = model_xml.find('"', y_pos + 3);
        const std::size_t z_end = model_xml.find('"', z_pos + 3);
        if (x_end == std::string::npos || y_end == std::string::npos ||
            z_end == std::string::npos) {
            throw std::runtime_error("Malformed vertex attribute in model XML");
        }

        const float x = std::stof(model_xml.substr(x_pos + 3, x_end - (x_pos + 3)));
        const float y = std::stof(model_xml.substr(y_pos + 3, y_end - (y_pos + 3)));
        const float z = std::stof(model_xml.substr(z_pos + 3, z_end - (z_pos + 3)));
        vertices.emplace_back(x, y, z);
        pos = z_end + 1;
    }
    return vertices;
}

std::string GetPresetDir() {
    const char* env = std::getenv("CHROMAPRINT3D_PRESET_DIR");
    if (env) return env;
    const char* pwd = std::getenv("PWD");
    return std::string(pwd ? pwd : ".") + "/data/presets";
}

bool PresetFileExists() {
    return std::filesystem::exists(GetPresetDir() + "/bambu_p2s_0.08mm_n04_faceup.json");
}

SlicerPreset MakeTestPreset() {
    SlicerPreset preset;
    preset.preset_json_path = GetPresetDir() + "/bambu_p2s_0.08mm_n04_faceup.json";
    preset.filaments        = {
        {.type = "PLA", .colour = "#C12E1F"},
        {.type = "PLA", .colour = "#00AE42"},
        {.type = "PLA", .colour = "#0A2989"},
    };
    return preset;
}

Mesh MakeBoxMesh() {
    VoxelGrid grid;
    grid.width      = 2;
    grid.height     = 2;
    grid.num_layers = 1;
    grid.ooc.assign(4, 0);
    grid.Set(0, 0, 0, true);
    grid.Set(1, 0, 0, true);
    grid.Set(0, 1, 0, true);
    grid.Set(1, 1, 0, true);
    return Mesh::Build(grid);
}

Mesh MakeDegenerateMesh() {
    Mesh mesh;
    mesh.vertices = {
        Vec3f{0.0f, 0.0f, 0.0f},
        Vec3f{1.0f, 0.0f, 0.0f},
        Vec3f{2.0f, 0.0f, 0.0f},
    };
    mesh.indices = {
        Vec3i{0, 1, 2},
    };
    return mesh;
}

} // namespace

TEST(SlicerPreset, FindPresetFileFindsExisting) {
    std::string dir  = GetPresetDir();
    std::string path = FindPresetFile(dir, "Bambu Lab P2S", 0.08f);
    if (PresetFileExists()) {
        ASSERT_FALSE(path.empty());
        EXPECT_NE(path.find("bambu_p2s_0.08mm_n04_faceup.json"), std::string::npos);
    }
}

TEST(SlicerPreset, FindPresetFileReturnsEmptyForMissing) {
    std::string dir  = GetPresetDir();
    std::string path = FindPresetFile(dir, "Bambu Lab P2S", 0.99f);
    EXPECT_TRUE(path.empty());
}

TEST(SlicerPreset, DoubleSidedForcesFaceDownPresetSelection) {
    PrintProfile profile;
    profile.nozzle_size      = NozzleSize::N04;
    profile.face_orientation = FaceOrientation::FaceUp;
    profile.layer_height_mm  = 0.08f;
    profile.palette.push_back(Channel{"Red", "PLA"});

    const SlicerPreset single_sided =
        SlicerPreset::FromProfile(GetPresetDir(), profile, nullptr, false);
    const SlicerPreset double_sided =
        SlicerPreset::FromProfile(GetPresetDir(), profile, nullptr, true);

    EXPECT_EQ(single_sided.face, FaceOrientation::FaceUp);
    EXPECT_EQ(double_sided.face, FaceOrientation::FaceDown);
}

TEST(SlicerPreset, ExportWithPresetContainsBambuMetadata) {
    if (!PresetFileExists()) { GTEST_SKIP() << "Preset file not found"; }
    SlicerPreset preset = MakeTestPreset();

    std::vector<Mesh> meshes     = {MakeBoxMesh(), MakeBoxMesh()};
    std::vector<Channel> palette = {{"Red", "PLA"}, {"Green", "PLA"}};

    std::vector<uint8_t> buffer = Export3mfFromMeshes(meshes, palette, -1, 0, preset);
    ASSERT_FALSE(buffer.empty());

    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    ASSERT_NE(FindEntry(entries, "[Content_Types].xml"), nullptr);
    ASSERT_NE(FindEntry(entries, "_rels/.rels"), nullptr);
    ASSERT_NE(FindEntry(entries, "3D/3dmodel.model"), nullptr);
    EXPECT_NE(FindEntry(entries, "Metadata/project_settings.config"), nullptr);
    EXPECT_NE(FindEntry(entries, "Metadata/model_settings.config"), nullptr);
    EXPECT_NE(FindEntry(entries, "Metadata/slice_info.config"), nullptr);
    EXPECT_NE(FindEntry(entries, "Metadata/cut_information.xml"), nullptr);
    EXPECT_NE(FindEntry(entries, "Metadata/filament_sequence.json"), nullptr);
}

TEST(SlicerPreset, ProjectSettingsContainsPatchedFilaments) {
    if (!PresetFileExists()) { GTEST_SKIP() << "Preset file not found"; }
    SlicerPreset preset = MakeTestPreset();

    std::vector<Mesh> meshes     = {MakeBoxMesh()};
    std::vector<Channel> palette = {{"Red", "PLA"}};

    std::vector<uint8_t> buffer = Export3mfFromMeshes(meshes, palette, -1, 0, preset);
    ASSERT_FALSE(buffer.empty());
    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    std::string project_settings =
        EntryAsString(FindEntry(entries, "Metadata/project_settings.config"));
    ASSERT_FALSE(project_settings.empty());

    nlohmann::json j = nlohmann::json::parse(project_settings);
    EXPECT_EQ(j["filament_colour"][0], "#C12E1F");
    EXPECT_EQ(j["filament_colour"][1], "#00AE42");
    EXPECT_EQ(j["filament_colour"][2], "#0A2989");
    EXPECT_EQ(j["filament_multi_colour"][0], "#C12E1F");
    EXPECT_EQ(j["filament_vendor"][0], "Bambu Lab");
    EXPECT_EQ(j["filament_ids"][0], "GFA00");
    EXPECT_EQ(j["from"], "project");
    EXPECT_TRUE(j.contains("layer_height"));
    EXPECT_TRUE(j.contains("printer_model"));
}

TEST(SlicerPreset, ModelSettingsXmlContainsObjectsAndExtruders) {
    if (!PresetFileExists()) { GTEST_SKIP() << "Preset file not found"; }
    SlicerPreset preset = MakeTestPreset();

    std::vector<Mesh> meshes     = {MakeBoxMesh(), MakeBoxMesh()};
    std::vector<Channel> palette = {{"Red", "PLA"}, {"Green", "PLA"}};

    std::vector<uint8_t> buffer = Export3mfFromMeshes(meshes, palette, -1, 0, preset);
    ASSERT_FALSE(buffer.empty());
    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    std::string model_settings =
        EntryAsString(FindEntry(entries, "Metadata/model_settings.config"));
    ASSERT_FALSE(model_settings.empty());

    EXPECT_NE(model_settings.find("<config>"), std::string::npos);
    EXPECT_NE(model_settings.find("plater_id"), std::string::npos);
    EXPECT_NE(model_settings.find("Red - PLA"), std::string::npos);
    EXPECT_NE(model_settings.find("Green - PLA"), std::string::npos);
    EXPECT_NE(model_settings.find("extruder\" value=\"1\""), std::string::npos);
    EXPECT_NE(model_settings.find("extruder\" value=\"2\""), std::string::npos);
}

TEST(SlicerPreset, ExplicitSlotsMappingSurvivesDroppedMesh) {
    if (!PresetFileExists()) { GTEST_SKIP() << "Preset file not found"; }
    SlicerPreset preset = MakeTestPreset();

    std::vector<Mesh> meshes       = {MakeBoxMesh(), MakeDegenerateMesh(), MakeBoxMesh()};
    std::vector<std::string> names = {"ObjA", "ObjDeg", "ObjC"};
    std::vector<int> slots         = {1, 8, 2};

    std::vector<Channel> palette(8);
    for (std::size_t i = 0; i < palette.size(); ++i) {
        palette[i].color     = "Slot" + std::to_string(i + 1);
        palette[i].material  = "PLA";
        palette[i].hex_color = "#FFFFFF";
    }

    std::vector<uint8_t> buffer = Export3mfFromMeshes(meshes, palette, names, slots, preset);
    ASSERT_FALSE(buffer.empty());
    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    std::string model_settings =
        EntryAsString(FindEntry(entries, "Metadata/model_settings.config"));
    ASSERT_FALSE(model_settings.empty());

    EXPECT_NE(model_settings.find("ObjA"), std::string::npos);
    EXPECT_NE(model_settings.find("ObjC"), std::string::npos);
    EXPECT_EQ(model_settings.find("ObjDeg"), std::string::npos);
    EXPECT_NE(model_settings.find("extruder\" value=\"1\""), std::string::npos);
    EXPECT_NE(model_settings.find("extruder\" value=\"2\""), std::string::npos);
    EXPECT_EQ(model_settings.find("extruder\" value=\"8\""), std::string::npos);
}

TEST(SlicerPreset, ExportWithoutPresetStillWorks) {
    std::vector<Mesh> meshes     = {MakeBoxMesh()};
    std::vector<Channel> palette = {{"Red", "PLA"}};

    std::vector<uint8_t> buffer = Export3mfFromMeshes(meshes, palette, -1, 0);
    ASSERT_FALSE(buffer.empty());

    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    ASSERT_NE(FindEntry(entries, "3D/3dmodel.model"), nullptr);
    EXPECT_EQ(FindEntry(entries, "Metadata/project_settings.config"), nullptr);
    EXPECT_EQ(FindEntry(entries, "Metadata/model_settings.config"), nullptr);
    EXPECT_EQ(FindEntry(entries, "Metadata/slice_info.config"), nullptr);
    EXPECT_EQ(FindEntry(entries, "Metadata/cut_information.xml"), nullptr);
    EXPECT_EQ(FindEntry(entries, "Metadata/filament_sequence.json"), nullptr);
    std::string model_xml = EntryAsString(FindEntry(entries, "3D/3dmodel.model"));
    EXPECT_NE(model_xml.find("unit=\"millimeter\""), std::string::npos);
    EXPECT_NE(model_xml.find("<metadata name=\"Application\">ChromaPrint3D</metadata>"),
              std::string::npos);
}

TEST(SlicerPreset, WriteToTempFile) {
    if (!PresetFileExists()) { GTEST_SKIP() << "Preset file not found"; }
    SlicerPreset preset = MakeTestPreset();

    std::vector<Mesh> meshes     = {MakeBoxMesh(), MakeBoxMesh()};
    std::vector<Channel> palette = {{"Red", "PLA"}, {"Green", "PLA"}};

    std::vector<uint8_t> buffer = Export3mfFromMeshes(meshes, palette, -1, 0, preset);
    ASSERT_FALSE(buffer.empty());

    auto tmp = std::filesystem::temp_directory_path() / "chromaprint3d_test_preset.3mf";
    {
        std::ofstream ofs(tmp, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(buffer.data()),
                  static_cast<std::streamsize>(buffer.size()));
    }
    EXPECT_TRUE(std::filesystem::exists(tmp));
}

TEST(SlicerPreset, FaceDownRotatesMeshesAroundGlobalBounds) {
    Mesh mesh_a;
    mesh_a.vertices = {
        Vec3f{0.0f, 0.0f, 0.0f},
        Vec3f{1.0f, 0.0f, 0.0f},
        Vec3f{0.0f, 0.0f, 1.0f},
    };
    mesh_a.indices = {Vec3i{0, 1, 2}};

    Mesh mesh_b;
    mesh_b.vertices = {
        Vec3f{10.0f, 0.0f, 5.0f},
        Vec3f{12.0f, 0.0f, 5.0f},
        Vec3f{10.0f, 0.0f, 7.0f},
    };
    mesh_b.indices = {Vec3i{0, 1, 2}};

    std::vector<Mesh> meshes     = {mesh_a, mesh_b};
    std::vector<Channel> palette = {{"Red", "PLA"}, {"Blue", "PLA"}};

    std::vector<uint8_t> face_up =
        Export3mfFromMeshes(meshes, palette, -1, 0, FaceOrientation::FaceUp);
    std::vector<uint8_t> face_down =
        Export3mfFromMeshes(meshes, palette, -1, 0, FaceOrientation::FaceDown);
    ASSERT_FALSE(face_up.empty());
    ASSERT_FALSE(face_down.empty());

    const std::vector<ZipEntry> up_entries   = ParseZipEntries(face_up);
    const std::vector<ZipEntry> down_entries = ParseZipEntries(face_down);
    const std::string model_up   = EntryAsString(FindEntry(up_entries, "3D/3dmodel.model"));
    const std::string model_down = EntryAsString(FindEntry(down_entries, "3D/3dmodel.model"));
    ASSERT_FALSE(model_up.empty());
    ASSERT_FALSE(model_down.empty());

    const std::vector<Vec3f> up_vertices   = ParseVerticesFromModelXml(model_up);
    const std::vector<Vec3f> down_vertices = ParseVerticesFromModelXml(model_down);
    ASSERT_EQ(up_vertices.size(), down_vertices.size());
    ASSERT_EQ(up_vertices.size(), 6u);

    // Global bounds from face-up geometry: minX=0, maxX=12, minZ=0, maxZ=7.
    const float sum_x = 12.0f;
    const float sum_z = 7.0f;
    for (std::size_t i = 0; i < up_vertices.size(); ++i) {
        EXPECT_NEAR(down_vertices[i].x, sum_x - up_vertices[i].x, 1e-5f);
        EXPECT_NEAR(down_vertices[i].y, up_vertices[i].y, 1e-5f);
        EXPECT_NEAR(down_vertices[i].z, sum_z - up_vertices[i].z, 1e-5f);
    }
}

// --- BuildLayerConfigRanges tests ---

TEST(LayerConfigRanges, EmptyWhenNoBaseLayers) {
    SlicerPreset preset;
    preset.base_layers     = 0;
    preset.color_layers    = 5;
    preset.layer_height_mm = 0.08f;
    preset.nozzle          = NozzleSize::N04;
    preset.face            = FaceOrientation::FaceUp;

    std::string xml = detail::BuildLayerConfigRanges(preset);
    EXPECT_TRUE(xml.empty());
}

TEST(LayerConfigRanges, EmptyWhenCoarseEqualsFineLH) {
    SlicerPreset preset;
    preset.base_layers     = 10;
    preset.color_layers    = 5;
    preset.layer_height_mm = 0.20f;
    preset.nozzle          = NozzleSize::N04;
    preset.face            = FaceOrientation::FaceUp;

    std::string xml = detail::BuildLayerConfigRanges(preset);
    EXPECT_TRUE(xml.empty());
}

TEST(LayerConfigRanges, FaceUpOnlyBaseRange) {
    SlicerPreset preset;
    preset.base_layers     = 10;
    preset.color_layers    = 5;
    preset.layer_height_mm = 0.08f;
    preset.nozzle          = NozzleSize::N04;
    preset.face            = FaceOrientation::FaceUp;

    std::string xml = detail::BuildLayerConfigRanges(preset);
    ASSERT_FALSE(xml.empty());

    // Only base region should be present: z=0..0.8 at 0.2mm
    EXPECT_NE(xml.find("min_z=\"0\""), std::string::npos);
    EXPECT_NE(xml.find("layer_height"), std::string::npos);
    EXPECT_NE(xml.find("extruder"), std::string::npos);
    EXPECT_NE(xml.find(">0</option>"), std::string::npos); // extruder=0

    // Color region should NOT be in the XML (uses project default)
    auto count_range = [](const std::string& s) {
        std::size_t n = 0, pos = 0;
        while ((pos = s.find("<range ", pos)) != std::string::npos) {
            ++n;
            ++pos;
        }
        return n;
    };
    EXPECT_EQ(count_range(xml), 1u);
}

TEST(LayerConfigRanges, FaceDownBaseRangeAboveColor) {
    SlicerPreset preset;
    preset.base_layers     = 10;
    preset.color_layers    = 5;
    preset.layer_height_mm = 0.08f;
    preset.nozzle          = NozzleSize::N04;
    preset.face            = FaceOrientation::FaceDown;

    std::string xml = detail::BuildLayerConfigRanges(preset);
    ASSERT_FALSE(xml.empty());

    // Only base region: z=0.4..1.2 at 0.2mm (color is at bottom, uses default)
    auto count_range = [](const std::string& s) {
        std::size_t n = 0, pos = 0;
        while ((pos = s.find("<range ", pos)) != std::string::npos) {
            ++n;
            ++pos;
        }
        return n;
    };
    EXPECT_EQ(count_range(xml), 1u);
    EXPECT_NE(xml.find("extruder"), std::string::npos);
}

TEST(LayerConfigRanges, DoubleSidedFaceUpPlacesBaseInMiddle) {
    SlicerPreset preset;
    preset.base_layers     = 10;
    preset.color_layers    = 5;
    preset.layer_height_mm = 0.08f;
    preset.nozzle          = NozzleSize::N04;
    preset.face            = FaceOrientation::FaceUp;
    preset.double_sided    = true;

    const std::string xml = detail::BuildLayerConfigRanges(preset);
    ASSERT_FALSE(xml.empty());

    constexpr double kTol     = 1e-6;
    const auto [min_z, max_z] = ParseFirstRangeBounds(xml);
    EXPECT_NEAR(min_z, 0.4, kTol);
    EXPECT_NEAR(max_z, 1.2, kTol);
}

TEST(LayerConfigRanges, DoubleSidedFaceDownKeepsBaseInMiddle) {
    SlicerPreset preset;
    preset.base_layers     = 10;
    preset.color_layers    = 5;
    preset.layer_height_mm = 0.08f;
    preset.nozzle          = NozzleSize::N04;
    preset.face            = FaceOrientation::FaceDown;
    preset.double_sided    = true;

    const std::string xml = detail::BuildLayerConfigRanges(preset);
    ASSERT_FALSE(xml.empty());

    constexpr double kTol     = 1e-6;
    const auto [min_z, max_z] = ParseFirstRangeBounds(xml);
    EXPECT_NEAR(min_z, 0.4, kTol);
    EXPECT_NEAR(max_z, 1.2, kTol);
}

TEST(LayerConfigRanges, N02NozzleUsesHalfDiameterCoarseLH) {
    SlicerPreset preset;
    preset.base_layers     = 10;
    preset.color_layers    = 5;
    preset.layer_height_mm = 0.08f;
    preset.nozzle          = NozzleSize::N02;
    preset.face            = FaceOrientation::FaceUp;

    std::string xml = detail::BuildLayerConfigRanges(preset);
    ASSERT_FALSE(xml.empty());

    // 0.2mm nozzle -> 0.1mm coarse layer height; only one range
    auto count_range = [](const std::string& s) {
        std::size_t n = 0, pos = 0;
        while ((pos = s.find("<range ", pos)) != std::string::npos) {
            ++n;
            ++pos;
        }
        return n;
    };
    EXPECT_EQ(count_range(xml), 1u);
    EXPECT_NE(xml.find("extruder"), std::string::npos);
}

TEST(LayerConfigRanges, ExportedZipContainsLayerRangesFile) {
    if (!PresetFileExists()) { GTEST_SKIP() << "Preset file not found"; }
    SlicerPreset preset = MakeTestPreset();
    preset.base_layers  = 10;
    preset.color_layers = 5;

    std::vector<Mesh> meshes     = {MakeBoxMesh(), MakeBoxMesh()};
    std::vector<Channel> palette = {{"Red", "PLA"}, {"Green", "PLA"}};

    std::vector<uint8_t> buffer = Export3mfFromMeshes(meshes, palette, -1, 0, preset);
    ASSERT_FALSE(buffer.empty());

    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    const ZipEntry* entry         = FindEntry(entries, "Metadata/layer_config_ranges.xml");
    ASSERT_NE(entry, nullptr);

    std::string xml = EntryAsString(entry);
    EXPECT_NE(xml.find("<objects>"), std::string::npos);
    EXPECT_NE(xml.find("layer_height"), std::string::npos);
}

TEST(LayerConfigRanges, EmptyWhenCustomBaseLayers) {
    SlicerPreset preset;
    preset.base_layers        = 10;
    preset.color_layers       = 5;
    preset.layer_height_mm    = 0.08f;
    preset.nozzle             = NozzleSize::N04;
    preset.face               = FaceOrientation::FaceUp;
    preset.custom_base_layers = true;

    std::string xml = detail::BuildLayerConfigRanges(preset);
    EXPECT_TRUE(xml.empty());
}

// --- Transparent layer tests ---

TEST(LayerConfigRanges, FaceDownWithTransparentLayerOffsetsBase) {
    SlicerPreset preset;
    preset.base_layers          = 10;
    preset.color_layers         = 5;
    preset.layer_height_mm      = 0.08f;
    preset.nozzle               = NozzleSize::N04;
    preset.face                 = FaceOrientation::FaceDown;
    preset.transparent_layer_mm = 0.04f;

    std::string xml = detail::BuildLayerConfigRanges(preset);
    ASSERT_FALSE(xml.empty());

    constexpr double kTol     = 1e-6;
    const auto [min_z, max_z] = ParseFirstRangeBounds(xml);
    const double color_h      = 5 * 0.08;
    const double total_h      = (5 + 10) * 0.08;
    EXPECT_NEAR(min_z, color_h + 0.04, kTol);
    EXPECT_NEAR(max_z, total_h + 0.04, kTol);
}

TEST(LayerConfigRanges, FaceUpIgnoresTransparentLayer) {
    SlicerPreset preset;
    preset.base_layers          = 10;
    preset.color_layers         = 5;
    preset.layer_height_mm      = 0.08f;
    preset.nozzle               = NozzleSize::N04;
    preset.face                 = FaceOrientation::FaceUp;
    preset.transparent_layer_mm = 0.04f;

    std::string xml = detail::BuildLayerConfigRanges(preset);
    ASSERT_FALSE(xml.empty());

    constexpr double kTol     = 1e-6;
    const auto [min_z, max_z] = ParseFirstRangeBounds(xml);
    EXPECT_NEAR(min_z, 0.0, kTol);
    EXPECT_NEAR(max_z, 10 * 0.08, kTol);
}

TEST(TransparentLayer, BuildTransparentLayerFromModelIR) {
    ModelIR model;
    model.width        = 3;
    model.height       = 3;
    model.color_layers = 5;
    model.base_layers  = 2;

    VoxelGrid grid;
    grid.width      = 3;
    grid.height     = 3;
    grid.num_layers = 7; // 2 base + 5 color
    grid.ooc.assign(static_cast<size_t>(3 * 3 * 7), 0);
    grid.Set(0, 0, 0, true);
    grid.Set(1, 0, 0, true);
    grid.Set(0, 1, 0, true);
    grid.Set(1, 1, 3, true);
    model.voxel_grids.push_back(std::move(grid));

    Mesh mesh = BuildTransparentLayerFromModelIR(model, 1.0f, 0.08f, 0.04f);
    ASSERT_FALSE(mesh.vertices.empty());
    ASSERT_FALSE(mesh.indices.empty());

    float total_z = static_cast<float>(grid.num_layers) * 0.08f;
    float min_z   = std::numeric_limits<float>::max();
    float max_z   = std::numeric_limits<float>::lowest();
    for (const auto& v : mesh.vertices) {
        min_z = std::min(min_z, v.z);
        max_z = std::max(max_z, v.z);
    }
    EXPECT_NEAR(min_z, total_z, 1e-4f);
    EXPECT_NEAR(max_z, total_z + 0.04f, 1e-4f);
}

TEST(TransparentLayer, BuildMeshNamesAndSlotsWithTransparent) {
    std::vector<Channel> palette = {
        {"Red", "PLA", "#C12E1F"},
        {"Green", "PLA", "#00AE42"},
    };

    auto ns = BuildMeshNamesAndSlots(4, palette, 0, 1, true);
    ASSERT_EQ(ns.names.size(), 4u);
    ASSERT_EQ(ns.slots.size(), 4u);

    EXPECT_NE(ns.names[0].find("Red"), std::string::npos);
    EXPECT_NE(ns.names[1].find("Green"), std::string::npos);
    EXPECT_NE(ns.names[2].find("Base"), std::string::npos);
    EXPECT_EQ(ns.names[3], "Transparent Layer");

    EXPECT_EQ(ns.slots[0], 1);
    EXPECT_EQ(ns.slots[1], 2);
    EXPECT_EQ(ns.slots[2], 1);
    EXPECT_EQ(ns.slots[3], 4);
}

TEST(TransparentLayer, BuildMeshNamesAndSlotsNoBase) {
    std::vector<Channel> palette = {
        {"Red", "PLA", "#C12E1F"},
    };

    auto ns = BuildMeshNamesAndSlots(2, palette, -1, 0, true);
    ASSERT_EQ(ns.names.size(), 2u);
    EXPECT_EQ(ns.names[1], "Transparent Layer");
    EXPECT_EQ(ns.slots[1], 2);
}

TEST(TransparentLayer, ExportWithTransparentLayerMesh) {
    if (!PresetFileExists()) { GTEST_SKIP() << "Preset file not found"; }
    SlicerPreset preset         = MakeTestPreset();
    preset.transparent_layer_mm = 0.04f;
    preset.face                 = FaceOrientation::FaceDown;
    FilamentSlot t_slot;
    t_slot.type        = "PLA";
    t_slot.colour      = "#FEFEFE";
    t_slot.settings_id = "Bambu PLA Basic @BBL P2S";
    preset.filaments.push_back(std::move(t_slot));

    std::vector<Mesh> meshes       = {MakeBoxMesh(), MakeBoxMesh(), MakeBoxMesh(), MakeBoxMesh()};
    std::vector<Channel> palette   = {{"Red", "PLA"}, {"Green", "PLA"}, {"Blue", "PLA"}};
    std::vector<std::string> names = {"Red PLA", "Green PLA", "Blue PLA", "Transparent Layer"};
    std::vector<int> slots         = {1, 2, 3, 4};

    std::vector<uint8_t> buffer =
        Export3mfFromMeshes(meshes, palette, names, slots, preset, FaceOrientation::FaceDown);
    ASSERT_FALSE(buffer.empty());

    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    std::string model_settings =
        EntryAsString(FindEntry(entries, "Metadata/model_settings.config"));
    ASSERT_FALSE(model_settings.empty());
    EXPECT_NE(model_settings.find("Transparent Layer"), std::string::npos);
    EXPECT_NE(model_settings.find("extruder\" value=\"4\""), std::string::npos);
}
