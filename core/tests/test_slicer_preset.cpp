#include <gtest/gtest.h>

#include "chromaprint3d/bambu_preset_catalog.h"
#include "chromaprint3d/export_3mf.h"
#include "chromaprint3d/slicer_preset.h"
#include "chromaprint3d/voxel.h"
#include "geo/bambu_metadata.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
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

std::string GetDataDir() {
    const char* env = std::getenv("CHROMAPRINT3D_DATA_DIR");
    if (env) return env;
    // Legacy env var still respected; treat its value as `<data_dir>/presets`.
    const char* legacy = std::getenv("CHROMAPRINT3D_PRESET_DIR");
    if (legacy) {
        std::filesystem::path p(legacy);
        if (p.filename() == "presets") return p.parent_path().string();
        return legacy;
    }
    const char* pwd = std::getenv("PWD");
    return std::string(pwd ? pwd : ".") + "/data";
}

bool PresetFilesExist() {
    return std::filesystem::exists(GetDataDir() + "/preset_bases/bambu_p2s_0.08mm_n04.json") &&
           std::filesystem::exists(GetDataDir() + "/presets/machines.json");
}

std::optional<BambuPresetCatalog> LoadCatalogOrNull() {
    try {
        return BambuPresetCatalog::LoadFromDir(GetDataDir());
    } catch (const std::exception&) { return std::nullopt; }
}

SlicerPreset MakeTestPreset(const BambuPresetCatalog& catalog,
                            FaceOrientation face = FaceOrientation::FaceUp) {
    PrintProfile profile;
    profile.layer_height_mm  = 0.08f;
    profile.nozzle_size      = NozzleSize::N04;
    profile.face_orientation = face;
    profile.palette          = {
        {"Red", "PLA", "#C12E1F"}, {"Green", "PLA", "#00AE42"}, {"Blue", "PLA", "#0A2989"}};

    auto preset = SlicerPreset::FromProfile(catalog, profile, "Bambu Lab P2S");
    // Ensure each slot's colour matches the test palette regardless of FilamentConfig defaults.
    for (size_t i = 0; i < preset.filaments.size() && i < profile.palette.size(); ++i) {
        preset.filaments[i].colour = profile.palette[i].hex_color;
    }
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

/// Build a SlicerPreset for any registered machine with N user filament slots.
/// Used by the multi-machine variant / N-slot expansion test groups.
SlicerPreset MakeTestPresetForMachine(const BambuPresetCatalog& catalog,
                                       const std::string& machine_name, NozzleSize nozzle,
                                       std::size_t N,
                                       FaceOrientation face = FaceOrientation::FaceUp) {
    PrintProfile profile;
    profile.layer_height_mm  = 0.08f;
    profile.nozzle_size      = nozzle;
    profile.face_orientation = face;
    profile.palette.reserve(N);
    static const char* kPaletteHex[] = {
        "#C12E1F", "#00AE42", "#0A2989", "#FFFFFF", "#000000", "#F4EE2A", "#EC008C", "#0086D6",
    };
    for (std::size_t i = 0; i < N; ++i) {
        Channel ch;
        ch.color     = "Slot" + std::to_string(i + 1);
        ch.material  = "PLA";
        ch.hex_color = kPaletteHex[i % (sizeof(kPaletteHex) / sizeof(kPaletteHex[0]))];
        profile.palette.push_back(ch);
    }
    auto preset = SlicerPreset::FromProfile(catalog, profile, machine_name);
    for (size_t i = 0; i < preset.filaments.size() && i < profile.palette.size(); ++i) {
        preset.filaments[i].colour = profile.palette[i].hex_color;
    }
    return preset;
}

/// Extract `Metadata/project_settings.config` JSON from an exported 3MF buffer.
nlohmann::json ExtractProjectSettings(const std::vector<uint8_t>& buffer) {
    auto entries        = ParseZipEntries(buffer);
    const ZipEntry* ent = FindEntry(entries, "Metadata/project_settings.config");
    if (!ent) throw std::runtime_error("project_settings.config missing from 3MF");
    return nlohmann::json::parse(EntryAsString(ent));
}

/// Helper: build a 3MF buffer from a preset, return its parsed project_settings.config JSON.
nlohmann::json ExportAndExtractProjectSettings(const SlicerPreset& preset) {
    std::vector<Mesh> meshes(preset.filaments.size(), MakeBoxMesh());
    std::vector<Channel> palette(preset.filaments.size());
    for (size_t i = 0; i < preset.filaments.size(); ++i) {
        palette[i].color     = "Slot" + std::to_string(i + 1);
        palette[i].material  = "PLA";
        palette[i].hex_color = preset.filaments[i].colour;
    }
    auto buf = Export3mfFromMeshes(meshes, palette, -1, 0, preset);
    return ExtractProjectSettings(buf);
}

} // namespace

TEST(SlicerPreset, CatalogResolvesP2SDefault) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto spec = catalog->Resolve("Bambu Lab P2S", NozzleSize::N04, 0.08f);
    ASSERT_TRUE(spec.has_value());
    EXPECT_EQ(spec->machine_name, "Bambu Lab P2S");
    EXPECT_EQ(spec->extruder_topology, "single");
    EXPECT_EQ(spec->printer_template, "Bambu Lab P2S 0.4 nozzle");
    EXPECT_FALSE(spec->compatible_printers.empty());
}

TEST(SlicerPreset, CatalogReturnsNulloptForMissingNozzle) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    // 0.99mm layer height has no base file; Resolve returns nullopt.
    auto spec = catalog->Resolve("Bambu Lab P2S", NozzleSize::N04, 0.99f);
    EXPECT_FALSE(spec.has_value());
}

TEST(SlicerPreset, DoubleSidedForcesFaceDownPresetSelection) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    PrintProfile profile;
    profile.nozzle_size      = NozzleSize::N04;
    profile.face_orientation = FaceOrientation::FaceUp;
    profile.layer_height_mm  = 0.08f;
    profile.palette.push_back(Channel{"Red", "PLA"});

    const SlicerPreset single_sided =
        SlicerPreset::FromProfile(*catalog, profile, "Bambu Lab P2S", nullptr, false);
    const SlicerPreset double_sided =
        SlicerPreset::FromProfile(*catalog, profile, "Bambu Lab P2S", nullptr, true);

    EXPECT_EQ(single_sided.face, FaceOrientation::FaceUp);
    EXPECT_EQ(double_sided.face, FaceOrientation::FaceDown);
    EXPECT_TRUE(single_sided.machine_resolved());
    EXPECT_TRUE(double_sided.machine_resolved());
}

TEST(SlicerPreset, ExportWithPresetContainsBambuMetadata) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    SlicerPreset preset = MakeTestPreset(*catalog);

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
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    SlicerPreset preset = MakeTestPreset(*catalog);

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
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    SlicerPreset preset = MakeTestPreset(*catalog);

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
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    SlicerPreset preset = MakeTestPreset(*catalog);

    std::vector<Mesh> meshes       = {MakeBoxMesh(), Mesh{}, MakeBoxMesh()};
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
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    SlicerPreset preset = MakeTestPreset(*catalog);

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
    mesh_a.indices = {Vec3u{0, 1, 2}};

    Mesh mesh_b;
    mesh_b.vertices = {
        Vec3f{10.0f, 0.0f, 5.0f},
        Vec3f{12.0f, 0.0f, 5.0f},
        Vec3f{10.0f, 0.0f, 7.0f},
    };
    mesh_b.indices = {Vec3u{0, 1, 2}};

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
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    SlicerPreset preset = MakeTestPreset(*catalog);
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
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    SlicerPreset preset         = MakeTestPreset(*catalog, FaceOrientation::FaceDown);
    preset.transparent_layer_mm = 0.04f;
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

// ===========================================================================
// Group 1: $variant_indexed patch application
// ===========================================================================

TEST(VariantIndexed, AppliedToP2sK2) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j.contains("outer_wall_speed"));
    ASSERT_TRUE(j["outer_wall_speed"].is_array());
    EXPECT_EQ(j["outer_wall_speed"].size(), 2u);
    // chromaprint_patches.json: DD Std=50, DD HF=60 for outer_wall_speed.
    EXPECT_EQ(j["outer_wall_speed"][0], "50");
    EXPECT_EQ(j["outer_wall_speed"][1], "60");
}

TEST(VariantIndexed, AppliedToH2dK5IncludesTpuHf) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2D", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["outer_wall_speed"].is_array());
    EXPECT_EQ(j["outer_wall_speed"].size(), 5u);
    // H2D K_process=5 with print_extruder_variant=[DD Std, DD HF, DD Std, DD HF, DD TPU HF].
    // DD TPU HF entry must be present at index 4.
    EXPECT_EQ(j["outer_wall_speed"][4], "50"); // chromaprint_patches DD TPU HF -> 50
}

TEST(VariantIndexed, AppliedToX2dK4IncludesBowden) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab X2D", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["outer_wall_speed"].is_array());
    EXPECT_EQ(j["outer_wall_speed"].size(), 4u);
    // X2D K_process=4 print_extruder_variant=[DD Std, DD HF, Bowden Std, Bowden HF].
    // Bowden Std at index 2; chromaprint_patches Bowden Std = 50.
    EXPECT_EQ(j["outer_wall_speed"][2], "50");
    EXPECT_EQ(j["outer_wall_speed"][3], "50"); // Bowden HF = 50
}

TEST(VariantIndexed, MissingMappingThrows) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    // Build a P2S preset, then mutate the patches to remove a variant entry,
    // and verify BuildProjectSettings throws when it encounters the missing key.
    // We exercise this via direct dictionary surgery on a copy of the catalog's
    // patches; since SlicerPreset.machine.patches is a shared_ptr<const>, we
    // construct a fake patches instance and swap it into the spec.
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2D", NozzleSize::N04, 3);
    ChromaPrintPatches bad;
    // Provide outer_wall_speed but only with DD Std (missing DD HF / DD TPU HF / Bowden).
    bad.process_common["outer_wall_speed"] =
        R"({"$variant_indexed":{"Direct Drive Standard":"50"}})";
    preset.machine.patches = std::make_shared<const ChromaPrintPatches>(std::move(bad));

    std::vector<Mesh> meshes(preset.filaments.size(), MakeBoxMesh());
    std::vector<Channel> palette(preset.filaments.size());
    for (size_t i = 0; i < palette.size(); ++i) {
        palette[i].material  = "PLA";
        palette[i].hex_color = "#FFFFFF";
    }
    EXPECT_THROW(Export3mfFromMeshes(meshes, palette, -1, 0, preset), std::exception);
}

// ===========================================================================
// Group 2: N-slot expansion (filament_no_variant / filament_with_variant /
//          print_with_variant) across machines
// ===========================================================================

TEST(NSlotExpand, FilamentNoVariantP2sN8) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["filament_colour"].is_array());
    EXPECT_EQ(j["filament_colour"].size(), 8u);
    // Each slot should match its user palette colour (set in MakeTestPresetForMachine).
    EXPECT_EQ(j["filament_colour"][0], "#C12E1F");
    EXPECT_EQ(j["filament_colour"][1], "#00AE42");
    EXPECT_EQ(j["filament_colour"][7], "#0086D6");
    // filament_ids and filament_vendor also expand to N.
    ASSERT_TRUE(j["filament_ids"].is_array());
    EXPECT_EQ(j["filament_ids"].size(), 8u);
    ASSERT_TRUE(j["filament_vendor"].is_array());
    EXPECT_EQ(j["filament_vendor"].size(), 8u);
}

TEST(NSlotExpand, FilamentWithVariantP2sN8) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    // P2S K_process=2; N=8 -> nozzle_temperature length = 16.
    ASSERT_TRUE(j["nozzle_temperature"].is_array());
    EXPECT_EQ(j["nozzle_temperature"].size(), 16u);
    ASSERT_TRUE(j["filament_max_volumetric_speed"].is_array());
    EXPECT_EQ(j["filament_max_volumetric_speed"].size(), 16u);
}

TEST(NSlotExpand, FilamentWithVariantH2dN8) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2D", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    // H2D K_per_extruder=2 (extruder 0 = [DD Std, DD HF]); N=8 -> length = 16.
    // Filament arrays use K_per_extruder, not K_process (=5 for H2D).
    ASSERT_TRUE(j["nozzle_temperature"].is_array());
    EXPECT_EQ(j["nozzle_temperature"].size(), 16u);
    // base nozzle_temperature aligned to K_per_extruder=2: ['220','220'].
    for (std::size_t k = 0; k < 16; ++k) { EXPECT_EQ(j["nozzle_temperature"][k], "220"); }
    // filament_max_volumetric_speed: H2D base aligned to K_per_extruder=2 = ['25','40'].
    ASSERT_TRUE(j["filament_max_volumetric_speed"].is_array());
    ASSERT_EQ(j["filament_max_volumetric_speed"].size(), 16u);
    EXPECT_EQ(j["filament_max_volumetric_speed"][0], "25");
    EXPECT_EQ(j["filament_max_volumetric_speed"][1], "40");
    // Slot 2 (indices 2..3) repeats the same K_per_extruder=2 slice.
    EXPECT_EQ(j["filament_max_volumetric_speed"][2],
              j["filament_max_volumetric_speed"][0]);
    EXPECT_EQ(j["filament_max_volumetric_speed"][3],
              j["filament_max_volumetric_speed"][1]);
    // Slot 8 (indices 14..15) is the tail.
    EXPECT_EQ(j["filament_max_volumetric_speed"][14], "25");
    EXPECT_EQ(j["filament_max_volumetric_speed"][15], "40");
}

TEST(NSlotExpand, FilamentWithVariantA1N8) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    // A1 K_process == K_filament == 1 (single direct-drive variant).
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab A1", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["nozzle_temperature"].is_array());
    EXPECT_EQ(j["nozzle_temperature"].size(), 8u);
    // All 8 entries should be the same (1 variant replicated 8 times).
    for (std::size_t i = 1; i < 8; ++i) {
        EXPECT_EQ(j["nozzle_temperature"][i], j["nozzle_temperature"][0]);
    }
}

TEST(NSlotExpand, FilamentWithVariantX2dN8) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    // X2D K_process=4, K_per_extruder=2 (extruder 0 = [DD Std, DD HF]; extruder 1 = Bowden).
    // Filament arrays use K_per_extruder=2 (extruder 0 only).
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab X2D", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["nozzle_temperature"].is_array());
    EXPECT_EQ(j["nozzle_temperature"].size(), 16u);
    // filament_max_volumetric_speed: X2D base truncated to K_per_extruder=2 = ['21','40'].
    ASSERT_TRUE(j["filament_max_volumetric_speed"].is_array());
    EXPECT_EQ(j["filament_max_volumetric_speed"].size(), 16u);
    EXPECT_EQ(j["filament_max_volumetric_speed"][0], "21");
    EXPECT_EQ(j["filament_max_volumetric_speed"][1], "40");
    // Slot 2 (indices 2..3) repeats the same K_per_extruder=2 slice.
    EXPECT_EQ(j["filament_max_volumetric_speed"][2],
              j["filament_max_volumetric_speed"][0]);
    EXPECT_EQ(j["filament_max_volumetric_speed"][3],
              j["filament_max_volumetric_speed"][1]);
}

TEST(NSlotExpand, PrintWithVariantKeepsKProcess) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2D", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    // print_options_with_variant arrays must NOT be expanded to N×K_process.
    ASSERT_TRUE(j["outer_wall_speed"].is_array());
    EXPECT_EQ(j["outer_wall_speed"].size(), 5u); // K_process for H2D
    ASSERT_TRUE(j["inner_wall_speed"].is_array());
    EXPECT_EQ(j["inner_wall_speed"].size(), 5u);
}

TEST(NSlotExpand, ThreeKeySetsAreDisjoint) {
    using detail::PrintWithVariantKeys;
    using detail::FilamentWithVariantKeys;
    const auto& a = PrintWithVariantKeys();
    const auto& b = FilamentWithVariantKeys();
    for (const auto& k : a) {
        EXPECT_EQ(b.count(k), 0u) << "Key `" << k << "` is in both Print and Filament with-variant sets";
    }
    for (const auto& k : b) {
        EXPECT_EQ(a.count(k), 0u) << "Key `" << k << "` is in both Filament and Print with-variant sets";
    }
}

// ===========================================================================
// Group 3: Mandatory variant meta fields
// (extruder_variant_list / filament_extruder_variant / filament_self_index)
// ===========================================================================

TEST(VariantMeta, P2sN8FilamentSelfIndex) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["filament_self_index"].is_array());
    EXPECT_EQ(j["filament_self_index"].size(), 16u); // N×K_process = 8×2
    // Pattern: ["1","1","2","2",...,"8","8"]
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(j["filament_self_index"][2 * i], std::to_string(i + 1));
        EXPECT_EQ(j["filament_self_index"][2 * i + 1], std::to_string(i + 1));
    }
}

TEST(VariantMeta, P2sN8FilamentExtruderVariant) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["filament_extruder_variant"].is_array());
    EXPECT_EQ(j["filament_extruder_variant"].size(), 16u);
    // Pattern: extruder 0's variants repeated N=8 times.
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(j["filament_extruder_variant"][2 * i], "Direct Drive Standard");
        EXPECT_EQ(j["filament_extruder_variant"][2 * i + 1], "Direct Drive High Flow");
    }
}

TEST(VariantMeta, H2dN8FilamentSelfIndexLength) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2D", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["filament_self_index"].is_array());
    EXPECT_EQ(j["filament_self_index"].size(), 16u); // N×K_per_extruder = 8×2
    // Slot 1 repeated K_per_extruder=2 times at the head.
    EXPECT_EQ(j["filament_self_index"][0], "1");
    EXPECT_EQ(j["filament_self_index"][1], "1");
    // Slot 8 at the tail.
    EXPECT_EQ(j["filament_self_index"][14], "8");
    EXPECT_EQ(j["filament_self_index"][15], "8");
}

TEST(VariantMeta, H2dN8FilamentExtruderVariantUsesExtruder0Only) {
    // BambuStudio's filament arrays use extruder 0's variants only
    // (DD Std + DD HF for H2D). DD TPU HF is in extruder 1 of H2D
    // (extruder_variant_list[1]) and lives ONLY in print_extruder_variant
    // (K_process), NOT in filament_extruder_variant (K_per_extruder).
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2D", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["filament_extruder_variant"].is_array());
    EXPECT_EQ(j["filament_extruder_variant"].size(), 16u); // N×K_per_extruder
    // Pattern: [DD Std, DD HF] × 8
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(j["filament_extruder_variant"][2 * i], "Direct Drive Standard");
        EXPECT_EQ(j["filament_extruder_variant"][2 * i + 1], "Direct Drive High Flow");
    }
    // Critically: NO entry should contain TPU HF (it's extruder 1 only).
    for (const auto& v : j["filament_extruder_variant"]) {
        EXPECT_NE(v.get<std::string>(), "Direct Drive TPU High Flow");
    }
}

TEST(VariantMeta, X2dN8FilamentSelfIndexLength) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab X2D", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["filament_self_index"].is_array());
    EXPECT_EQ(j["filament_self_index"].size(), 16u); // N×K_per_extruder=8×2
}

TEST(VariantMeta, X2dN8FilamentExtruderVariantUsesExtruder0Only) {
    // X2D extruder 0 = DD Std + DD HF; extruder 1 = Bowden Std + Bowden HF.
    // filament_extruder_variant uses ONLY extruder 0's variants.
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab X2D", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["filament_extruder_variant"].is_array());
    EXPECT_EQ(j["filament_extruder_variant"].size(), 16u);
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(j["filament_extruder_variant"][2 * i], "Direct Drive Standard");
        EXPECT_EQ(j["filament_extruder_variant"][2 * i + 1], "Direct Drive High Flow");
    }
    // No entry should contain Bowden (it's extruder 1 only).
    for (const auto& v : j["filament_extruder_variant"]) {
        const std::string& s = v.get_ref<const std::string&>();
        EXPECT_EQ(s.find("Bowden"), std::string::npos)
            << "Bowden variant must NOT appear in filament_extruder_variant: " << s;
    }
}

// ===========================================================================
// Group 5: print_extruder_variant preserves all variants (K_process)
// ===========================================================================

TEST(PrintExtruderVariant, H2dPreservesTpuHf) {
    // print_extruder_variant retains K_process (5 for H2D),
    // including DD TPU HF on extruder 1.
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2D", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["print_extruder_variant"].is_array());
    EXPECT_EQ(j["print_extruder_variant"].size(), 5u);
    EXPECT_EQ(j["print_extruder_variant"][4], "Direct Drive TPU High Flow");
}

TEST(PrintExtruderVariant, X2dPreservesBowden) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab X2D", NozzleSize::N04, 8);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["print_extruder_variant"].is_array());
    EXPECT_EQ(j["print_extruder_variant"].size(), 4u);
    EXPECT_EQ(j["print_extruder_variant"][2], "Bowden Standard");
    EXPECT_EQ(j["print_extruder_variant"][3], "Bowden High Flow");
}

// ===========================================================================
// Group 6: Patch value alignment with H2C primary machine real-file values
// (chromaprint_patches.json field audit)
// ===========================================================================

TEST(PatchValueAlignment, OuterWallSpeedDdStd50) {
    // All machines: outer_wall_speed[0] (DD Std) = 50 (slow speed for color edges).
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["outer_wall_speed"].is_array());
    EXPECT_EQ(j["outer_wall_speed"][0], "50");
}

TEST(PatchValueAlignment, SparseInfillSpeedH2cDdHf100) {
    // H2C primary machine: sparse_infill_speed.DD HF = 100 (not P2S's 150).
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2C", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["sparse_infill_speed"].is_array());
    EXPECT_EQ(j["sparse_infill_speed"].size(), 4u); // K_process=4 for H2C
    EXPECT_EQ(j["sparse_infill_speed"][0], "50"); // DD Std
    EXPECT_EQ(j["sparse_infill_speed"][1], "100"); // DD HF (H2C value)
}

TEST(PatchValueAlignment, BottomColorPenetrationLayers1) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    EXPECT_EQ(j["bottom_color_penetration_layers"], "1");
    EXPECT_EQ(j["top_color_penetration_layers"], "1");
}

TEST(PatchValueAlignment, ShellLayersBothOne) {
    // Bottom + top shell layers both = 1 (symmetric design, matches user
    // real-file `varesa_foreground`; H2C/P2S use thickness instead but
    // BambuStudio's max(layers, thickness/h) rule makes layers=1 + thickness=0
    // the most explicit "1 layer" expression). Color penetration system:
    //   *_color_penetration_layers=1 + *_shell_layers=1 + *_shell_thickness=0
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    EXPECT_EQ(j["bottom_shell_layers"], "1");
    EXPECT_EQ(j["top_shell_layers"], "1");
}

TEST(PatchValueAlignment, ShellThicknessZero) {
    // thickness=0 means "trust shell_layers count exactly", no minimum-thickness
    // coercion. User real-file 3MFs all use bottom_shell_thickness=0 (varesa
    // also has top_shell_thickness=0 for symmetric design).
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    EXPECT_EQ(j["bottom_shell_thickness"], "0");
    EXPECT_EQ(j["top_shell_thickness"], "0");
}

TEST(PatchValueAlignment, SurfaceDensity100pct) {
    // Solid fill density 100% on both surfaces (no gaps for color continuity).
    // User real-file all 3 use 100%.
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    EXPECT_EQ(j["bottom_surface_density"], "100%");
    EXPECT_EQ(j["top_surface_density"], "100%");
}

TEST(PatchValueAlignment, SurfacePatternsMonotonicForQuality) {
    // varesa-standard: monotonic mode mitigates surface-quality issues caused
    // by zig-zag (straight-line) print direction reversals.
    // - bottom_surface_pattern: system default is already 'monotonic' (no patch needed).
    // - top_surface_pattern: system default is 'monotonicline'; patch to 'monotonic'
    //   per user varesa real-file standard.
    // - sparse_infill_pattern: system default is 'gyroid' (unsuitable for
    //   multi-color planar fill); patch to 'zig-zag' per all 3 user real-files.
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    EXPECT_EQ(j["bottom_surface_pattern"], "monotonic");
    EXPECT_EQ(j["top_surface_pattern"], "monotonic");
    EXPECT_EQ(j["sparse_infill_pattern"], "zig-zag");
}

TEST(PatchValueAlignment, MinBeadWidth65pct) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    EXPECT_EQ(j["min_bead_width"], "65%");
}

TEST(PatchValueAlignment, InitialLayerFlowRatio12) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    EXPECT_EQ(j["initial_layer_flow_ratio"], "1.2");
}

TEST(PatchValueAlignment, ElefantFootCompensationZero) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    EXPECT_EQ(j["elefant_foot_compensation"], "0");
}

// ===========================================================================
// Group 7: Metadata field cleanup (mirrors BambuStudio user real-file output)
// ===========================================================================

TEST(MetadataCleanup, NoCompatibleExpressionGroupFields) {
    // BambuStudio's `add_if_some_non_empty` (PresetBundle.cpp:264) only writes
    // these when non-empty; user real-file 3MFs omit them. CP3D should not
    // write empty arrays.
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    EXPECT_FALSE(j.contains("compatible_machine_expression_group"))
        << "compatible_machine_expression_group should not be written (BambuStudio omits)";
    EXPECT_FALSE(j.contains("compatible_process_expression_group"))
        << "compatible_process_expression_group should not be written";
}

TEST(MetadataCleanup, NoIsCustomDefinedField) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    EXPECT_FALSE(j.contains("is_custom_defined"))
        << "is_custom_defined should not be written (absent in real-file 3MFs)";
}

TEST(MetadataCleanup, InheritsGroupHeadIsProcessTemplate) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j      = ExportAndExtractProjectSettings(preset);
    ASSERT_TRUE(j["inherits_group"].is_array());
    EXPECT_EQ(j["inherits_group"][0], "0.08mm High Quality @BBL P2S");
    // Tail entries (printer + N filaments) should be empty.
    for (std::size_t i = 1; i < j["inherits_group"].size(); ++i) {
        EXPECT_EQ(j["inherits_group"][i], "");
    }
}

// ===========================================================================
// Group 4: Error / formatting guardrails
// ===========================================================================

TEST(Patches, UnknownFieldFiltered) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    // Inject a patch with an unknown scalar key. BuildProjectSettings should
    // accept it (treat as scalar passthrough) without throwing - regression
    // protection against future strict-mode regressions.
    auto patches = std::make_shared<ChromaPrintPatches>(*preset.machine.patches);
    patches->process_common["chromaprint3d_unknown_test_field"] = "\"sentinel\"";
    preset.machine.patches = patches;
    auto j = ExportAndExtractProjectSettings(preset);
    EXPECT_TRUE(j.contains("chromaprint3d_unknown_test_field"));
    EXPECT_EQ(j["chromaprint3d_unknown_test_field"], "sentinel");
}

TEST(CompatiblePrinters, P2sN02ElementFormatStrict) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto spec = catalog->Resolve("Bambu Lab P2S", NozzleSize::N02, 0.08f);
    ASSERT_TRUE(spec.has_value());
    ASSERT_FALSE(spec->compatible_printers.empty());
    // Every element MUST end in " 0.2 nozzle" (not 0.4) and start with "Bambu Lab ".
    for (const auto& s : spec->compatible_printers) {
        EXPECT_TRUE(s.rfind("Bambu Lab ", 0) == 0) << "Bad prefix: " << s;
        const std::string suffix = " 0.2 nozzle";
        EXPECT_TRUE(s.size() >= suffix.size() &&
                    std::equal(suffix.rbegin(), suffix.rend(), s.rbegin()))
            << "Bad suffix: " << s;
    }
}

TEST(PrinterModel, P2sMetadataPlate) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset                  = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S",
                                            NozzleSize::N04, 3);
    std::vector<Mesh> meshes(3, MakeBoxMesh());
    std::vector<Channel> palette = {{"Red", "PLA"}, {"Green", "PLA"}, {"Blue", "PLA"}};
    auto buf = Export3mfFromMeshes(meshes, palette, -1, 0, preset);
    auto entries = ParseZipEntries(buf);
    auto* ent    = FindEntry(entries, "Metadata/model_settings.config");
    ASSERT_NE(ent, nullptr);
    auto xml = EntryAsString(ent);
    // BuildModelSettings emits printer_model_id metadata when printer_model is set.
    EXPECT_NE(xml.find("printer_model_id"), std::string::npos);
    // P2S printer_model = "Bambu Lab P2S" (from base file _chromaprint3d_meta).
    EXPECT_NE(xml.find("Bambu Lab P2S"), std::string::npos);
}

TEST(ExpandFilament, MismatchedLengthThrows) {
    // Synthesize a base_dict with a deliberately-bad length for a
    // filament-with-variant key, and verify ExpandFilamentWithVariantToN throws
    // (silent skip path removed; mismatched length now throws).
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    // Use H2D so K_process=5; BUT pass a base file whose
    // filament_max_volumetric_speed is already aligned to 5. The on-disk file
    // is correct, so we cannot easily induce a mismatch through the public
    // API without a custom base. Instead exercise the throw path via a unit
    // test on the BuildProjectSettings flow with a P2S preset whose JSON has
    // been mutated. We use a temporary file approach: copy the base, edit
    // nozzle_temperature to a wrong length, point the preset at it.
    auto spec = catalog->Resolve("Bambu Lab P2S", NozzleSize::N04, 0.08f);
    ASSERT_TRUE(spec.has_value());

    auto tmp_path = std::filesystem::temp_directory_path() / "chromaprint3d_bad_base.json";
    {
        std::ifstream ifs(spec->preset_base_path);
        nlohmann::json bd = nlohmann::json::parse(ifs);
        // Force nozzle_temperature to length 3 (P2S K_process = 2; 3 != 2 and 3 != N×2).
        bd["nozzle_temperature"] = nlohmann::json::array({"210", "220", "230"});
        std::ofstream ofs(tmp_path);
        ofs << bd.dump();
    }

    SlicerPreset bad = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 4);
    bad.machine.preset_base_path = tmp_path;

    std::vector<Mesh> meshes(4, MakeBoxMesh());
    std::vector<Channel> palette(4, Channel{"Slot", "PLA", "#FFFFFF"});
    EXPECT_THROW(Export3mfFromMeshes(meshes, palette, -1, 0, bad), std::exception);

    std::filesystem::remove(tmp_path);
}

// ===========================================================================
// Group: FlushMatrixGeneration
//
// Verifies that BuildProjectSettings emits a fully-populated
// `flush_volumes_matrix` (and supporting fields) so BambuStudio does NOT
// fall back to the degenerate `8×8=280` default that causes color bleed.
// ===========================================================================

TEST(FlushMatrixGeneration, P2sN3MatrixSizeAndDiagonalZero) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j = ExportAndExtractProjectSettings(preset);

    // P2S = 1 physical nozzle → 3*3*1 = 9 entries.
    ASSERT_TRUE(j.contains("flush_volumes_matrix"));
    ASSERT_TRUE(j["flush_volumes_matrix"].is_array());
    const std::size_t nozzle_count = j["flush_multiplier"].size();
    EXPECT_EQ(nozzle_count, 1u);
    EXPECT_EQ(j["flush_volumes_matrix"].size(), 3u * 3u * nozzle_count);

    // Diagonal entries = 0 (same-color switch).
    const auto& matrix = j["flush_volumes_matrix"];
    constexpr int N = 3;
    for (std::size_t n = 0; n < nozzle_count; ++n) {
        for (int i = 0; i < N; ++i) {
            std::size_t idx = n * N * N + i * N + i;
            EXPECT_EQ(matrix[idx].get<std::string>(), "0")
                << "diagonal at nozzle=" << n << " filament=" << i;
        }
    }
}

// Helper: derive "machine facts" from the exported project_settings.config
// rather than hardcoding nozzle counts in tests. This keeps tests valid as
// catalog/base files evolve.
struct MachineFacts {
    std::size_t nozzle_count;       // = flush_multiplier.size()
    std::size_t filament_count;     // = N (preset.filaments.size())
    std::size_t expected_matrix_sz; // = N * N * nozzle_count
};
MachineFacts ExtractMachineFacts(const nlohmann::json& project_settings,
                                  std::size_t N) {
    const std::size_t nozzle_count = project_settings["flush_multiplier"].size();
    return {nozzle_count, N, N * N * nozzle_count};
}

TEST(FlushMatrixGeneration, H2cN5MatrixShapeFromFacts) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    constexpr std::size_t N = 5;
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2C", NozzleSize::N04, N);
    auto j = ExportAndExtractProjectSettings(preset);
    auto facts = ExtractMachineFacts(j, N);

    EXPECT_EQ(j["flush_volumes_matrix"].size(), facts.expected_matrix_sz);

    // Diagonal entries on each nozzle slab must be zero.
    const auto& matrix = j["flush_volumes_matrix"];
    for (std::size_t n = 0; n < facts.nozzle_count; ++n) {
        for (std::size_t i = 0; i < N; ++i) {
            std::size_t idx = n * N * N + i * N + i;
            EXPECT_EQ(matrix[idx].get<std::string>(), "0")
                << "nozzle=" << n << " i=" << i;
        }
    }
}

TEST(FlushMatrixGeneration, FlushVolumesVectorAndMultiplier) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    constexpr std::size_t N = 5;
    // H2C exercises nozzle_count > 1 in the multiplier.
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2C", NozzleSize::N04, N);
    auto j = ExportAndExtractProjectSettings(preset);
    auto facts = ExtractMachineFacts(j, N);

    ASSERT_TRUE(j["flush_volumes_vector"].is_array());
    EXPECT_EQ(j["flush_volumes_vector"].size(), 2u * N); // 2 × N (push/pull)
    for (const auto& v : j["flush_volumes_vector"]) {
        EXPECT_EQ(v.get<std::string>(), "140");
    }

    ASSERT_TRUE(j["flush_multiplier"].is_array());
    EXPECT_EQ(j["flush_multiplier"].size(), facts.nozzle_count);
    for (const auto& v : j["flush_multiplier"]) {
        EXPECT_EQ(v.get<std::string>(), "1");
    }
}

TEST(FlushMatrixGeneration, H2cN8MatrixShapeFromFacts) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    constexpr std::size_t N = 8; // typical heavy multi-color project
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2C", NozzleSize::N04, N);
    auto j = ExportAndExtractProjectSettings(preset);
    auto facts = ExtractMachineFacts(j, N);
    EXPECT_EQ(j["flush_volumes_matrix"].size(), facts.expected_matrix_sz);
}

TEST(FlushMatrixGeneration, SideChannelDefaults) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, 3);
    auto j = ExportAndExtractProjectSettings(preset);

    // Mirrors varesa baseline.
    EXPECT_EQ(j.value("flush_into_objects",   ""), "0");
    EXPECT_EQ(j.value("flush_into_infill",    ""), "0");
    EXPECT_EQ(j.value("flush_into_support",   ""), "1");
    EXPECT_EQ(j.value("prime_volume_mode",    ""), "Default");
    EXPECT_EQ(j.value("role_base_wipe_speed", ""), "1");
}

TEST(FlushMatrixGeneration, UserOverrideTakesPriority) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    constexpr std::size_t N = 2;
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab P2S", NozzleSize::N04, N);
    auto baseline = ExportAndExtractProjectSettings(preset);
    auto facts    = ExtractMachineFacts(baseline, N);
    // Manually supply a flat override array sized to whatever the catalog
    // says this machine produces.
    std::vector<int> override_matrix(facts.expected_matrix_sz, 999);
    for (std::size_t n = 0; n < facts.nozzle_count; ++n) {
        for (std::size_t i = 0; i < N; ++i) {
            override_matrix[n * N * N + i * N + i] = 0;
        }
    }
    preset.flush_volumes_matrix = override_matrix;
    auto j = ExportAndExtractProjectSettings(preset);

    ASSERT_EQ(j["flush_volumes_matrix"].size(), facts.expected_matrix_sz);
    // First off-diagonal entry should reflect the user override (999), not
    // the HSV-formula auto-computed value.
    const std::size_t off_diag_idx = 1; // (i=0, j=1) in nozzle 0
    EXPECT_EQ(j["flush_volumes_matrix"][off_diag_idx].get<std::string>(), "999")
        << "user-supplied override must win over auto-generated values";
}

TEST(FlushMatrixGeneration, NotAll280Sentinel) {
    // Regression for the bug fixed by this change: previously the matrix
    // was empty on export → BBS filled it with the 280-fallback (luminance
    // formula yields different non-280 values for any non-trivial palette).
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2C", NozzleSize::N04, 4);
    // Override colors to a high-contrast quad.
    const char* colors[] = {"#000000", "#FFFFFF", "#C12E1F", "#0086D6"};
    for (size_t i = 0; i < preset.filaments.size() && i < 4; ++i) {
        preset.filaments[i].colour = colors[i];
    }
    auto j = ExportAndExtractProjectSettings(preset);

    int count_280 = 0;
    for (const auto& v : j["flush_volumes_matrix"]) {
        if (v.get<std::string>() == "280") ++count_280;
    }
    // BBS fallback would yield 280 on every off-diagonal entry (24 of 32);
    // our generator should produce far fewer (typically zero) such entries
    // because the HSV formula and dataset return different values per pair.
    EXPECT_LT(count_280, 5)
        << "matrix has too many 280 entries — looks like BBS fallback";
}

// Replaces the dataset-specific test that was deleted with the
// FlushVolPredictor removal: now we just verify the matrix entry for a
// high-contrast pair lands somewhere in the expected ballpark of the HSV
// formula (≈ 560 mm³ for black→white before any min-flush addition).
TEST(FlushMatrixGeneration, BlackWhitePairUsesHsvFormula) {
    auto catalog = LoadCatalogOrNull();
    if (!catalog) GTEST_SKIP() << "BambuPresetCatalog not available";
    auto preset = MakeTestPresetForMachine(*catalog, "Bambu Lab H2C", NozzleSize::N04, 2);
    preset.filaments[0].colour = "#000000";
    preset.filaments[1].colour = "#FFFFFF";
    auto j = ExportAndExtractProjectSettings(preset);
    // Layout for N=2, nozzle 0: [0]=black→black=0, [1]=black→white, ...
    // HSV formula: ~560 mm³ + min_flush_volume (= nozzle_volume[0] = 130).
    int v = std::stoi(j["flush_volumes_matrix"][1].get<std::string>());
    EXPECT_GE(v, 560);   // floor: HSV formula alone
    EXPECT_LE(v, 900);   // clamp: kMaxFlushVolume
}
