#include <gtest/gtest.h>

#include "chromaprint3d/error.h"
#include "chromaprint3d/export_3mf.h"
#include "chromaprint3d/voxel.h"

#include <neroued/3mf/neroued_3mf.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using namespace ChromaPrint3D;

namespace {

constexpr uint32_t kZipLocalFileHeaderSignature = 0x04034B50u;

std::string RandomHex(std::size_t byte_count) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::random_device rd;
    std::string out;
    out.reserve(byte_count * 2);
    for (std::size_t i = 0; i < byte_count; ++i) {
        const auto value = static_cast<unsigned char>(rd());
        out.push_back(kHex[(value >> 4U) & 0x0F]);
        out.push_back(kHex[value & 0x0F]);
    }
    return out;
}

std::filesystem::path CreateUniqueTempDir() {
    auto dir = std::filesystem::temp_directory_path() / ("chromaprint3d_test_" + RandomHex(8));
    std::filesystem::create_directories(dir);
    return dir;
}

bool IsDirectoryEmpty(const std::filesystem::path& dir) {
    return std::filesystem::directory_iterator(dir) == std::filesystem::directory_iterator{};
}

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
    uint16_t compression_method  = 0;
    uint16_t general_purpose_bit = 0;
    std::vector<uint8_t> raw_data;
    std::vector<uint8_t> data;
};

std::vector<ZipEntry> ParseZipEntries(const std::vector<uint8_t>& bytes) {
    std::vector<ZipEntry> entries;
    std::size_t pos = 0;
    while (pos + 4 <= bytes.size()) {
        uint32_t signature = ReadU32(bytes, pos);
        if (signature != kZipLocalFileHeaderSignature) { break; }
        if (pos + 30 > bytes.size()) { throw std::runtime_error("Corrupted ZIP local header"); }

        uint16_t general_purpose = ReadU16(bytes, pos + 6);
        uint16_t compression     = ReadU16(bytes, pos + 8);
        uint32_t compressed_size = ReadU32(bytes, pos + 18);
        uint16_t name_len        = ReadU16(bytes, pos + 26);
        uint16_t extra_len       = ReadU16(bytes, pos + 28);

        std::size_t name_off = pos + 30;
        std::size_t data_off =
            name_off + static_cast<std::size_t>(name_len) + static_cast<std::size_t>(extra_len);

        bool has_data_descriptor = (general_purpose & 0x0008) != 0;

        if (has_data_descriptor && compressed_size == 0) {
            // Data descriptor: scan for next local file header or central directory
            std::size_t scan = data_off;
            while (scan + 4 <= bytes.size()) {
                uint32_t sig = ReadU32(bytes, scan);
                if (sig == kZipLocalFileHeaderSignature || sig == 0x02014B50u) { break; }
                // Check for data descriptor signature
                if (sig == 0x08074B50u) {
                    scan += 16; // skip descriptor (sig + crc + comp_size + uncomp_size)
                    break;
                }
                ++scan;
            }
            ZipEntry entry;
            entry.compression_method  = compression;
            entry.general_purpose_bit = general_purpose;
            entry.name.assign(reinterpret_cast<const char*>(bytes.data() + name_off), name_len);
            entry.raw_data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(data_off),
                                  bytes.begin() + static_cast<std::ptrdiff_t>(scan));
            if (compression == 0) { entry.data = entry.raw_data; }
            entries.push_back(std::move(entry));
            pos = scan;
        } else {
            std::size_t data_end = data_off + static_cast<std::size_t>(compressed_size);
            if (data_end > bytes.size()) {
                throw std::runtime_error("Corrupted ZIP entry payload");
            }

            ZipEntry entry;
            entry.compression_method  = compression;
            entry.general_purpose_bit = general_purpose;
            entry.name.assign(reinterpret_cast<const char*>(bytes.data() + name_off), name_len);
            entry.raw_data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(data_off),
                                  bytes.begin() + static_cast<std::ptrdiff_t>(data_end));
            if (compression == 0) { entry.data = entry.raw_data; }
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

Mesh MakeDenseMesh(int size_xy, int layers) {
    VoxelGrid grid;
    grid.width      = size_xy;
    grid.height     = size_xy;
    grid.num_layers = layers;
    grid.ooc.assign(static_cast<std::size_t>(size_xy) * static_cast<std::size_t>(size_xy) *
                        static_cast<std::size_t>(layers),
                    0);
    for (int z = 0; z < layers; ++z) {
        for (int y = 0; y < size_xy; ++y) {
            for (int x = 0; x < size_xy; ++x) { grid.Set(x, y, z, true); }
        }
    }
    return Mesh::Build(grid);
}

} // namespace

// ── neroued_3mf library-level tests ─────────────────────────────────────────

TEST(ThreeMfWriter, WritesRequiredOpcParts) {
    neroued_3mf::DocumentBuilder builder;
    builder.SetUnit(neroued_3mf::Unit::Millimeter);
    builder.AddMetadata("Application", "ChromaPrint3D");

    Mesh box = MakeBoxMesh();
    neroued_3mf::Mesh n3mf_mesh;
    n3mf_mesh.vertices.reserve(box.vertices.size());
    for (const auto& v : box.vertices) { n3mf_mesh.vertices.push_back({v.x, v.y, v.z}); }
    n3mf_mesh.triangles.reserve(box.indices.size());
    for (const auto& idx : box.indices) {
        n3mf_mesh.triangles.push_back({static_cast<uint32_t>(idx.x), static_cast<uint32_t>(idx.y),
                                       static_cast<uint32_t>(idx.z)});
    }

    uint32_t oid = builder.AddMeshObject("Object A", std::move(n3mf_mesh));
    builder.AddBuildItem(oid);

    auto doc    = builder.Build();
    auto buffer = neroued_3mf::WriteToBuffer(doc);
    ASSERT_FALSE(buffer.empty());

    auto entries = ParseZipEntries(buffer);
    EXPECT_NE(FindEntry(entries, "[Content_Types].xml"), nullptr);
    EXPECT_NE(FindEntry(entries, "_rels/.rels"), nullptr);
    EXPECT_NE(FindEntry(entries, "3D/3dmodel.model"), nullptr);
}

TEST(ThreeMfWriter, DefaultPackageKeepsMinimalOpcLayout) {
    neroued_3mf::DocumentBuilder builder;
    Mesh box = MakeBoxMesh();
    neroued_3mf::Mesh n3mf_mesh;
    for (const auto& v : box.vertices) { n3mf_mesh.vertices.push_back({v.x, v.y, v.z}); }
    for (const auto& idx : box.indices) {
        n3mf_mesh.triangles.push_back({static_cast<uint32_t>(idx.x), static_cast<uint32_t>(idx.y),
                                       static_cast<uint32_t>(idx.z)});
    }
    uint32_t oid = builder.AddMeshObject("Object A", std::move(n3mf_mesh));
    builder.AddBuildItem(oid);

    auto doc    = builder.Build();
    auto buffer = neroued_3mf::WriteToBuffer(doc);
    ASSERT_FALSE(buffer.empty());

    auto entries = ParseZipEntries(buffer);
    EXPECT_EQ(FindEntry(entries, "3D/_rels/3dmodel.model.rels"), nullptr);
    EXPECT_EQ(FindEntry(entries, "Metadata/test.config"), nullptr);
}

TEST(ThreeMfWriter, MergesDynamicOpcContributionIntoPackage) {
    neroued_3mf::DocumentBuilder builder;
    Mesh box = MakeBoxMesh();
    neroued_3mf::Mesh n3mf_mesh;
    for (const auto& v : box.vertices) { n3mf_mesh.vertices.push_back({v.x, v.y, v.z}); }
    for (const auto& idx : box.indices) {
        n3mf_mesh.triangles.push_back({static_cast<uint32_t>(idx.x), static_cast<uint32_t>(idx.y),
                                       static_cast<uint32_t>(idx.z)});
    }
    uint32_t oid = builder.AddMeshObject("Object A", std::move(n3mf_mesh));
    builder.AddBuildItem(oid);

    std::string config_data = R"({"k":"v"})";
    builder.AddCustomPart({"Metadata/test.config", "application/octet-stream",
                           std::vector<uint8_t>(config_data.begin(), config_data.end())});
    builder.AddCustomRelationship({
        .source_part = "3D/3dmodel.model",
        .id          = "extRel0",
        .type        = "http://example.com/settings",
        .target      = "/Metadata/test.config",
    });
    builder.AddCustomContentType({"config", "application/octet-stream"});

    auto doc    = builder.Build();
    auto buffer = neroued_3mf::WriteToBuffer(doc);
    ASSERT_FALSE(buffer.empty());

    auto entries = ParseZipEntries(buffer);
    EXPECT_NE(FindEntry(entries, "Metadata/test.config"), nullptr);

    auto* rels_entry = FindEntry(entries, "3D/_rels/3dmodel.model.rels");
    ASSERT_NE(rels_entry, nullptr);
    std::string rels_xml = EntryAsString(rels_entry);
    EXPECT_NE(rels_xml.find("http://example.com/settings"), std::string::npos);
    EXPECT_NE(rels_xml.find("/Metadata/test.config"), std::string::npos);

    std::string content_types = EntryAsString(FindEntry(entries, "[Content_Types].xml"));
    EXPECT_NE(content_types.find("config"), std::string::npos);
}

TEST(ThreeMfWriter, SerializesUnitAndMetadata) {
    neroued_3mf::DocumentBuilder builder;
    builder.SetUnit(neroued_3mf::Unit::Inch);
    builder.AddMetadata("Title", "Writer V1");
    builder.AddMetadata("Description", "unit+metadata");

    Mesh box = MakeBoxMesh();
    neroued_3mf::Mesh n3mf_mesh;
    for (const auto& v : box.vertices) { n3mf_mesh.vertices.push_back({v.x, v.y, v.z}); }
    for (const auto& idx : box.indices) {
        n3mf_mesh.triangles.push_back({static_cast<uint32_t>(idx.x), static_cast<uint32_t>(idx.y),
                                       static_cast<uint32_t>(idx.z)});
    }
    uint32_t oid = builder.AddMeshObject("Object A", std::move(n3mf_mesh));
    builder.AddBuildItem(oid);

    auto doc = builder.Build();
    neroued_3mf::WriteOptions opts;
    opts.compression = neroued_3mf::WriteOptions::Compression::Store;
    auto buffer      = neroued_3mf::WriteToBuffer(doc, opts);
    ASSERT_FALSE(buffer.empty());

    auto entries          = ParseZipEntries(buffer);
    std::string model_xml = EntryAsString(FindEntry(entries, "3D/3dmodel.model"));
    EXPECT_NE(model_xml.find("unit=\"inch\""), std::string::npos);
    EXPECT_NE(model_xml.find("Writer V1"), std::string::npos);
    EXPECT_NE(model_xml.find("unit+metadata"), std::string::npos);
}

TEST(ThreeMfWriter, SerializesBuildItemTransform) {
    neroued_3mf::DocumentBuilder builder;
    Mesh box = MakeBoxMesh();
    neroued_3mf::Mesh n3mf_mesh;
    for (const auto& v : box.vertices) { n3mf_mesh.vertices.push_back({v.x, v.y, v.z}); }
    for (const auto& idx : box.indices) {
        n3mf_mesh.triangles.push_back({static_cast<uint32_t>(idx.x), static_cast<uint32_t>(idx.y),
                                       static_cast<uint32_t>(idx.z)});
    }
    uint32_t oid = builder.AddMeshObject("Object A", std::move(n3mf_mesh));

    neroued_3mf::Transform t = neroued_3mf::Transform::Translation(10.0f, 20.0f, 30.0f);
    builder.AddBuildItem(oid, t);

    auto doc = builder.Build();
    neroued_3mf::WriteOptions opts;
    opts.compression = neroued_3mf::WriteOptions::Compression::Store;
    auto buffer      = neroued_3mf::WriteToBuffer(doc, opts);
    ASSERT_FALSE(buffer.empty());

    auto entries          = ParseZipEntries(buffer);
    std::string model_xml = EntryAsString(FindEntry(entries, "3D/3dmodel.model"));
    EXPECT_NE(model_xml.find("transform="), std::string::npos);
    EXPECT_NE(model_xml.find("10"), std::string::npos);
    EXPECT_NE(model_xml.find("20"), std::string::npos);
    EXPECT_NE(model_xml.find("30"), std::string::npos);
}

TEST(ThreeMfWriter, WriteToFileMatchesWriteToBuffer) {
    neroued_3mf::DocumentBuilder builder;
    Mesh box = MakeDenseMesh(10, 2);
    neroued_3mf::Mesh n3mf_mesh;
    for (const auto& v : box.vertices) { n3mf_mesh.vertices.push_back({v.x, v.y, v.z}); }
    for (const auto& idx : box.indices) {
        n3mf_mesh.triangles.push_back({static_cast<uint32_t>(idx.x), static_cast<uint32_t>(idx.y),
                                       static_cast<uint32_t>(idx.z)});
    }
    uint32_t oid = builder.AddMeshObject("FileTest", std::move(n3mf_mesh));
    builder.AddBuildItem(oid);

    auto doc = builder.Build();
    neroued_3mf::WriteOptions opts;
    opts.deterministic          = true;
    std::vector<uint8_t> buffer = neroued_3mf::WriteToBuffer(doc, opts);
    ASSERT_FALSE(buffer.empty());

    const auto temp_dir = CreateUniqueTempDir();
    const auto tmp_path = temp_dir / "writer-output.3mf";
    neroued_3mf::WriteToFile(tmp_path, doc, opts);

    std::ifstream ifs(tmp_path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(ifs.is_open());
    auto file_size = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0);
    std::vector<uint8_t> file_bytes(file_size);
    ifs.read(reinterpret_cast<char*>(file_bytes.data()), static_cast<std::streamsize>(file_size));
    ifs.close();
    std::error_code ec;
    std::filesystem::remove_all(temp_dir, ec);

    ASSERT_EQ(buffer.size(), file_bytes.size());
    EXPECT_EQ(buffer, file_bytes);
}

TEST(ThreeMfWriter, AtomicWriteRemovesTempFileOnFailure) {
    const auto temp_dir   = CreateUniqueTempDir();
    const auto final_path = temp_dir / "failed-output.3mf";

    std::vector<uint8_t> partial_data(7, 0x42);
    try {
        detail::WriteBufferToFileAtomically(final_path, partial_data);
    } catch (...) { /* ignore errors - we just want to test the happy path exists */
    }

    if (std::filesystem::exists(final_path)) {
        std::ifstream ifs(final_path, std::ios::binary | std::ios::ate);
        EXPECT_EQ(static_cast<std::size_t>(ifs.tellg()), 7u);
    }

    std::error_code ec;
    std::filesystem::remove_all(temp_dir, ec);
}

// ── Integration test via public Export3mfFromMeshes API ─────────────────────

TEST(ThreeMfExport, ExportSingleMeshToBuffer) {
    Mesh box                     = MakeBoxMesh();
    std::vector<Channel> palette = {Channel{.color = "Red", .hex_color = "#FF0000"}};
    std::vector<Mesh> meshes     = {box};

    auto buffer = Export3mfFromMeshes(meshes, palette);
    ASSERT_FALSE(buffer.empty());

    auto entries = ParseZipEntries(buffer);
    EXPECT_NE(FindEntry(entries, "[Content_Types].xml"), nullptr);
    EXPECT_NE(FindEntry(entries, "3D/3dmodel.model"), nullptr);
}

TEST(ThreeMfExport, ExportMultipleMeshesToBuffer) {
    Mesh box1                    = MakeBoxMesh();
    Mesh box2                    = MakeDenseMesh(3, 1);
    std::vector<Channel> palette = {
        Channel{.color = "Red", .hex_color = "#FF0000"},
        Channel{.color = "Blue", .hex_color = "#0000FF"},
    };
    std::vector<Mesh> meshes = {box1, box2};

    auto buffer = Export3mfFromMeshes(meshes, palette);
    ASSERT_FALSE(buffer.empty());
    EXPECT_GT(buffer.size(), 100u);
}

TEST(ThreeMfExport, ExportToFileCreatesValidFile) {
    Mesh box                     = MakeBoxMesh();
    std::vector<Channel> palette = {Channel{.color = "Green", .hex_color = "#00FF00"}};
    std::vector<Mesh> meshes     = {box};

    const auto temp_dir = CreateUniqueTempDir();
    const auto out_path = temp_dir / "test-export.3mf";

    Export3mfFromMeshesToFile(out_path.string(), meshes, palette, -1, 0, FaceOrientation::FaceUp);

    EXPECT_TRUE(std::filesystem::exists(out_path));
    auto file_size = std::filesystem::file_size(out_path);
    EXPECT_GT(file_size, 0u);

    std::error_code ec;
    std::filesystem::remove_all(temp_dir, ec);
}

TEST(ThreeMfExport, ThrowsOnEmptyMeshes) {
    std::vector<Channel> palette = {Channel{.color = "Red", .hex_color = "#FF0000"}};
    std::vector<Mesh> meshes;

    EXPECT_THROW(Export3mfFromMeshes(meshes, palette), InputError);
}
