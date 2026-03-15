#include <gtest/gtest.h>

#include "chromaprint3d/error.h"
#include "chromaprint3d/voxel.h"
#include "geo/three_mf_writer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(CHROMAPRINT3D_HAS_ZLIB)
#    include <zlib.h>
#endif

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
    uint16_t compression_method = 0;
    std::vector<uint8_t> raw_data;
    std::vector<uint8_t> data; // uncompressed payload
};

std::vector<uint8_t> InflateRawDeflate(const std::vector<uint8_t>& compressed,
                                       std::size_t expected_size) {
#if defined(CHROMAPRINT3D_HAS_ZLIB)
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

        uint16_t compression       = ReadU16(bytes, pos + 8);
        uint32_t compressed_size   = ReadU32(bytes, pos + 18);
        uint32_t uncompressed_size = ReadU32(bytes, pos + 22);
        uint16_t name_len          = ReadU16(bytes, pos + 26);
        uint16_t extra_len         = ReadU16(bytes, pos + 28);

        std::size_t name_off = pos + 30;
        std::size_t data_off =
            name_off + static_cast<std::size_t>(name_len) + static_cast<std::size_t>(extra_len);
        std::size_t data_end = data_off + static_cast<std::size_t>(compressed_size);
        if (data_end > bytes.size()) { throw std::runtime_error("Corrupted ZIP entry payload"); }
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

class TestOpcContributionExtension final : public detail::IThreeMfExtension {
public:
    int Priority() const override { return 90; }

    void Apply(detail::ThreeMfDocument& document,
               const std::vector<detail::ThreeMfInputObject>& /*objects*/,
               const detail::ThreeMfExportOptions& /*options*/) const override {
        document.extra_parts.push_back(detail::OpcPart{
            .path_in_zip  = "Metadata/test.config",
            .content_type = "application/octet-stream",
            .data         = "{\"k\":\"v\"}",
        });
        document.relationship_sets.push_back(detail::OpcRelationshipSet{
            .source_part_path = "3D/3dmodel.model",
            .relationships =
                {
                    detail::OpcRelationship{
                        .id     = "extRel0",
                        .type   = "http://example.com/settings",
                        .target = "/Metadata/test.config",
                    },
                },
        });
        document.content_type_defaults.push_back(detail::OpcContentTypeDefault{
            .extension    = "config",
            .content_type = "application/octet-stream",
        });
    }
};

} // namespace

TEST(ThreeMfWriter, WritesRequiredOpcParts) {
    Mesh mesh = MakeBoxMesh();
    detail::ThreeMfInputObject object;
    object.name              = "Object A";
    object.display_color_hex = "#00AE42";
    object.mesh              = &mesh;

    detail::ThreeMfWriter writer;
    std::vector<uint8_t> buffer = writer.WriteToBuffer({object});
    ASSERT_FALSE(buffer.empty());

    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    EXPECT_NE(FindEntry(entries, "[Content_Types].xml"), nullptr);
    EXPECT_NE(FindEntry(entries, "_rels/.rels"), nullptr);
    EXPECT_NE(FindEntry(entries, "3D/3dmodel.model"), nullptr);
}

TEST(ThreeMfWriter, DefaultPackageKeepsMinimalOpcLayout) {
    Mesh mesh = MakeBoxMesh();
    detail::ThreeMfInputObject object;
    object.name              = "Object A";
    object.display_color_hex = "#00AE42";
    object.mesh              = &mesh;

    detail::ThreeMfWriter writer;
    std::vector<uint8_t> buffer = writer.WriteToBuffer({object});
    ASSERT_FALSE(buffer.empty());

    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    EXPECT_EQ(FindEntry(entries, "3D/_rels/3dmodel.model.rels"), nullptr);
    EXPECT_EQ(FindEntry(entries, "Metadata/test.config"), nullptr);
}

TEST(ThreeMfWriter, MergesDynamicOpcContributionIntoPackage) {
    Mesh mesh = MakeBoxMesh();
    detail::ThreeMfInputObject object;
    object.name              = "Object A";
    object.display_color_hex = "#00AE42";
    object.mesh              = &mesh;

    detail::ThreeMfWriter writer;
    writer.RegisterExtension(std::make_unique<TestOpcContributionExtension>());
    std::vector<uint8_t> buffer = writer.WriteToBuffer({object});
    ASSERT_FALSE(buffer.empty());

    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    EXPECT_NE(FindEntry(entries, "Metadata/test.config"), nullptr);
    std::string rels_xml = EntryAsString(FindEntry(entries, "3D/_rels/3dmodel.model.rels"));
    EXPECT_NE(rels_xml.find("http://example.com/settings"), std::string::npos);
    EXPECT_NE(rels_xml.find("/Metadata/test.config"), std::string::npos);

    std::string content_types = EntryAsString(FindEntry(entries, "[Content_Types].xml"));
    EXPECT_NE(content_types.find("Extension=\"config\""), std::string::npos);
    EXPECT_NE(content_types.find("PartName=\"/Metadata/test.config\""), std::string::npos);
}

TEST(ThreeMfWriter, SerializesUnitAndMetadata) {
    detail::ThreeMfExportOptions options;
    options.unit = detail::ThreeMfUnit::Inch;
    options.metadata.push_back({"Title", "Writer V1"});
    options.metadata.push_back({"Description", "unit+metadata"});

    Mesh mesh = MakeBoxMesh();
    detail::ThreeMfInputObject object;
    object.name              = "Object A";
    object.display_color_hex = "#FFFFFF";
    object.mesh              = &mesh;

    detail::ThreeMfWriter writer(options);
    std::vector<uint8_t> buffer = writer.WriteToBuffer({object});
    ASSERT_FALSE(buffer.empty());

    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    std::string model_xml         = EntryAsString(FindEntry(entries, "3D/3dmodel.model"));
    EXPECT_NE(model_xml.find("unit=\"inch\""), std::string::npos);
    EXPECT_NE(model_xml.find("<metadata name=\"Title\">Writer V1</metadata>"), std::string::npos);
    EXPECT_NE(model_xml.find("<metadata name=\"Description\">unit+metadata</metadata>"),
              std::string::npos);
}

TEST(ThreeMfWriter, SerializesBuildItemTransform) {
    Mesh mesh = MakeBoxMesh();
    detail::ThreeMfInputObject object;
    object.name              = "Object A";
    object.display_color_hex = "#C12E1F";
    object.mesh              = &mesh;
    object.transform.values  = {1.0f, 0.0f,  0.0f, 10.0f, 0.0f, 1.0f,
                                0.0f, 20.0f, 0.0f, 0.0f,  1.0f, 30.0f};

    detail::ThreeMfWriter writer;
    std::vector<uint8_t> buffer = writer.WriteToBuffer({object});
    ASSERT_FALSE(buffer.empty());

    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    std::string model_xml         = EntryAsString(FindEntry(entries, "3D/3dmodel.model"));
    EXPECT_NE(model_xml.find("transform=\"1 0 0 10 0 1 0 20 0 0 1 30\""), std::string::npos);
}

TEST(ThreeMfWriter, RejectsNonFiniteTransform) {
    Mesh mesh = MakeBoxMesh();
    detail::ThreeMfInputObject object;
    object.name                = "Object A";
    object.display_color_hex   = "#C12E1F";
    object.mesh                = &mesh;
    object.transform.values[0] = std::nanf("");

    detail::ThreeMfWriter writer;
    EXPECT_THROW(writer.WriteToBuffer({object}), InputError);
}

TEST(ThreeMfWriter, AutoThresholdUsesDeflateForLargePartWhenAvailable) {
    detail::ThreeMfExportOptions options;
    options.compression_mode            = detail::ThreeMfCompressionMode::AutoThreshold;
    options.compression_threshold_bytes = 1;
    options.deflate_level               = 1;

    Mesh mesh = MakeDenseMesh(20, 3);
    detail::ThreeMfInputObject object;
    object.name              = "Dense";
    object.display_color_hex = "#FFFFFF";
    object.mesh              = &mesh;

    detail::ThreeMfWriter writer(options);
    std::vector<uint8_t> buffer = writer.WriteToBuffer({object});
    ASSERT_FALSE(buffer.empty());
    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    const ZipEntry* model_entry   = FindEntry(entries, "3D/3dmodel.model");
    ASSERT_NE(model_entry, nullptr);

#if defined(CHROMAPRINT3D_3MF_FORCE_STORE)
    EXPECT_EQ(model_entry->compression_method, 0);
#elif defined(CHROMAPRINT3D_HAS_ZLIB)
    EXPECT_EQ(model_entry->compression_method, 8);
#else
    EXPECT_EQ(model_entry->compression_method, 0);
#endif
}

TEST(ThreeMfWriter, WriteToFileMatchesWriteToBuffer) {
    detail::ThreeMfExportOptions options;
    options.deterministic = true;

    Mesh mesh = MakeDenseMesh(10, 2);
    detail::ThreeMfInputObject object;
    object.name              = "FileTest";
    object.display_color_hex = "#FF0000";
    object.mesh              = &mesh;

    detail::ThreeMfWriter writer(options);
    std::vector<uint8_t> buffer = writer.WriteToBuffer({object});
    ASSERT_FALSE(buffer.empty());

    const auto temp_dir = CreateUniqueTempDir();
    const auto tmp_path = temp_dir / "writer-output.3mf";
    writer.WriteToFile(tmp_path.string(), {object});

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

    try {
        detail::WriteFileAtomically(final_path, [&](const std::filesystem::path& temp_path) {
            std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) {
                throw IOError("Cannot open file for writing: " + temp_path.string());
            }
            out.write("partial", 7);
            out.flush();
            throw IOError("simulated write failure");
        });
        FAIL() << "expected atomic write helper to propagate writer failure";
    } catch (const IOError& e) { EXPECT_EQ(std::string(e.what()), "simulated write failure"); }

    EXPECT_FALSE(std::filesystem::exists(final_path));
    EXPECT_TRUE(IsDirectoryEmpty(temp_dir));

    std::error_code ec;
    std::filesystem::remove_all(temp_dir, ec);
}

TEST(ThreeMfWriter, DeflateModeFallsBackToStoreOnCompressionFailure) {
    detail::ThreeMfExportOptions options;
    options.compression_mode = detail::ThreeMfCompressionMode::Deflate;
    options.deflate_level    = 99; // invalid zlib level, should trigger safe fallback

    Mesh mesh = MakeBoxMesh();
    detail::ThreeMfInputObject object;
    object.name              = "Object A";
    object.display_color_hex = "#FFFFFF";
    object.mesh              = &mesh;

    detail::ThreeMfWriter writer(options);
    std::vector<uint8_t> buffer = writer.WriteToBuffer({object});
    ASSERT_FALSE(buffer.empty());
    std::vector<ZipEntry> entries = ParseZipEntries(buffer);
    ASSERT_FALSE(entries.empty());
    EXPECT_TRUE(std::all_of(entries.begin(), entries.end(),
                            [](const ZipEntry& entry) { return entry.compression_method == 0; }));
}
