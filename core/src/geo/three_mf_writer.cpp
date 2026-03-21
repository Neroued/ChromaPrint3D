#include "three_mf_writer.h"

#include "chromaprint3d/error.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <system_error>
#include <utility>

namespace ChromaPrint3D::detail {

namespace {

constexpr const char* kDefaultApplication = "ChromaPrint3D";

bool HasMetadataKey(const std::vector<ThreeMfMetadataEntry>& metadata, const std::string& key) {
    return std::any_of(metadata.begin(), metadata.end(),
                       [&](const ThreeMfMetadataEntry& entry) { return entry.name == key; });
}

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

std::filesystem::path MakeAtomicTempPath(const std::filesystem::path& final_path) {
    const auto dir           = final_path.parent_path();
    const auto filename_base = final_path.filename().string();
    for (int attempt = 0; attempt < 16; ++attempt) {
        auto candidate = dir / (filename_base + ".tmp-" + RandomHex(8));
        std::error_code ec;
        if (!std::filesystem::exists(candidate, ec) && !ec) { return candidate; }
    }
    throw IOError("Failed to allocate temporary output path for " + final_path.string());
}

class PendingAtomicFile {
public:
    explicit PendingAtomicFile(std::filesystem::path path) : path_(std::move(path)) {}

    ~PendingAtomicFile() { Cleanup(); }

    PendingAtomicFile(const PendingAtomicFile&)            = delete;
    PendingAtomicFile& operator=(const PendingAtomicFile&) = delete;

    const std::filesystem::path& path() const { return path_; }

    void CommitTo(const std::filesystem::path& final_path) {
        std::error_code ec;
#if defined(_WIN32)
        if (std::filesystem::exists(final_path, ec) && !ec) {
            std::filesystem::remove(final_path, ec);
            if (ec) {
                throw IOError("Failed to replace existing output file " + final_path.string() +
                              ": " + ec.message());
            }
        }
#endif
        std::filesystem::rename(path_, final_path, ec);
        if (ec) {
            throw IOError("Failed to move temp output file into place for " + final_path.string() +
                          ": " + ec.message());
        }
        path_.clear();
    }

private:
    void Cleanup() noexcept {
        if (path_.empty()) return;
        std::error_code ec;
        std::filesystem::remove(path_, ec);
        path_.clear();
    }

    std::filesystem::path path_;
};

} // namespace

bool ThreeMfTransform::IsIdentity(float eps) const {
    const auto& m                        = values;
    const std::array<float, 12> identity = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    for (std::size_t i = 0; i < m.size(); ++i) {
        if (std::abs(m[i] - identity[i]) > eps) { return false; }
    }
    return true;
}

ThreeMfWriter::ThreeMfWriter(ThreeMfExportOptions options) : options_(std::move(options)) {
    if (!HasMetadataKey(options_.metadata, "Application")) {
        options_.metadata.push_back({"Application", kDefaultApplication});
    }
    RegisterExtension(MakeUnitAndMetadataExtension());
    RegisterExtension(MakeBaseMaterialExtension());
    RegisterExtension(MakeCoreModelExtension());
    RegisterExtension(MakeBuildTransformExtension());
}

void ThreeMfWriter::RegisterExtension(std::unique_ptr<IThreeMfExtension> extension) {
    if (!extension) { return; }
    extensions_.push_back(std::move(extension));
    std::stable_sort(
        extensions_.begin(), extensions_.end(),
        [](const std::unique_ptr<IThreeMfExtension>& a,
           const std::unique_ptr<IThreeMfExtension>& b) { return a->Priority() < b->Priority(); });
}

ThreeMfDocument ThreeMfWriter::BuildDocument(const std::vector<ThreeMfInputObject>& objects) const {
    if (objects.empty()) { throw InputError("No objects to export"); }

    ThreeMfDocument document;
    for (const auto& extension : extensions_) { extension->Apply(document, objects, options_); }

    bool has_geometry = !document.mesh_resources.empty() ||
                        document.assembly_object_id.has_value() ||
                        document.deferred_mesh_part.has_value();
    if (!has_geometry || document.build_items.empty()) {
        throw InputError("No geometry to export");
    }
    return document;
}

namespace {

void WriteAllEntries(StreamingZipWriter& zip, const OpcPartSet& part_set,
                     const ThreeMfDocument& document) {
    for (const auto& part : part_set.parts) { zip.WriteWholeEntry(part.path_in_zip, part.data); }

    if (part_set.deferred) {
        auto format            = part_set.deferred->path_in_zip.empty() ? MeshXmlFormat::FlatModel
                                                                        : MeshXmlFormat::ObjectsModel;
        std::string entry_path = part_set.deferred->path_in_zip.empty()
                                     ? std::string("3D/3dmodel.model")
                                     : part_set.deferred->path_in_zip;
        zip.BeginDeflateEntry(entry_path);
        StreamMeshXml(part_set.deferred->resources, format, document,
                      [&](std::string_view chunk) { zip.WriteChunk(chunk.data(), chunk.size()); });
        zip.EndEntry();
    }

    zip.Finalize();
}

} // namespace

std::vector<uint8_t>
ThreeMfWriter::WriteToBuffer(const std::vector<ThreeMfInputObject>& objects) const {
    ThreeMfDocument document = BuildDocument(objects);
    OpcPartSet part_set      = BuildOpcPartSet(document);

    std::vector<uint8_t> zip_bytes;
    StreamingZipWriter zip(zip_bytes, options_);
    WriteAllEntries(zip, part_set, document);

    constexpr std::size_t kMinZipSize = 22;
    if (zip_bytes.size() < kMinZipSize) {
        spdlog::error("WriteToBuffer: generated buffer too small ({} bytes), expected >= {}",
                      zip_bytes.size(), kMinZipSize);
        throw IOError("Generated 3MF buffer is too small (" + std::to_string(zip_bytes.size()) +
                      " bytes), likely corrupt");
    }
    if (zip_bytes.size() >= 4 && !(zip_bytes[0] == 0x50 && zip_bytes[1] == 0x4B &&
                                   zip_bytes[2] == 0x03 && zip_bytes[3] == 0x04)) {
        spdlog::error("WriteToBuffer: generated buffer missing ZIP magic bytes (PK\\x03\\x04)");
        throw IOError("Generated 3MF buffer has invalid ZIP header, likely corrupt");
    }

    return zip_bytes;
}

void WriteFileAtomically(const std::filesystem::path& final_path,
                         const std::function<void(const std::filesystem::path&)>& writer) {
    if (final_path.empty()) { throw InputError("output path is empty"); }

    spdlog::debug("WriteFileAtomically: target={}", final_path.string());

    const auto dir = final_path.parent_path();
    if (!dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            spdlog::error("WriteFileAtomically: failed to create directory {}: {}", dir.string(),
                          ec.message());
            throw IOError("Failed to create output directory " + dir.string() + ": " +
                          ec.message());
        }
    }

    PendingAtomicFile pending(MakeAtomicTempPath(final_path));
    writer(pending.path());
    pending.CommitTo(final_path);
}

void WriteBufferToFileAtomically(const std::filesystem::path& final_path,
                                 const std::vector<uint8_t>& bytes) {
    spdlog::info("WriteBufferToFileAtomically: path={}, bytes={}", final_path.string(),
                 bytes.size());

    WriteFileAtomically(final_path, [&](const std::filesystem::path& temp_path) {
        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            spdlog::error("WriteBufferToFileAtomically: cannot open temp file {}",
                          temp_path.string());
            throw IOError("Cannot open file for writing: " + temp_path.string());
        }
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        if (!out.good()) {
            spdlog::error("WriteBufferToFileAtomically: write failed for {}", final_path.string());
            throw IOError("Cannot write to " + final_path.string());
        }
        out.close();
        if (out.fail()) {
            spdlog::error("WriteBufferToFileAtomically: close/flush failed for {}",
                          temp_path.string());
            throw IOError("Failed to flush/close output file: " + temp_path.string());
        }

        std::error_code ec;
        auto actual_size = std::filesystem::file_size(temp_path, ec);
        if (ec) {
            spdlog::error("WriteBufferToFileAtomically: cannot stat temp file {}: {}",
                          temp_path.string(), ec.message());
            throw IOError("Cannot verify temp file size: " + temp_path.string() + ": " +
                          ec.message());
        }
        if (actual_size != bytes.size()) {
            spdlog::error(
                "WriteBufferToFileAtomically: size mismatch for {}: expected={}, actual={}",
                temp_path.string(), bytes.size(), actual_size);
            throw IOError("File size mismatch after write: expected " +
                          std::to_string(bytes.size()) + " but got " + std::to_string(actual_size));
        }
    });

    spdlog::info("WriteBufferToFileAtomically: written {} bytes to {}", bytes.size(),
                 final_path.string());
}

void ThreeMfWriter::WriteToFile(const std::string& path,
                                const std::vector<ThreeMfInputObject>& objects) const {
    if (path.empty()) { throw InputError("3MF output path is empty"); }

    ThreeMfDocument document = BuildDocument(objects);
    OpcPartSet part_set      = BuildOpcPartSet(document);
    WriteFileAtomically(path, [&](const std::filesystem::path& temp_path) {
        StreamingZipWriter zip(temp_path.string(), options_);
        WriteAllEntries(zip, part_set, document);
    });
}

} // namespace ChromaPrint3D::detail
