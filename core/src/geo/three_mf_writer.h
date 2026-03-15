#pragma once

#include "three_mf_ir.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ChromaPrint3D {
struct SlicerPreset;
}

namespace ChromaPrint3D::detail {

class IThreeMfExtension {
public:
    virtual ~IThreeMfExtension() = default;

    virtual int Priority() const = 0;

    virtual void Apply(ThreeMfDocument& document, const std::vector<ThreeMfInputObject>& objects,
                       const ThreeMfExportOptions& options) const = 0;
};

class ThreeMfWriter {
public:
    explicit ThreeMfWriter(ThreeMfExportOptions options = {});

    void RegisterExtension(std::unique_ptr<IThreeMfExtension> extension);

    std::vector<uint8_t> WriteToBuffer(const std::vector<ThreeMfInputObject>& objects) const;

    void WriteToFile(const std::string& path, const std::vector<ThreeMfInputObject>& objects) const;

private:
    ThreeMfDocument BuildDocument(const std::vector<ThreeMfInputObject>& objects) const;

    ThreeMfExportOptions options_;
    std::vector<std::unique_ptr<IThreeMfExtension>> extensions_;
};

// Extension factories (implemented in three_mf_extensions.cpp)
std::unique_ptr<IThreeMfExtension> MakeCoreModelExtension();
std::unique_ptr<IThreeMfExtension> MakeBaseMaterialExtension();
std::unique_ptr<IThreeMfExtension> MakeUnitAndMetadataExtension();
std::unique_ptr<IThreeMfExtension> MakeBuildTransformExtension();
std::unique_ptr<IThreeMfExtension> MakeBambuMetadataExtension(const SlicerPreset& preset,
                                                              std::vector<int> input_slots,
                                                              std::string model_name = {});

// OPC + serialization (implemented in three_mf_opc.cpp)
std::vector<OpcPart> BuildOpcParts(const ThreeMfDocument& document);
OpcPartSet BuildOpcPartSet(const ThreeMfDocument& document);

// Generate external objects model XML (mesh data only, Bambu namespaces, empty build).
std::string BuildObjectsModelXml(const std::vector<ThreeMfMeshResource>& resources);

// Atomically replace the target file by first writing a sibling temporary file.
void WriteFileAtomically(const std::filesystem::path& final_path,
                         const std::function<void(const std::filesystem::path&)>& writer);

// Convenience wrapper for atomically writing an in-memory buffer to disk.
void WriteBufferToFileAtomically(const std::filesystem::path& final_path,
                                 const std::vector<uint8_t>& bytes);

// Streaming mesh XML serialization (implemented in three_mf_opc.cpp).
// Writes XML in chunks to the sink callback, avoiding full XML materialization.
void StreamMeshXml(const std::vector<ThreeMfMeshResource>& resources, MeshXmlFormat format,
                   const ThreeMfDocument& document,
                   const std::function<void(std::string_view)>& sink);

// ZIP package assembly (implemented in three_mf_zip.cpp)
std::vector<uint8_t> BuildZipPackage(const std::vector<OpcPart>& parts,
                                     const ThreeMfExportOptions& options);

// Streaming ZIP writer that avoids full-content buffering for large entries.
// Supports two output modes: in-memory buffer or direct-to-file.
class StreamingZipWriter {
public:
    // Write to an in-memory buffer.
    explicit StreamingZipWriter(std::vector<uint8_t>& output, const ThreeMfExportOptions& options);
    // Write directly to a file (avoids holding the full ZIP in memory).
    explicit StreamingZipWriter(const std::string& file_path, const ThreeMfExportOptions& options);
    ~StreamingZipWriter();

    StreamingZipWriter(const StreamingZipWriter&)            = delete;
    StreamingZipWriter& operator=(const StreamingZipWriter&) = delete;

    void WriteWholeEntry(const std::string& path, const std::string& data);

    void BeginDeflateEntry(const std::string& path);
    void WriteChunk(const void* data, std::size_t len);
    void EndEntry();

    void Finalize();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ChromaPrint3D::detail
