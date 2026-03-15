#pragma once

#include <cstddef>
#include <filesystem>

namespace chromaprint3d::backend {

/// RAII wrapper for a temporary file backing a spilled artifact.
/// Deletes the underlying file on destruction.
class SpillableArtifact {
public:
    SpillableArtifact() = default;
    ~SpillableArtifact();

    SpillableArtifact(SpillableArtifact&& other) noexcept;
    SpillableArtifact& operator=(SpillableArtifact&& other) noexcept;

    SpillableArtifact(const SpillableArtifact&)            = delete;
    SpillableArtifact& operator=(const SpillableArtifact&) = delete;

    /// Take ownership of an existing temp file.
    static SpillableArtifact FromFile(std::filesystem::path path, std::size_t size);

    bool has_file() const { return !path_.empty(); }

    std::size_t file_size() const { return size_; }

    const std::filesystem::path& path() const { return path_; }

private:
    void Cleanup() noexcept;

    std::filesystem::path path_;
    std::size_t size_ = 0;
};

/// RAII guard for a temp file that has been created but not yet committed into a SpillableArtifact.
/// Deletes the file on destruction unless Commit() transfers ownership.
class PendingArtifactFile {
public:
    PendingArtifactFile() = default;
    explicit PendingArtifactFile(std::filesystem::path path);
    ~PendingArtifactFile();

    PendingArtifactFile(PendingArtifactFile&& other) noexcept;
    PendingArtifactFile& operator=(PendingArtifactFile&& other) noexcept;

    PendingArtifactFile(const PendingArtifactFile&)            = delete;
    PendingArtifactFile& operator=(const PendingArtifactFile&) = delete;

    bool has_path() const { return !path_.empty(); }

    const std::filesystem::path& path() const { return path_; }

    SpillableArtifact Commit(std::size_t size);

private:
    void Cleanup() noexcept;

    std::filesystem::path path_;
};

} // namespace chromaprint3d::backend
