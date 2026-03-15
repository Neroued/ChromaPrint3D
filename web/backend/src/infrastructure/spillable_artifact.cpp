#include "infrastructure/spillable_artifact.h"

#include <spdlog/spdlog.h>

#include <system_error>
#include <utility>

namespace chromaprint3d::backend {

namespace {

void RemoveTrackedFile(std::filesystem::path& path, const char* owner) noexcept {
    if (path.empty()) return;

    std::error_code ec;
    if (std::filesystem::remove(path, ec)) {
        spdlog::debug("{}: removed temp file {}", owner, path.string());
    } else if (ec) {
        spdlog::warn("{}: failed to remove {}: {}", owner, path.string(), ec.message());
    }
    path.clear();
}

} // namespace

SpillableArtifact::~SpillableArtifact() { Cleanup(); }

SpillableArtifact::SpillableArtifact(SpillableArtifact&& other) noexcept
    : path_(std::move(other.path_)), size_(other.size_) {
    other.path_.clear();
    other.size_ = 0;
}

SpillableArtifact& SpillableArtifact::operator=(SpillableArtifact&& other) noexcept {
    if (this != &other) {
        Cleanup();
        path_ = std::move(other.path_);
        size_ = other.size_;
        other.path_.clear();
        other.size_ = 0;
    }
    return *this;
}

SpillableArtifact SpillableArtifact::FromFile(std::filesystem::path path, std::size_t size) {
    SpillableArtifact a;
    a.path_ = std::move(path);
    a.size_ = size;
    return a;
}

void SpillableArtifact::Cleanup() noexcept {
    RemoveTrackedFile(path_, "SpillableArtifact");
    size_ = 0;
}

PendingArtifactFile::PendingArtifactFile(std::filesystem::path path) : path_(std::move(path)) {}

PendingArtifactFile::~PendingArtifactFile() { Cleanup(); }

PendingArtifactFile::PendingArtifactFile(PendingArtifactFile&& other) noexcept
    : path_(std::move(other.path_)) {
    other.path_.clear();
}

PendingArtifactFile& PendingArtifactFile::operator=(PendingArtifactFile&& other) noexcept {
    if (this != &other) {
        Cleanup();
        path_ = std::move(other.path_);
        other.path_.clear();
    }
    return *this;
}

SpillableArtifact PendingArtifactFile::Commit(std::size_t size) {
    SpillableArtifact artifact = SpillableArtifact::FromFile(std::move(path_), size);
    path_.clear();
    return artifact;
}

void PendingArtifactFile::Cleanup() noexcept { RemoveTrackedFile(path_, "PendingArtifactFile"); }

} // namespace chromaprint3d::backend
